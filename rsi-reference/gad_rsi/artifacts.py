"""Bounded, inert NumPy weights for a model schema supplied by the evaluator."""
import math
import os
from pathlib import Path
import stat
import tempfile
import zipfile

import numpy as np


HEADER_LIMIT = 4096


def load_weights(path, expected):
    """Load exactly the expected arrays without loading pickle or candidate code.

    ``expected`` maps trusted state names to arrays from the fixed model. Sizes,
    shapes and dtypes are checked before allocating candidate array storage.
    """
    schemas = {name: (value.shape, value.dtype) for name, value in expected.items()}
    names = {name + ".npy": name for name in schemas}
    total_bytes = sum(math.prod(shape) * dtype.itemsize for shape, dtype in schemas.values())
    archive_limit = total_bytes + len(schemas) * (HEADER_LIMIT + 4096) + 65536
    flags = os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, "rb") as file:
        info = os.fstat(file.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > archive_limit:
            raise ValueError("Weights must be a bounded regular archive")
        with zipfile.ZipFile(file) as archive:
            members = archive.infolist()
            filenames = [member.filename for member in members]
            if len(filenames) != len(names) or set(filenames) != set(names):
                raise ValueError("Weight names must match the trusted model exactly, without duplicates")
            arrays = {}
            for member in members:
                key = names[member.filename]
                shape, dtype = schemas[key]
                count = math.prod(shape)
                nbytes = count * dtype.itemsize
                if (member.flag_bits & 1 or member.compress_type not in {
                        zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED} or
                        member.file_size > nbytes + HEADER_LIMIT + 12):
                    raise ValueError("Unsupported or oversized weight member")
                with archive.open(member) as stream:
                    version = np.lib.format.read_magic(stream)
                    if version == (1, 0):
                        header = np.lib.format.read_array_header_1_0(stream, max_header_size=HEADER_LIMIT)
                    elif version == (2, 0):
                        header = np.lib.format.read_array_header_2_0(stream, max_header_size=HEADER_LIMIT)
                    else:
                        raise ValueError("Unsupported NumPy header version")
                    found_shape, fortran, found_dtype = header
                    if (found_shape != shape or found_dtype != dtype or found_dtype.hasobject
                            or found_dtype.fields is not None or fortran):
                        raise ValueError("Weight shape, dtype or memory order differs from the trusted model")
                    if member.file_size != stream.tell() + nbytes:
                        raise ValueError("Weight member size does not match its header")
                    payload = stream.read(nbytes)
                    if len(payload) != nbytes or stream.read(1):
                        raise ValueError("Truncated weight payload or trailing data")
                    value = np.frombuffer(payload, dtype=dtype).reshape(shape)
                    if not np.isfinite(value).all():
                        raise ValueError("Weights must be finite")
                    arrays[key] = value.copy()
    return arrays


def write_weights(path, state):
    """Atomically export an operator-owned model state as ordinary NumPy arrays."""
    arrays = {}
    for name, value in state.items():
        value = np.asarray(value)
        if (not isinstance(name, str) or not name or "/" in name or "\\" in name
                or value.dtype.hasobject or value.dtype.fields is not None
                or not np.issubdtype(value.dtype, np.number) or not np.isfinite(value).all()):
            raise ValueError("Invalid export name or array")
        arrays[name] = np.ascontiguousarray(value) if value.ndim else value.copy()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", delete=False) as file:
            temporary = Path(file.name)
            np.savez(file, **arrays)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
