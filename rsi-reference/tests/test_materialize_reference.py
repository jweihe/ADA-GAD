"""Boundary checks for the trusted reference-delivery interface."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile

import pytest


spec = importlib.util.spec_from_file_location(
    "materializer", Path(__file__).resolve().parents[1] / "scripts/materialize_reference.py")
materializer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(materializer)


def locked_archive(tmp_path, monkeypatch, names, kind=tarfile.REGTYPE):
    archive = tmp_path / "reference.tar.gz"
    payload = b"frozen reference bytes"
    with tarfile.open(archive, "w:gz") as stream:
        for name in names:
            info = tarfile.TarInfo(name)
            info.type = kind
            info.linkname = "outside" if kind == tarfile.SYMTYPE else ""
            info.size = len(payload) if kind == tarfile.REGTYPE else 0
            stream.addfile(info, io.BytesIO(payload) if info.isfile() else None)
    lock = {"schema_version": 1, "bytes": archive.stat().st_size,
            "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            "files": {name: {"bytes": len(payload),
                              "sha256": hashlib.sha256(payload).hexdigest()} for name in names}}
    (tmp_path / "reference.lock.json").write_text(json.dumps(lock))
    monkeypatch.setattr(materializer, "ROOT", tmp_path)
    return archive, tmp_path / "prepared", payload


def test_materializes_exact_locked_bytes_without_training(tmp_path, monkeypatch):
    archive, destination, payload = locked_archive(tmp_path, monkeypatch, ["candidate/weights.npz"])
    result = materializer.materialize(destination, archive)
    assert (destination / "candidate/weights.npz").read_bytes() == payload
    assert result["training_performed"] is False


def test_rejects_corrupted_archive(tmp_path, monkeypatch):
    archive, destination, _ = locked_archive(tmp_path, monkeypatch, ["weights.npz"])
    archive.write_bytes(archive.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="locked bytes"):
        materializer.materialize(destination, archive)
    assert not destination.exists()


@pytest.mark.parametrize("name", ["../outside", "/absolute", "candidate/../outside"])
def test_rejects_traversal_even_in_a_pinned_archive(tmp_path, monkeypatch, name):
    archive, destination, _ = locked_archive(tmp_path, monkeypatch, [name])
    with pytest.raises(ValueError, match="Invalid reference archive member"):
        materializer.materialize(destination, archive)
    assert not destination.exists()


def test_rejects_links(tmp_path, monkeypatch):
    archive, destination, _ = locked_archive(tmp_path, monkeypatch, ["weights.npz"], tarfile.SYMTYPE)
    with pytest.raises(ValueError, match="Invalid reference archive member"):
        materializer.materialize(destination, archive)


def test_rejects_duplicate_members(tmp_path, monkeypatch):
    archive, destination, _ = locked_archive(tmp_path, monkeypatch, ["weights.npz", "weights.npz"])
    with pytest.raises(ValueError, match="duplicate members"):
        materializer.materialize(destination, archive)


def test_preserves_existing_destination(tmp_path, monkeypatch):
    archive, destination, _ = locked_archive(tmp_path, monkeypatch, ["weights.npz"])
    destination.mkdir()
    marker = destination / "existing"
    marker.write_text("preserve")
    with pytest.raises(FileExistsError):
        materializer.materialize(destination, archive)
    assert marker.read_text() == "preserve"
