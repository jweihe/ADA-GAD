"""Download and verify the existing frozen reference; perform no training."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
import urllib.request


ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def materialize(destination, archive=None):
    lock = json.loads((ROOT / "reference.lock.json").read_text())
    if lock["schema_version"] != 1:
        raise ValueError("Unknown reference lock version")
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError("Use a fresh destination for the frozen reference")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".reference-", dir=destination.parent) as temporary:
        temporary = Path(temporary)
        if archive is None:
            archive = temporary / "reference.tar.gz"
            request = urllib.request.Request(lock["url"], headers={"User-Agent": "ADA-GAD-reference"})
            received = 0
            with urllib.request.urlopen(request, timeout=120) as response, archive.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    received += len(chunk)
                    if received > lock["bytes"]:
                        raise ValueError("Reference download exceeds its locked size")
                    output.write(chunk)
        archive = archive.resolve()
        if archive.stat().st_size != lock["bytes"] or digest(archive) != lock["sha256"]:
            raise ValueError("Reference archive differs from its locked bytes")
        stage = temporary / "extracted"
        stage.mkdir()
        with tarfile.open(archive, "r:gz") as bundle:
            members = bundle.getmembers()
            names = [member.name for member in members]
            if len(names) != len(set(names)) or set(names) != set(lock["files"]):
                raise ValueError("Reference archive has unexpected or duplicate members")
            for member in members:
                relative = PurePosixPath(member.name)
                if (not member.isfile() or relative.is_absolute() or ".." in relative.parts
                        or member.name != relative.as_posix()):
                    raise ValueError("Invalid reference archive member")
                expected = lock["files"][member.name]
                if member.size != expected["bytes"]:
                    raise ValueError("Reference member differs from its locked size")
                target = stage / member.name
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.extractfile(member) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
                if digest(target) != expected["sha256"]:
                    raise ValueError("Reference member hash mismatch")
        stage.rename(destination)
    return {"status": "materialized", "destination": str(destination),
            "archive_sha256": lock["sha256"], "files_verified": len(lock["files"]),
            "training_performed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--archive", type=Path, help="Use an already staged archive without network")
    args = parser.parse_args()
    print(json.dumps(materialize(args.destination, args.archive)))


if __name__ == "__main__":
    main()
