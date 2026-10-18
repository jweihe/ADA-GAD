"""Fetch the official Python 3.12/Linux CUDA 12.1 PyG wheels by fixed hashes."""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
destination = ROOT / "cache/wheels"
destination.mkdir(parents=True, exist_ok=True)
for record in json.loads((ROOT / "wheels.lock.json").read_text())["files"]:
    path = destination / record["filename"]
    if path.exists():
        if path.stat().st_size != record["bytes"] or hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError("Existing wheel differs from the fixed manifest: " + path.name)
        continue
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with urllib.request.urlopen(record["source"], timeout=120) as response, temporary.open("wb") as stream:
            count = 0
            while chunk := response.read(1024 * 1024):
                count += len(chunk)
                if count > record["bytes"]:
                    raise ValueError("Oversized wheel download")
                stream.write(chunk)
        if count != record["bytes"] or hashlib.sha256(temporary.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError("Wheel does not match its official pinned bytes")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
print("Verified three fixed official extension wheels")
