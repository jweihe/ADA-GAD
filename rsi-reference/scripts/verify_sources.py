"""Check byte-exact source provenance without importing any upstream code."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--manifest", default="sources.lock.json")
args = parser.parse_args()
manifest = json.loads((ROOT / args.manifest).read_text())
for record in manifest["files"]:
    data = (ROOT / record["local_path"]).read_bytes()
    actual = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
    if actual != record["blob_sha"]:
        raise SystemExit("Source modified: " + record["local_path"])
print("Verified", len(manifest["files"]), "pinned source files")
