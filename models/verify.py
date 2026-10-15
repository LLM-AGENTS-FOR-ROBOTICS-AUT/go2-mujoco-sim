"""Verify either existing model's bytes against catalog.json; no downloads."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS = json.loads((ROOT / "catalog.json").read_text())["models"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--weights", required=True)
    args = parser.parse_args()
    spec = MODELS[args.model]
    path = Path(args.weights).expanduser().resolve(strict=True)
    if not path.is_file() or path.stat().st_size != spec["weights_size_bytes"]:
        raise ValueError("Selected model size differs from its pinned catalog entry.")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != spec["weights_sha256"]:
        raise ValueError("Selected model checksum mismatch; no substitution performed.")
    print(json.dumps({"model": spec["alias"], "size_bytes": path.stat().st_size,
                      "sha256": digest, "verified": True}))


if __name__ == "__main__":
    main()
