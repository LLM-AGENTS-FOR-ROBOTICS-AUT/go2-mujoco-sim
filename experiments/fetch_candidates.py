"""Download exact candidate models/configs for repeatable, simulation-only trials."""

import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]


def main():
    sources = json.loads(Path(__file__).with_name("sources.json").read_text())
    for source in sources:
        dest = ROOT / source["path"]
        expected = source["sha256"]
        if dest.is_file() and hashlib.sha256(dest.read_bytes()).hexdigest() == expected:
            continue
        with urlopen(source["url"], timeout=60) as response:
            content = response.read()
        if hashlib.sha256(content).hexdigest() != expected:
            raise RuntimeError(f"Checksum mismatch: {source['url']}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)
        print(f"Verified {dest.name}")


if __name__ == "__main__":
    main()
