"""Fetch the pinned Go2 Walk These Ways weights; no robot connection."""

from __future__ import annotations

import hashlib
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent
POLICY_DIR = ROOT / "_deps" / "go2-policy"
REVISION = "2ee4388672fa121eb6f827984dfdeb16e415fabe"
BASE_URL = f"https://raw.githubusercontent.com/lupinjia/go2_deploy/{REVISION}/models"
FILES = {
    "wtw_model.pt": "946a1631a05e8c0d7d94b79a8c1bbed73bf68adffa504e2ddba8188fbd10bf19",
}


def verified(path: Path, digest: str) -> bool:
    return path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == digest


def check_assets() -> None:
    for name, digest in FILES.items():
        if not verified(POLICY_DIR / name, digest):
            raise RuntimeError(
                f"Missing or changed policy file: {name}. "
                "Run macOS setup with --locomotion or Windows setup with -Locomotion."
            )


def download() -> None:
    POLICY_DIR.mkdir(parents=True, exist_ok=True)
    for name, digest in FILES.items():
        dest = POLICY_DIR / name
        if verified(dest, digest):
            print(f"Policy file already verified: {name}")
            continue
        print(f"Downloading {name}...")
        with urlopen(f"{BASE_URL}/{name}", timeout=60) as response:
            content = response.read()
        if hashlib.sha256(content).hexdigest() != digest:
            raise RuntimeError(f"Checksum mismatch for {name}; download was not installed.")
        temporary = dest.with_suffix(dest.suffix + ".download")
        temporary.write_bytes(content)
        temporary.replace(dest)
    check_assets()


if __name__ == "__main__":
    download()
