"""Verify or symlink an existing RWF-2000 directory into ./data/RWF-2000."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "data" / "RWF-2000"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, type=Path,
                    help="Path to an existing RWF-2000 folder containing train/ and val/")
    args = ap.parse_args()

    src = args.data_dir.resolve()
    if not src.exists():
        sys.exit(f"Not found: {src}")
    if not (src / "train").exists() or not (src / "val").exists():
        sys.exit(f"{src} must contain train/ and val/ subfolders")

    TARGET.parent.mkdir(parents=True, exist_ok=True)
    if TARGET.exists() or TARGET.is_symlink():
        if TARGET.is_symlink() or not any(TARGET.iterdir()):
            TARGET.unlink() if TARGET.is_symlink() else TARGET.rmdir()
        else:
            sys.exit(f"{TARGET} already exists and is non-empty. Remove it first.")
    try:
        os.symlink(src, TARGET, target_is_directory=True)
        print(f"Linked {TARGET} -> {src}")
    except OSError:
        # Windows without symlink rights: fall back to printing the config edit
        print(f"Could not symlink. Edit configs/default.yaml -> data.root: {src}")


if __name__ == "__main__":
    main()
