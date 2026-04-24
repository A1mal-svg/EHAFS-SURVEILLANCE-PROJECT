"""Helper to obtain the RWF-2000 dataset.

The dataset is NOT redistributable with this repo. This script:
  1. Prints the official sources.
  2. Optionally extracts a local archive (.zip / .rar / .tar.gz) into data/RWF-2000.
  3. Verifies the resulting folder structure.

Usage:
  python scripts/download_rwf2000.py                        # show instructions
  python scripts/download_rwf2000.py --extract /path/RWF-2000.zip
  python scripts/download_rwf2000.py --verify
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "RWF-2000"

INSTRUCTIONS = """
================================================================
  RWF-2000 dataset (Cheng et al., ICCVW 2020)
================================================================

The dataset is hosted by the original authors. Download from one of:

  1. Official GitHub (request form):
     https://github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection

  2. Kaggle mirror (requires Kaggle account):
     https://www.kaggle.com/datasets/vulamnguyen/rwf2000

  3. Papers-with-Code:
     https://paperswithcode.com/dataset/rwf-2000

Expected layout AFTER extraction:

  data/RWF-2000/
    train/
      Fight/      (~800 .avi)
      NonFight/   (~800 .avi)
    val/
      Fight/      (~200 .avi)
      NonFight/   (~200 .avi)

If your download contains 'fight'/'nonFight' or different casing, this
script will normalize folder names. Re-run with --verify to check.

Once you have the archive, run:
  python scripts/download_rwf2000.py --extract /path/to/RWF-2000.zip
"""


def extract_archive(archive: Path) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {archive} -> {DATA_DIR}")
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as z:
            z.extractall(DATA_DIR)
    elif archive.suffixes[-2:] == [".tar", ".gz"] or archive.suffix == ".tgz":
        with tarfile.open(archive, "r:gz") as t:
            t.extractall(DATA_DIR)
    elif archive.suffix == ".tar":
        with tarfile.open(archive, "r:") as t:
            t.extractall(DATA_DIR)
    else:
        sys.exit(f"Unsupported archive type: {archive.suffix}. Extract manually into {DATA_DIR}.")
    normalize_layout()


def normalize_layout() -> None:
    """Walk DATA_DIR and try to flatten common nested layouts."""
    if not DATA_DIR.exists():
        return
    # Some archives nest one extra folder
    children = [p for p in DATA_DIR.iterdir() if p.is_dir()]
    if len(children) == 1 and not (DATA_DIR / "train").exists():
        inner = children[0]
        for sub in inner.iterdir():
            shutil.move(str(sub), str(DATA_DIR / sub.name))
        inner.rmdir()

    # normalize casing
    rename_map = {"fight": "Fight", "nonfight": "NonFight", "non_fight": "NonFight",
                  "non-fight": "NonFight", "training": "train", "validation": "val", "test": "val"}
    for split_dir in DATA_DIR.iterdir():
        if not split_dir.is_dir():
            continue
        target_split = rename_map.get(split_dir.name.lower(), split_dir.name)
        if target_split != split_dir.name:
            split_dir.rename(DATA_DIR / target_split)
    for split_dir in DATA_DIR.iterdir():
        if not split_dir.is_dir():
            continue
        for cls_dir in split_dir.iterdir():
            if not cls_dir.is_dir():
                continue
            target = rename_map.get(cls_dir.name.lower(), cls_dir.name)
            if target != cls_dir.name:
                cls_dir.rename(split_dir / target)


def verify() -> bool:
    ok = True
    expected = {
        "train/Fight": 600,
        "train/NonFight": 600,
        "val/Fight": 100,
        "val/NonFight": 100,
    }
    print(f"Verifying layout under {DATA_DIR}\n")
    for rel, min_count in expected.items():
        d = DATA_DIR / rel
        if not d.exists():
            print(f"  ✗ MISSING: {rel}")
            ok = False
            continue
        videos = [p for p in d.iterdir() if p.suffix.lower() in (".avi", ".mp4", ".mov", ".mkv")]
        flag = "✓" if len(videos) >= min_count else "⚠"
        print(f"  {flag} {rel:20s}  {len(videos)} videos  (expected ~{min_count})")
        if len(videos) == 0:
            ok = False
    print("\n" + ("Layout looks good." if ok else "Layout incomplete — see notes above."))
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description="RWF-2000 download/setup helper")
    ap.add_argument("--extract", type=Path, help="Local archive to extract into data/RWF-2000/")
    ap.add_argument("--verify", action="store_true", help="Verify expected layout")
    args = ap.parse_args()

    if not args.extract and not args.verify:
        print(INSTRUCTIONS)
        return
    if args.extract:
        if not args.extract.exists():
            sys.exit(f"Archive not found: {args.extract}")
        extract_archive(args.extract)
    verify()


if __name__ == "__main__":
    main()
