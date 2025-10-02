#!/usr/bin/env python3
import argparse
import os
import zipfile
import urllib.request
from pathlib import Path


COCO_BASE = "http://images.cocodataset.org"
URLS = {
    "val_images": f"{COCO_BASE}/zips/val2017.zip",
    "train_images": f"{COCO_BASE}/zips/train2017.zip",
    "annotations": f"{COCO_BASE}/annotations/annotations_trainval2017.zip",
}


def download(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"✓ Exists: {dest}")
        return dest
    print(f"↓ Downloading: {url}\n→ {dest}")
    urllib.request.urlretrieve(url, dest)  # no resume; simple and reliable
    print(f"✓ Downloaded: {dest}")
    return dest


def unzip(zpath: Path, out_dir: Path, members: tuple[str, ...] | None = None):
    print(f"⇣ Extracting: {zpath} → {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zpath, 'r') as zf:
        if members:
            for m in members:
                for name in zf.namelist():
                    if name.endswith(m):
                        zf.extract(name, out_dir)
        else:
            zf.extractall(out_dir)
    print(f"✓ Extracted: {out_dir}")


def main():
    ap = argparse.ArgumentParser(description="Download COCO2017 images and annotations")
    ap.add_argument("--root", default="data/raw/coco2017", help="Root folder for COCO2017")
    ap.add_argument("--val", type=int, default=1, help="Download val2017 images (≈1GB)")
    ap.add_argument("--train", type=int, default=0, help="Download train2017 images (≈18GB)")
    ap.add_argument("--ann", type=int, default=1, help="Download annotations (≈250MB)")
    args = ap.parse_args()

    root = Path(args.root)
    zroot = root / "_zips"
    zroot.mkdir(parents=True, exist_ok=True)

    if args.ann:
        ann_zip = download(URLS["annotations"], zroot / "annotations_trainval2017.zip")
        unzip(ann_zip, root)
        # Move instances files to expected location if needed
        src_dir = root / "annotations"
        (root / "annotations").mkdir(parents=True, exist_ok=True)
        for fn in ("instances_train2017.json", "instances_val2017.json"):
            p = src_dir / fn
            if p.exists():
                # already in place after extract
                pass

    if args.val:
        val_zip = download(URLS["val_images"], zroot / "val2017.zip")
        unzip(val_zip, root)
        src = root / "val2017"
        dst = root / "images" / "val2017"
        dst.mkdir(parents=True, exist_ok=True)
        # Move files into configured layout if different
        if src.exists() and src != dst:
            for name in os.listdir(src):
                os.replace(src / name, dst / name)
            try:
                os.rmdir(src)
            except OSError:
                pass

    if args.train:
        tr_zip = download(URLS["train_images"], zroot / "train2017.zip")
        unzip(tr_zip, root)
        src = root / "train2017"
        dst = root / "images" / "train2017"
        dst.mkdir(parents=True, exist_ok=True)
        if src.exists() and src != dst:
            for name in os.listdir(src):
                os.replace(src / name, dst / name)
            try:
                os.rmdir(src)
            except OSError:
                pass

    print("All done. Verify paths in configs/v3.yaml point to:\n"
          f"  images.val   → {root/'images'/'val2017'}\n"
          f"  images.train → {root/'images'/'train2017'}\n"
          f"  annotations  → {root/'annotations'}")


if __name__ == "__main__":
    main()

