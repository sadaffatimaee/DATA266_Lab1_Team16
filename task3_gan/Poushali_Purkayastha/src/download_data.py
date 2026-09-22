import argparse
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
REPO_DIR = SRC_DIR.parents[2]
DATA_DIR = REPO_DIR / "task3_gan" / "data"
BERKELEY_URL = "https://efrosgans.eecs.berkeley.edu/cyclegan/datasets/monet2photo.zip"


def from_kaggle(competition):
    subprocess.run([sys.executable, "-m", "kaggle", "competitions", "download", "-c", competition, "-p", str(DATA_DIR)], check=True)
    for z in DATA_DIR.glob("*.zip"):
        with zipfile.ZipFile(z) as zf:
            zf.extractall(DATA_DIR)
        z.unlink()


def from_berkeley():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DATA_DIR / "monet2photo.zip"
    if not zip_path.exists():
        print("downloading", BERKELEY_URL)
        urllib.request.urlretrieve(BERKELEY_URL, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(DATA_DIR)
    (DATA_DIR / "monet_jpg").mkdir(exist_ok=True)
    (DATA_DIR / "photo_jpg").mkdir(exist_ok=True)
    for sub, dest in (("trainA", "monet_jpg"), ("testA", "monet_jpg"), ("trainB", "photo_jpg"), ("testB", "photo_jpg")):
        for f in (DATA_DIR / "monet2photo" / sub).iterdir():
            if f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                shutil.move(str(f), DATA_DIR / dest / f.name)
    shutil.rmtree(DATA_DIR / "monet2photo")
    zip_path.unlink()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["kaggle", "berkeley"], default="kaggle")
    ap.add_argument("--competition", default="data-266-fall-2026-gan-image-style-transfer")
    args = ap.parse_args()
    if args.source == "kaggle":
        from_kaggle(args.competition)
    else:
        from_berkeley()
    for name in ("monet_jpg", "photo_jpg"):
        d = DATA_DIR / name
        n = len([p for p in d.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]) if d.exists() else 0
        print(f"{name}: {n} images")


if __name__ == "__main__":
    main()
