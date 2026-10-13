import argparse
import contextlib
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))
MEMBER_DIR = SRC_DIR.parent

from data import load_image, to_tensor, to_uint8
from translate import load_generators


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--direction", choices=["photo2monet", "monet2photo"], default="photo2monet")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load((SRC_DIR / "configs" / "full.yaml").read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g_ab, g_ba, ckpt = load_generators(MEMBER_DIR / "checkpoints" / "full" / "generators.pt", cfg["model"], device)
    gen = g_ab if args.direction == "photo2monet" else g_ba
    src = load_image(args.image, args.size)
    with torch.no_grad():
        out = to_uint8(gen(to_tensor(np.array(src[None]), device)))[0]
    side = np.concatenate([src, out], axis=1)
    out_path = Path(args.out) if args.out else Path(args.image).with_name(Path(args.image).stem + f"_{args.direction}.jpg")
    Image.fromarray(side).save(out_path, quality=95)
    print(f"{args.direction} with the epoch {ckpt['epoch']} generator on {device}; input left, translation right: {out_path}")


if __name__ == "__main__":
    main()
