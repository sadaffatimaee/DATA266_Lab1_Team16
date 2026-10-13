import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))
MEMBER_DIR = SRC_DIR.parent

from models import build_model
from preprocess import UNK_IDX, Preprocessor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--run", default="full")
    args = ap.parse_args()
    cfg = yaml.safe_load((SRC_DIR / "configs" / f"{args.run}.yaml").read_text(encoding="utf-8"))
    data_dir = MEMBER_DIR / "data_processed" / args.run
    vocab = json.loads((data_dir / "vocab.json").read_text(encoding="utf-8"))
    stoi = vocab["stoi"]
    d = cfg["data"]
    pre = Preprocessor(bool(d["remove_stopwords"]), bool(d["keep_negations"]), bool(d["stem"]))
    tokens = pre(args.text)[: int(d["max_len"])]
    ids = [stoi.get(t, UNK_IDX) for t in tokens] or [UNK_IDX]
    print(f"tokens after cleaning ({len(tokens)}): {' '.join(tokens)}\n")
    device = torch.device("cpu")
    x = torch.tensor([ids], dtype=torch.long)
    length = torch.tensor([len(ids)], dtype=torch.long)
    for name, mcfg in cfg["models"].items():
        ckpt = torch.load(MEMBER_DIR / "checkpoints" / args.run / name / "best.pt", map_location=device, weights_only=False)
        model = build_model(mcfg, ckpt["vocab_size"])
        model.load_state_dict(ckpt["model"])
        model.eval()
        with torch.no_grad():
            p = torch.sigmoid(model(x, length)).item()
        label = "positive" if p >= 0.5 else "negative"
        print(f"{name:9s} p(positive) = {p:.3f}  ->  {label}  (best epoch {ckpt['epoch']})")


if __name__ == "__main__":
    main()
