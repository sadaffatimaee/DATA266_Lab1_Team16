import argparse
import sys
from pathlib import Path

import torch

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

from generate import load_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="Once upon a time")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--tokens", type=int, default=300)
    ap.add_argument("--checkpoint", default=str(SRC_DIR.parent / "checkpoints" / "full" / "final.pt"))
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tok, cfg = load_model(args.checkpoint, device)
    torch.manual_seed(20)
    idx = torch.tensor([tok.encode(args.prompt)], dtype=torch.long, device=device)
    temp = None if args.temperature <= 0 else args.temperature
    out = model.generate(idx, args.tokens, temperature=temp)
    mode = "greedy" if temp is None else f"temperature {temp}"
    print(f"[{model.count_params():,} parameters, {cfg['model']['n_layer']} layers, context {cfg['data']['block_size']}, {mode}]\n")
    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
