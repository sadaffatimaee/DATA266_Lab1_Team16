import json
import time
from pathlib import Path

import torch

from data import CharTokenizer
from model import CharGPT


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    tok = CharTokenizer(ckpt["chars"])
    model = CharGPT(vocab_size=tok.vocab_size, block_size=cfg["data"]["block_size"], **cfg["model"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, tok, cfg


def generate_samples(model, tok, cfg, device, logger):
    g = cfg["generation"]
    torch.manual_seed(cfg["seed"])
    modes = [("greedy", None)] + [(f"temperature_{t}", float(t)) for t in g["temperatures"]]
    samples = []
    total_tokens = 0
    total_time = 0.0
    for prompt in g["prompts"]:
        for mode, temp in modes:
            n = 1 if temp is None else int(g["n_per_prompt"])
            idx = torch.tensor([tok.encode(prompt)] * n, dtype=torch.long, device=device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            out = model.generate(idx, g["max_new_tokens"], temperature=temp)
            if device.type == "cuda":
                torch.cuda.synchronize()
            total_time += time.time() - t0
            total_tokens += n * g["max_new_tokens"]
            for row in out.tolist():
                text = tok.decode(row)
                samples.append(
                    {
                        "prompt": prompt,
                        "mode": mode,
                        "temperature": temp,
                        "text": text,
                        "continuation": text[len(prompt) :],
                    }
                )
    tps = total_tokens / total_time if total_time > 0 else 0.0
    logger.info("generated %d samples, %d tokens, %.0f tokens/sec", len(samples), total_tokens, tps)
    return samples, tps


def save_samples(samples, out_dir):
    out_dir = Path(out_dir)
    (out_dir / "samples.json").write_text(json.dumps(samples, indent=1, ensure_ascii=False), encoding="utf-8")
    lines = []
    for i, s in enumerate(samples, 1):
        lines.append(f"=== sample {i} | prompt: {s['prompt']!r} | mode: {s['mode']} ===")
        lines.append(s["text"])
        lines.append("")
    (out_dir / "samples.txt").write_text("\n".join(lines), encoding="utf-8")
