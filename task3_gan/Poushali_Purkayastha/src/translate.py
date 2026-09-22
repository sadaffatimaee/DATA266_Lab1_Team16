import csv
import shutil
import time
import zipfile
from pathlib import Path

import numpy as np
import torch

from data import load_image, save_jpg, to_tensor, to_uint8
from models import build_generator


def load_generators(path, mcfg, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    g_ab, g_ba = build_generator(mcfg), build_generator(mcfg)
    g_ab.load_state_dict({k: v.float() for k, v in ckpt["g_ab"].items()})
    g_ba.load_state_dict({k: v.float() for k, v in ckpt["g_ba"].items()})
    g_ab.to(device).eval()
    g_ba.to(device).eval()
    return g_ab, g_ba, ckpt


@torch.no_grad()
def translate_paths(gen, paths, size, out_dir, device, batch_size, autocast_ctx):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    n = 0
    for i in range(0, len(paths), batch_size):
        chunk = paths[i : i + batch_size]
        arr = np.stack([load_image(p, size) for p in chunk])
        with autocast_ctx():
            out = gen(to_tensor(arr, device))
        for p, img in zip(chunk, to_uint8(out)):
            save_jpg(img, out_dir / f"{p.stem}.jpg")
            n += 1
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = max(time.time() - t0, 1e-9)
    return n, n / elapsed, elapsed


def write_preview(pred_dir, preview_dir, n):
    preview_dir = Path(preview_dir)
    if preview_dir.exists():
        shutil.rmtree(preview_dir)
    preview_dir.mkdir(parents=True)
    files = sorted(Path(pred_dir).glob("*.jpg"))[: int(n)]
    for f in files:
        shutil.copy(f, preview_dir / f.name)
    return len(files)


def write_submission(pred_dir, zip_path, csv_path):
    files = sorted(Path(pred_dir).glob("*.jpg"))
    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as z:
        for f in files:
            z.write(f, arcname=f.name)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "filename"])
        for i, p in enumerate(files):
            w.writerow([i, p.name])
    return len(files)
