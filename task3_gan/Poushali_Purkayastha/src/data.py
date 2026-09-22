import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

EXTS = {".jpg", ".jpeg", ".png"}


def list_images(folder):
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in EXTS)


def load_image(path, size):
    img = Image.open(path).convert("RGB")
    if img.size != (size, size):
        img = img.resize((size, size), Image.BICUBIC)
    return np.asarray(img, dtype=np.uint8)


def save_jpg(arr, path, quality=95):
    Image.fromarray(arr).save(path, quality=quality)


def cache_folder(folder, size, cache_path, max_images=None):
    paths = list_images(folder)
    if max_images:
        paths = paths[: int(max_images)]
    names = [p.name for p in paths]
    key = hashlib.md5(("|".join(names) + f"|{size}").encode("utf-8")).hexdigest()
    cache_path = Path(cache_path)
    meta_path = cache_path.with_suffix(".json")
    if cache_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("key") == key:
            return np.load(cache_path), paths
    arr = np.zeros((len(paths), size, size, 3), dtype=np.uint8)
    for i, p in enumerate(paths):
        arr[i] = load_image(p, size)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, arr)
    meta = {"key": key, "folder": str(folder), "count": len(names), "size": size, "files": names}
    meta_path.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return arr, paths


def to_tensor(batch, device):
    x = torch.from_numpy(np.ascontiguousarray(batch)).to(device).permute(0, 3, 1, 2).float()
    return x / 127.5 - 1.0


def to_uint8(x):
    x = x.detach().float().clamp(-1, 1)
    return ((x + 1) * 127.5).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()


def split_holdout(n, n_holdout, rng):
    n_holdout = min(int(n_holdout), max(0, n - 2))
    perm = rng.permutation(n)
    return np.sort(perm[n_holdout:]), np.sort(perm[:n_holdout])
