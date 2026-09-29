import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scipy.linalg
import torch
from PIL import Image

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))
MEMBER_DIR = SRC_DIR.parent
REPO_DIR = MEMBER_DIR.parents[1]

from data import list_images
from metrics import InceptionExtractor

MU_KEYS = ["mu", "mean", "mu_real", "real_mu", "mu1"]
SIGMA_KEYS = ["sigma", "cov", "covariance", "sigma_real", "real_sigma", "sigma1"]
FEATURE_KEYS = [
    "feats_real", "features_real", "real_features", "real_feats", "features", "feats",
    "acts_real", "real_activations", "activations", "acts", "embeddings", "real",
]


def pick(stats, names):
    for n in names:
        if n in stats:
            return stats[n]
    return None


def fid_from_stats(mu1, s1, mu2, s2):
    diff = mu1 - mu2
    covmean, _ = scipy.linalg.sqrtm(s1 @ s2, disp=False)
    if not np.isfinite(covmean).all():
        eps = np.eye(s1.shape[0]) * 1e-6
        covmean = scipy.linalg.sqrtm((s1 + eps) @ (s2 + eps))
    return float(diff @ diff + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean.real))


def normalize(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)


def check_images(paths, limit=50):
    problems = []
    for p in paths[:limit]:
        with Image.open(p) as im:
            if im.size != (256, 256) or im.mode != "RGB" or im.format != "JPEG":
                problems.append(f"{p.name}: {im.size} {im.mode} {im.format}")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generated", default=str(MEMBER_DIR / "outputs" / "full" / "pred_A2B"))
    ap.add_argument("--real-stats", default=str(REPO_DIR / "task3_gan" / "data" / "real_stats.npz"))
    ap.add_argument("--real-dir", default=str(REPO_DIR / "task3_gan" / "data" / "monet_jpg"))
    ap.add_argument("--out", default=str(MEMBER_DIR / "outputs" / "full" / "kaggle" / "official_scores.json"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=50)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen_paths = list_images(args.generated)
    if not gen_paths:
        raise SystemExit(f"no images in {args.generated}")
    problems = check_images(gen_paths)
    if problems:
        print("images that break the 256x256 RGB JPG rule:", *problems[:10], sep="\n  ")
    inception = InceptionExtractor(device)
    print(f"extracting features for {len(gen_paths)} generated images on {device}")
    gen_feats = inception.from_paths(gen_paths, 256, args.batch)

    stats = {}
    stats_path = Path(args.real_stats)
    if stats_path.exists():
        loaded = np.load(stats_path, allow_pickle=True)
        stats = {k: loaded[k] for k in loaded.files}
        print("real_stats.npz keys:", {k: getattr(v, "shape", None) for k, v in stats.items()})
    mu = pick(stats, MU_KEYS)
    sigma = pick(stats, SIGMA_KEYS)
    real_feats = pick(stats, FEATURE_KEYS)
    real_source = "real_stats.npz"
    inception_match = None
    if real_feats is None or real_feats.ndim != 2:
        real_paths = list_images(args.real_dir)
        print(f"extracting features for {len(real_paths)} real images from {args.real_dir}")
        real_feats = inception.from_paths(real_paths, 256, args.batch)
        real_source = str(args.real_dir)
    elif Path(args.real_dir).exists():
        real_paths = list_images(args.real_dir)
        own = normalize(inception.from_paths(real_paths, 256, args.batch).astype(np.float64))
        theirs = normalize(np.asarray(real_feats, dtype=np.float64))
        inception_match = float(np.mean((own @ theirs.T).max(axis=1)))
        print(f"cosine match between our Inception features and the course's feats_real: {inception_match:.4f} (1.0 means identical network and preprocessing)")
    if mu is None or sigma is None:
        mu = real_feats.mean(axis=0)
        sigma = np.cov(real_feats, rowvar=False)
        stats_source = real_source
    else:
        stats_source = "real_stats.npz"
    mu = np.asarray(mu, dtype=np.float64).reshape(-1)
    sigma = np.asarray(sigma, dtype=np.float64)

    fid = fid_from_stats(gen_feats.mean(axis=0), np.cov(gen_feats, rowvar=False), mu, sigma)

    rng = np.random.default_rng(args.seed)
    n = min(len(gen_feats), len(real_feats))
    gi = rng.choice(len(gen_feats), n, replace=False)
    ri = rng.choice(len(real_feats), n, replace=False)
    g = normalize(gen_feats[gi].astype(np.float64))
    r = normalize(real_feats[ri].astype(np.float64))
    mifid_paired = float(np.mean(1.0 - np.sum(g * r, axis=1)))
    all_g = normalize(gen_feats.astype(np.float64))
    all_r = normalize(real_feats.astype(np.float64))
    mifid_all_pairs = float(np.mean(1.0 - all_g @ all_r.T))

    result = {
        "FID": fid,
        "MiFID": mifid_paired,
        "score": (fid + mifid_paired) / 2,
        "MiFID_all_pairs": mifid_all_pairs,
        "n_generated": int(len(gen_feats)),
        "n_real": int(len(real_feats)),
        "subsample_size": int(n),
        "seed": args.seed,
        "fid_stats_source": stats_source,
        "real_feature_source": real_source,
        "inception_match_cosine": inception_match,
        "real_stats_keys": {k: list(getattr(v, "shape", [])) for k, v in stats.items()},
        "inception": "torchvision inception_v3 IMAGENET1K_V1 pool features, 299 px bilinear",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"written {out}")


if __name__ == "__main__":
    main()
