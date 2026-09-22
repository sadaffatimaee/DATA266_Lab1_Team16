import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import scipy.linalg
import torch
import torch.nn as nn
import torch.nn.functional as F

from data import list_images, load_image, to_tensor, to_uint8


class InceptionExtractor:
    def __init__(self, device):
        from torchvision.models import Inception_V3_Weights, inception_v3

        self.model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Identity()
        self.model.eval().to(device)
        self.device = device
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def batch(self, arr):
        x = torch.from_numpy(np.ascontiguousarray(arr)).to(self.device).permute(0, 3, 1, 2).float() / 255.0
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        return self.model(x).float().cpu().numpy()

    def from_arrays(self, arr, batch_size):
        return np.concatenate([self.batch(arr[i : i + batch_size]) for i in range(0, len(arr), batch_size)])

    def from_paths(self, paths, size, batch_size):
        feats = []
        for i in range(0, len(paths), batch_size):
            feats.append(self.batch(np.stack([load_image(p, size) for p in paths[i : i + batch_size]])))
        return np.concatenate(feats)


class LPIPSMetric:
    def __init__(self, net, device):
        import lpips

        self.fn = lpips.LPIPS(net=net, verbose=False).to(device).eval()

    @torch.no_grad()
    def __call__(self, x, y):
        return self.fn(x, y).flatten().float().cpu().numpy()


def fid_from_features(f1, f2):
    mu1, mu2 = f1.mean(axis=0), f2.mean(axis=0)
    s1 = np.cov(f1, rowvar=False)
    s2 = np.cov(f2, rowvar=False)
    diff = mu1 - mu2
    covmean, _ = scipy.linalg.sqrtm(s1 @ s2, disp=False)
    if not np.isfinite(covmean).all():
        eps = np.eye(s1.shape[0]) * 1e-6
        covmean = scipy.linalg.sqrtm((s1 + eps) @ (s2 + eps))
    covmean = covmean.real
    return float(diff @ diff + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean))


def kid_from_features(f1, f2, n_subsets, subset_size, rng):
    d = f1.shape[1]
    m = int(min(subset_size, len(f1), len(f2)))
    vals = []
    for _ in range(int(n_subsets)):
        x = f1[rng.choice(len(f1), m, replace=False)]
        y = f2[rng.choice(len(f2), m, replace=False)]
        kxx = (x @ x.T / d + 1) ** 3
        kyy = (y @ y.T / d + 1) ** 3
        kxy = (x @ y.T / d + 1) ** 3
        mmd = (kxx.sum() - np.trace(kxx)) / (m * (m - 1)) + (kyy.sum() - np.trace(kyy)) / (m * (m - 1)) - 2 * kxy.mean()
        vals.append(mmd)
    return float(np.mean(vals)), float(np.std(vals))


def manifold_metrics(real, fake, k):
    real_t = torch.from_numpy(real)
    fake_t = torch.from_numpy(fake)
    k = int(min(k, len(real) - 1, len(fake) - 1))
    radii_r = torch.cdist(real_t, real_t).kthvalue(k + 1, dim=1).values
    radii_f = torch.cdist(fake_t, fake_t).kthvalue(k + 1, dim=1).values
    d_rf = torch.cdist(real_t, fake_t)
    inside_real = d_rf <= radii_r.unsqueeze(1)
    inside_fake = d_rf <= radii_f.unsqueeze(0)
    return {
        "precision": float(inside_real.any(dim=0).float().mean()),
        "recall": float(inside_fake.any(dim=1).float().mean()),
        "density": float(inside_real.sum(dim=0).float().mean() / k),
        "coverage": float((d_rf.min(dim=1).values <= radii_r).float().mean()),
    }


def cosine_rows(a, b):
    a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-8)
    b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-8)
    return (a * b).sum(axis=1)


def memorization_distance(fake, real):
    fake_n = fake / np.maximum(np.linalg.norm(fake, axis=1, keepdims=True), 1e-8)
    real_n = real / np.maximum(np.linalg.norm(real, axis=1, keepdims=True), 1e-8)
    sims = torch.from_numpy(fake_n) @ torch.from_numpy(real_n).T
    return float((1.0 - sims.max(dim=1).values).mean())


def subsample(items, n, rng):
    if n is None or len(items) <= int(n):
        return list(items)
    idx = np.sort(rng.choice(len(items), int(n), replace=False))
    return [items[i] for i in idx]


@torch.no_grad()
def paired_direction(gen_fwd, gen_back, arr, device, batch_size, lpips_fn, autocast_ctx):
    l1_cycle, l1_idt, lp_trans, lp_rec, fakes = [], [], [], [], []
    for i in range(0, len(arr), batch_size):
        x = to_tensor(arr[i : i + batch_size], device)
        with autocast_ctx():
            fake = gen_fwd(x).float()
            rec = gen_back(fake).float()
            idt = gen_back(x).float()
        l1_cycle.append(((rec - x).abs().mean(dim=(1, 2, 3)) / 2).cpu().numpy())
        l1_idt.append(((idt - x).abs().mean(dim=(1, 2, 3)) / 2).cpu().numpy())
        if lpips_fn is not None:
            lp_trans.append(lpips_fn(x, fake))
            lp_rec.append(lpips_fn(x, rec))
        fakes.append(to_uint8(fake))
    return {
        "cycle_l1": float(np.concatenate(l1_cycle).mean()),
        "identity_l1": float(np.concatenate(l1_idt).mean()),
        "lpips_translation": float(np.concatenate(lp_trans).mean()) if lp_trans else "unavailable",
        "lpips_reconstruction": float(np.concatenate(lp_rec).mean()) if lp_rec else "unavailable",
        "fakes": np.concatenate(fakes),
    }


def compute_metrics(cfg, paths, holdout, gens, out_dir, train_summary, translate_summary, logger):
    mc = cfg["metrics"]
    out_dir = Path(out_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(cfg["seed"])
    use_amp = device.type == "cuda"

    def autocast_ctx():
        if use_amp:
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        import contextlib

        return contextlib.nullcontext()

    g_ab, g_ba = gens
    size = int(cfg["data"]["infer_size"])
    bs = int(mc["batch_size"])
    nmax = mc.get("fid_max_images")
    inception = InceptionExtractor(device)
    logger.info("inception feature extraction on %s", device)

    real_monet_paths = subsample(paths["monet"], nmax, rng)
    real_photo_paths = subsample(paths["photo"], nmax, rng)
    fake_monet_paths = subsample(list_images(out_dir / "pred_A2B"), nmax, rng)
    fake_photo_paths = subsample(list_images(out_dir / "pred_B2A"), nmax, rng)
    feats = {
        "real_monet": inception.from_paths(real_monet_paths, size, bs),
        "real_photo": inception.from_paths(real_photo_paths, size, bs),
        "fake_monet": inception.from_paths(fake_monet_paths, size, bs),
        "fake_photo": inception.from_paths(fake_photo_paths, size, bs),
    }
    logger.info("features %s", json.dumps({k: list(v.shape) for k, v in feats.items()}))

    m = OrderedDict()
    for label, real, fake in (("photo->monet", "real_monet", "fake_monet"), ("monet->photo", "real_photo", "fake_photo")):
        m[f"FID {label}"] = fid_from_features(feats[real], feats[fake])
        kid_mean, kid_std = kid_from_features(feats[real], feats[fake], mc["kid_subsets"], mc["kid_subset_size"], rng)
        m[f"KID mean {label}"] = kid_mean
        m[f"KID std {label}"] = kid_std
        mm = manifold_metrics(feats[real], feats[fake], mc["knn_k"])
        m[f"Precision {label}"] = mm["precision"]
        m[f"Recall {label}"] = mm["recall"]
        m[f"Density {label}"] = mm["density"]
        m[f"Coverage {label}"] = mm["coverage"]
        m[f"Images compared {label} (real, fake)"] = f"{len(feats[real])}, {len(feats[fake])}"
    d = memorization_distance(feats["fake_monet"], feats["real_monet"])
    eps = float(mc["mifid_epsilon"])
    m["Memorization distance photo->monet"] = d
    m[f"MiFID-like photo->monet (epsilon {eps})"] = m["FID photo->monet"] / d if d < eps else m["FID photo->monet"]

    try:
        lpips_fn = LPIPSMetric(mc["lpips_net"], device)
    except Exception as exc:
        logger.warning("LPIPS unavailable, its rows are left for a re-run of the metrics stage: %s", exc)
        lpips_fn = None
    ab = paired_direction(g_ab, g_ba, holdout["photo"], device, bs, lpips_fn, autocast_ctx)
    ba = paired_direction(g_ba, g_ab, holdout["monet"], device, bs, lpips_fn, autocast_ctx)
    m["Cycle-reconstruction L1 photo->monet->photo"] = ab["cycle_l1"]
    m["Cycle-reconstruction L1 monet->photo->monet"] = ba["cycle_l1"]
    m["Identity L1 G_BA(photo) vs photo"] = ab["identity_l1"]
    m["Identity L1 G_AB(monet) vs monet"] = ba["identity_l1"]
    m["LPIPS input vs translation photo->monet"] = ab["lpips_translation"]
    m["LPIPS input vs translation monet->photo"] = ba["lpips_translation"]
    m["LPIPS input vs reconstruction photo->monet->photo"] = ab["lpips_reconstruction"]
    m["LPIPS input vs reconstruction monet->photo->monet"] = ba["lpips_reconstruction"]
    in_photo = inception.from_arrays(holdout["photo"], bs)
    in_monet = inception.from_arrays(holdout["monet"], bs)
    m["Content cosine similarity photo->monet"] = float(cosine_rows(in_photo, inception.from_arrays(ab["fakes"], bs)).mean())
    m["Content cosine similarity monet->photo"] = float(cosine_rows(in_monet, inception.from_arrays(ba["fakes"], bs)).mean())
    m["Holdout images (photo, monet)"] = f"{len(holdout['photo'])}, {len(holdout['monet'])}"

    if train_summary:
        last = train_summary["final_epoch_losses"]
        for k in ["loss_g", "loss_d_a", "loss_d_b", "loss_gan_ab", "loss_gan_ba", "loss_cycle", "loss_identity"]:
            m[f"Final epoch mean {k}"] = last[k]
        m["Gradient norm G mean"] = train_summary["grad_norm_g_mean"]
        m["Gradient norm G max"] = train_summary["grad_norm_g_max"]
        m["Gradient norm D mean"] = train_summary["grad_norm_d_mean"]
        m["Gradient norm D max"] = train_summary["grad_norm_d_max"]
        m["NaN or Inf losses"] = train_summary["nan_losses"]
        m["Mixed precision disabled at step"] = train_summary["amp_disabled_step"]
        for k, v in train_summary["param_count"].items():
            m[f"Parameter count {k}"] = v
        m["Training time (sec)"] = train_summary["train_time_sec"]
        m["Training images/sec"] = train_summary["train_images_per_sec"]
        m["Training image size"] = train_summary["image_size"]
        m["Epochs"] = train_summary["epochs"]
        m["Optimizer steps"] = train_summary["optimizer_steps"]
        m["Peak memory (MB)"] = train_summary["peak_memory_mb"]
        m["Peak memory source"] = train_summary["peak_memory_source"]
        m["Device"] = train_summary["device"]
    if translate_summary:
        m["Inference images/sec photo->monet"] = translate_summary["a2b"]["images_per_sec"]
        m["Inference images/sec monet->photo"] = translate_summary["b2a"]["images_per_sec"]
        m["Inference image size"] = translate_summary["infer_size"]
        m["Submission images"] = translate_summary["submission_images"]

    audit_path = out_dir / "audit" / "audit_results.json"
    if audit_path.exists():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        for k, v in audit.get("report_rows", {}).items():
            m[k] = v
    else:
        for k in ["Human audit style mean", "Human audit content mean", "Human audit artifacts mean", "Human audit Cohen's kappa (mean over criteria)", "Human audit percent agreement (mean over criteria)"]:
            m[k] = "to fill: run audit.py"
    kaggle_path = out_dir / "kaggle" / "kaggle_results.json"
    if kaggle_path.exists():
        kg = json.loads(kaggle_path.read_text(encoding="utf-8"))
        m["Kaggle public score"] = kg.get("public_score", "to fill")
        m["Kaggle private score"] = kg.get("private_score", "to fill")
        m["Kaggle leaderboard rank"] = kg.get("rank", "to fill")
    else:
        m["Kaggle public score"] = "to fill: outputs/<run>/kaggle/kaggle_results.json"
        m["Kaggle private score"] = "to fill"
        m["Kaggle leaderboard rank"] = "to fill"
    return m


def write_report(metrics, csv_path, json_path):
    import csv

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Metric", "Value"])
        for k, v in metrics.items():
            w.writerow([k, v])
    Path(json_path).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
