
# plus KID and generative precision / recall / density / coverage on the same Inception features.
# Usage: python evaluate_local.py --data task3_gan/data --pred task3_gan/Sadaf_Fatima_Syeda/outputs
import os, glob, json, argparse
import numpy as np, pandas as pd, torch, torch.nn as nn
import torchvision.transforms as T, torchvision.models as models
import scipy.linalg
from scipy.spatial.distance import cosine
from PIL import Image

INCEPTION_TF = T.Compose([T.Resize(299), T.CenterCrop(299), T.ToTensor(),
                          T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

def list_images(folder):
    exts = (".jpg", ".jpeg", ".png")
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(folder, f"*{ext}")))
        paths.extend(glob.glob(os.path.join(folder, f"*{ext.upper()}")))
    return sorted(list(set(paths)))

def take_n(paths, n):
    return paths if n is None else paths[:min(n, len(paths))]

def get_inception_model(device, random_weights=False):
    weights = None if random_weights else models.Inception_V3_Weights.IMAGENET1K_V1
    m = models.inception_v3(weights=weights, transform_input=False, aux_logits=True, init_weights=random_weights)
    m.fc = nn.Identity()
    return m.to(device).eval()

@torch.no_grad()
def get_activations(model, paths, device, batch_size=32):
    feats = []
    for i in range(0, len(paths), batch_size):
        x = torch.stack([INCEPTION_TF(Image.open(p).convert("RGB")) for p in paths[i:i + batch_size]]).to(device)
        feats.append(model(x).detach().cpu().numpy())
    return np.concatenate(feats, axis=0)

def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    diff = mu1 - mu2
    return float(diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean))

def fid_mifid(real_act, gen_act):
    """ Same as the official calculate_fid_mifid, on precomputed features (inputs already sorted and matched) """
    mu_r, sig_r = real_act.mean(axis=0), np.cov(real_act, rowvar=False)
    mu_g, sig_g = gen_act.mean(axis=0), np.cov(gen_act, rowvar=False)
    fid = frechet_distance(mu_r, sig_r, mu_g, sig_g)
    m = min(len(real_act), len(gen_act))
    mifid = float(np.mean([cosine(real_act[i], gen_act[i]) for i in range(m)]))
    return fid, mifid

def kid(real, gen, n_subsets=50, subset_size=100, seed=0):
    """ Kernel Inception Distance: unbiased MMD^2 with a cubic polynomial kernel """
    rng = np.random.default_rng(seed); d = real.shape[1]; m = min(subset_size, len(real), len(gen)); vals = []
    for _ in range(n_subsets):
        x = real[rng.choice(len(real), m, replace=False)]; y = gen[rng.choice(len(gen), m, replace=False)]
        kxx = (x @ x.T / d + 1) ** 3; kyy = (y @ y.T / d + 1) ** 3; kxy = (x @ y.T / d + 1) ** 3
        vals.append((kxx.sum() - np.trace(kxx)) / (m * (m - 1)) + (kyy.sum() - np.trace(kyy)) / (m * (m - 1)) - 2 * kxy.mean())
    return float(np.mean(vals)), float(np.std(vals))

def prdc(real, gen, k=3):
    """ Precision / recall (Kynkaanniemi et al.) and density / coverage (Naeem et al.) """
    def dists(a, b):
        return np.sqrt(np.maximum((a ** 2).sum(1)[:, None] + (b ** 2).sum(1)[None] - 2 * a @ b.T, 0))
    rr, gg, rg = dists(real, real), dists(gen, gen), dists(real, gen)
    r_rad = np.sort(rr, 1)[:, k]; g_rad = np.sort(gg, 1)[:, k]
    precision = float((rg <= r_rad[:, None]).any(0).mean())
    recall = float((rg <= g_rad[None, :]).any(1).mean())
    density = float((rg <= r_rad[:, None]).sum(0).mean() / k)
    coverage = float((rg.min(1) <= r_rad).mean())
    return precision, recall, density, coverage

def evaluate(data_dir, pred_dir, n_eval=300, out_dir=".", random_weights=False, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    sets = {"real_monet": f"{data_dir}/monet_jpg", "real_photo": f"{data_dir}/photo_jpg",
            "gen_a2b": f"{pred_dir}/pred_A2B", "gen_b2a": f"{pred_dir}/pred_B2A"}
    paths = {k: take_n(list_images(v), n_eval) for k, v in sets.items()}
    model = get_inception_model(device, random_weights)
    res = {}
    for name, real_k, gen_k in [("B2A", "real_monet", "gen_b2a"), ("A2B", "real_photo", "gen_a2b")]:
        n = min(len(paths[real_k]), len(paths[gen_k]))
        real = get_activations(model, sorted(paths[real_k])[:n], device)
        gen = get_activations(model, sorted(paths[gen_k])[:n], device)
        res[f"FID_{name}"], res[f"MiFID_{name}"] = fid_mifid(real, gen)
        res[f"KID_{name}"], res[f"KID_std_{name}"] = kid(real, gen)
        res[f"precision_{name}"], res[f"recall_{name}"], res[f"density_{name}"], res[f"coverage_{name}"] = prdc(real, gen)
        res[f"n_{name}"] = n
    res["FID"] = (res["FID_A2B"] + res["FID_B2A"]) / 2
    res["MiFID"] = (res["MiFID_A2B"] + res["MiFID_B2A"]) / 2
    res["leaderboard_score_estimate"] = (res["FID"] + res["MiFID"]) / 2
    pd.DataFrame([{"ID": 1, "FID": float(res["FID"]), "MiFID": float(res["MiFID"])}]).to_csv(f"{out_dir}/submission.csv", index=False)
    json.dump(res, open(f"{out_dir}/evaluation.json", "w"), indent=2)
    return res

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="task3_gan/data")
    ap.add_argument("--pred", default="task3_gan/Sadaf_Fatima_Syeda/outputs")
    ap.add_argument("--n_eval", type=int, default=300)
    ap.add_argument("--out", default="task3_gan/Sadaf_Fatima_Syeda")
    a, _ = ap.parse_known_args()
    print(json.dumps(evaluate(a.data, a.pred, a.n_eval, a.out), indent=2))
