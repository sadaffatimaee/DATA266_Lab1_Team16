import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import cohen_kappa_score

SRC_DIR = Path(__file__).resolve().parent
MEMBER_DIR = SRC_DIR.parent
REPO_DIR = MEMBER_DIR.parents[1]
CRITERIA = ["style", "content", "artifacts"]


def load_config(arg):
    p = Path(arg) if Path(arg).exists() else SRC_DIR / arg
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def make(cfg, extras):
    run = cfg["run_name"]
    out_dir = MEMBER_DIR / "outputs" / run
    audit_dir = out_dir / "audit"
    holdout = json.loads((MEMBER_DIR / "data_processed" / run / "holdout.json").read_text(encoding="utf-8"))
    rng = np.random.default_rng(cfg["seed"])
    n = min(int(cfg["audit"]["n_samples"]), len(holdout["photo"]))
    chosen = sorted(rng.choice(holdout["photo"], n, replace=False).tolist())
    models = {"Poushali_Purkayastha": out_dir / "pred_A2B"}
    for item in extras:
        name, path = item.split("=", 1)
        models[name] = Path(path) if Path(path).is_absolute() else REPO_DIR / path
    photo_dir = REPO_DIR / cfg["data"]["photo_dir"]
    for sub in ("inputs", "blind"):
        d = audit_dir / sub
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
    key = []
    for name in chosen:
        shutil.copy(photo_dir / name, audit_dir / "inputs" / name)
        for model, pdir in models.items():
            src = pdir / f"{Path(name).stem}.jpg"
            if not src.exists():
                print(f"missing translation for {model}: {src}")
                continue
            blind_id = f"{rng.integers(0, 16**6):06x}"
            shutil.copy(src, audit_dir / "blind" / f"{blind_id}.jpg")
            key.append({"blind_id": blind_id, "model": model, "input_file": name})
    rng.shuffle(key)
    with open(audit_dir / "audit_key.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["blind_id", "model", "input_file"])
        w.writeheader()
        w.writerows(key)
    for rater in cfg["audit"]["raters"]:
        with open(audit_dir / f"sheet_{rater}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["blind_id", "input_file"] + [f"{c}_1to5" for c in CRITERIA])
            for row in key:
                w.writerow([row["blind_id"], row["input_file"], "", "", ""])
    print(f"audit set: {n} inputs, {len(key)} blinded outputs from {list(models)}")
    print(f"raters fill outputs/{run}/audit/sheet_<name>.csv looking only at audit/blind/<blind_id>.jpg next to audit/inputs/<input_file>")
    print("scale: 1 (poor) to 5 (excellent) for style and content; 1 (many artifacts) to 5 (none) for artifacts")
    print("do not open audit_key.csv until every sheet is filled")


def read_sheet(path):
    ratings = {}
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            vals = {}
            for c in CRITERIA:
                v = row.get(f"{c}_1to5", "").strip()
                if v:
                    vals[c] = int(float(v))
            if len(vals) == len(CRITERIA):
                ratings[row["blind_id"]] = vals
    return ratings


def score(cfg, sheets):
    run = cfg["run_name"]
    audit_dir = MEMBER_DIR / "outputs" / run / "audit"
    key = {}
    with open(audit_dir / "audit_key.csv", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            key[row["blind_id"]] = row
    raters = {Path(s).stem.replace("sheet_", ""): read_sheet(s) for s in sheets}
    common = sorted(set.intersection(*(set(r) for r in raters.values())) & set(key))
    if not common:
        raise SystemExit("no blind ids rated by every rater")
    models = sorted({key[b]["model"] for b in common})
    results = {"n_rated": len(common), "raters": list(raters), "per_model": {}, "agreement": {}}
    for model in models:
        ids = [b for b in common if key[b]["model"] == model]
        results["per_model"][model] = {}
        for c in CRITERIA:
            per_rater = {r: float(np.mean([raters[r][b][c] for b in ids])) for r in raters}
            results["per_model"][model][c] = {"mean": float(np.mean(list(per_rater.values()))), "per_rater": per_rater, "n": len(ids)}
    names = list(raters)
    for c in CRITERIA:
        a = [raters[names[0]][b][c] for b in common]
        b_ = [raters[names[1]][b][c] for b in common]
        results["agreement"][c] = {
            "cohen_kappa": float(cohen_kappa_score(a, b_)),
            "cohen_kappa_linear_weighted": float(cohen_kappa_score(a, b_, weights="linear")),
            "percent_agreement": float(np.mean(np.array(a) == np.array(b_))),
            "percent_agreement_within_1": float(np.mean(np.abs(np.array(a) - np.array(b_)) <= 1)),
        }
    mine = "Poushali_Purkayastha"
    rows = {}
    if mine in results["per_model"]:
        for c in CRITERIA:
            rows[f"Human audit {c} mean"] = results["per_model"][mine][c]["mean"]
    rows["Human audit Cohen's kappa (mean over criteria)"] = float(np.mean([results["agreement"][c]["cohen_kappa"] for c in CRITERIA]))
    rows["Human audit linear-weighted kappa (mean over criteria)"] = float(np.mean([results["agreement"][c]["cohen_kappa_linear_weighted"] for c in CRITERIA]))
    rows["Human audit percent agreement (mean over criteria)"] = float(np.mean([results["agreement"][c]["percent_agreement"] for c in CRITERIA]))
    rows["Human audit rated outputs"] = len(common)
    results["report_rows"] = rows
    (audit_dir / "audit_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    with open(audit_dir / "audit_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "criterion", "mean", "n"])
        for model, crit in results["per_model"].items():
            for c, v in crit.items():
                w.writerow([model, c, v["mean"], v["n"]])
        w.writerow([])
        w.writerow(["criterion", "cohen_kappa", "linear_weighted_kappa", "percent_agreement", "within_1"])
        for c, v in results["agreement"].items():
            w.writerow([c, v["cohen_kappa"], v["cohen_kappa_linear_weighted"], v["percent_agreement"], v["percent_agreement_within_1"]])
    print(json.dumps(results, indent=2))
    print("re-run the metrics stage or evaluate_local.py to pull these rows into full_metrics_report.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["make", "score"])
    ap.add_argument("--config", default="configs/full.yaml")
    ap.add_argument("--extra", action="append", default=[])
    ap.add_argument("--sheets", nargs="*", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.action == "make":
        make(cfg, args.extra)
    else:
        if len(args.sheets) != 2:
            raise SystemExit("score needs exactly two filled sheets")
        score(cfg, args.sheets)


if __name__ == "__main__":
    main()
