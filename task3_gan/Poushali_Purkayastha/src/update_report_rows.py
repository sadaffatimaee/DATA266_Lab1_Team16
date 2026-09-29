import argparse
import csv
import json
import sys
from pathlib import Path

import yaml

SRC_DIR = Path(__file__).resolve().parent
MEMBER_DIR = SRC_DIR.parent


def read_rows(path):
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    return rows[0], rows[1:]


def write_rows(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/full.yaml")
    args = ap.parse_args()
    cfg_path = Path(args.config) if Path(args.config).exists() else SRC_DIR / args.config
    run = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))["run_name"]
    out_dir = MEMBER_DIR / "outputs" / run
    updates = {}
    audit = out_dir / "audit" / "audit_results.json"
    if audit.exists():
        updates.update(json.loads(audit.read_text(encoding="utf-8")).get("report_rows", {}))
    kaggle = out_dir / "kaggle" / "kaggle_results.json"
    if kaggle.exists():
        kg = json.loads(kaggle.read_text(encoding="utf-8"))
        updates["Kaggle public score"] = kg.get("public_score", "to fill")
        updates["Kaggle private score"] = kg.get("private_score") if kg.get("private_score") is not None else "to fill"
        updates["Kaggle leaderboard rank"] = kg.get("rank", "to fill")
    official = out_dir / "kaggle" / "official_scores.json"
    if official.exists():
        sc = json.loads(official.read_text(encoding="utf-8"))
        updates["Submission FID"] = sc["FID"]
        updates["Submission MiFID"] = sc["MiFID"]
        updates["Submission score source"] = "official_scores.json from kaggle_eval.py, course definitions with the course real_stats.npz"
    if not updates:
        print("nothing to update: no audit_results.json, kaggle_results.json, or official_scores.json found")
        return
    targets = [out_dir / "full_metrics_report.csv", MEMBER_DIR / "full_metrics_report.csv", MEMBER_DIR / "metrics_report.csv"]
    for path in targets:
        if not path.exists():
            continue
        header, rows = read_rows(path)
        present = {r[0] for r in rows}
        for r in rows:
            if r[0] in updates:
                r[1] = updates[r[0]]
        for k, v in updates.items():
            if k not in present:
                rows.append([k, v])
        write_rows(path, header, rows)
        print("updated", path)
    for k, v in updates.items():
        print(f"  {k} = {v}")


if __name__ == "__main__":
    main()
