import argparse
import csv
import json
import logging
import math
import sys
from collections import OrderedDict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

SLICE_COLUMNS = ["length_bucket", "has_negation", "has_but", "has_exclamation"]
HARDWARE_KEYS = [
    ("Parameter Count", "param_count"),
    ("Training Time (sec)", "train_time_sec"),
    ("Training Examples/sec", "train_examples_per_sec"),
    ("Inference Examples/sec", "test_inference_examples_per_sec"),
    ("Peak Memory (MB)", "peak_memory_mb"),
    ("Peak Memory Source", "peak_memory_source"),
    ("Device", "device"),
    ("Epochs", "epochs"),
    ("Best Epoch", "best_epoch"),
    ("Optimizer Steps", "optimizer_steps"),
]


def counts_metrics(y, pred):
    tp = float(np.sum((pred == 1) & (y == 1)))
    tn = float(np.sum((pred == 0) & (y == 0)))
    fp = float(np.sum((pred == 1) & (y == 0)))
    fn = float(np.sum((pred == 0) & (y == 1)))
    n = tp + tn + fp + fn
    acc = (tp + tn) / n if n else 0.0
    f1_pos = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    f1_neg = 2 * tn / (2 * tn + fn + fp) if (2 * tn + fn + fp) else 0.0
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denom if denom else 0.0
    return acc, (f1_pos + f1_neg) / 2, mcc


def expected_calibration_error(y, p, n_bins):
    pred = (p >= 0.5).astype(np.int64)
    conf = np.where(pred == 1, p, 1 - p)
    correct = (pred == y).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    rows = []
    for j, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        m = (conf > lo) & (conf <= hi) if j > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            gap = abs(correct[m].mean() - conf[m].mean())
            ece += m.mean() * gap
            rows.append({
                "bin_low": float(lo), "bin_high": float(hi), "count": int(m.sum()),
                "confidence": float(conf[m].mean()), "accuracy": float(correct[m].mean()),
            })
    return float(ece), rows


def bootstrap_ci(y, pred, n_samples, seed):
    rng = np.random.default_rng(seed)
    n = len(y)
    stats = np.zeros((n_samples, 3))
    for b in range(n_samples):
        idx = rng.integers(0, n, n)
        stats[b] = counts_metrics(y[idx], pred[idx])
    lo = np.percentile(stats, 2.5, axis=0)
    hi = np.percentile(stats, 97.5, axis=0)
    return {"accuracy": (lo[0], hi[0]), "macro_f1": (lo[1], hi[1]), "mcc": (lo[2], hi[2])}


def mcnemar_test(y, pred_baseline, pred_model):
    base_ok = pred_baseline == y
    model_ok = pred_model == y
    b = int(np.sum(base_ok & ~model_ok))
    c = int(np.sum(~base_ok & model_ok))
    if b + c == 0:
        return {"b": b, "c": c, "statistic": 0.0, "p_value": 1.0}
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    return {"b": b, "c": c, "statistic": float(stat), "p_value": float(math.erfc(math.sqrt(stat / 2)))}


def slice_rows(y, pred, slices):
    rows = []
    for col in SLICE_COLUMNS:
        values = slices[col]
        for v in sorted(set(values.tolist())):
            m = values == v
            acc, f1, _ = counts_metrics(y[m], pred[m])
            rows.append({"slice": f"{col}={v}", "support": int(m.sum()), "macro_f1": f1, "error_rate": 1.0 - acc})
    return rows


def model_metrics(y, p, slices, ev, seed):
    pred = (p >= 0.5).astype(np.int64)
    m = OrderedDict()
    m["Accuracy"] = float((pred == y).mean())
    for avg in ("macro", "micro", "weighted"):
        pr, rc, f1, _ = precision_recall_fscore_support(y, pred, average=avg, zero_division=0)
        m[f"Precision ({avg})"] = float(pr)
        m[f"Recall ({avg})"] = float(rc)
        m[f"F1 ({avg})"] = float(f1)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    m["Confusion TN"], m["Confusion FP"], m["Confusion FN"], m["Confusion TP"] = int(tn), int(fp), int(fn), int(tp)
    m["ROC-AUC"] = float(roc_auc_score(y, p))
    m["PR-AUC"] = float(average_precision_score(y, p))
    m["MCC"] = float(matthews_corrcoef(y, pred))
    m["Brier Score"] = float(np.mean((p - y) ** 2))
    ece, bins = expected_calibration_error(y, p, int(ev["ece_bins"]))
    m["ECE"] = ece
    ci = bootstrap_ci(y, pred, int(ev["bootstrap_samples"]), seed)
    m["Accuracy 95% CI low"], m["Accuracy 95% CI high"] = map(float, ci["accuracy"])
    m["Macro-F1 95% CI low"], m["Macro-F1 95% CI high"] = map(float, ci["macro_f1"])
    m["MCC 95% CI low"], m["MCC 95% CI high"] = map(float, ci["mcc"])
    for row in slice_rows(y, pred, slices):
        m[f"Macro-F1 [{row['slice']}]"] = row["macro_f1"]
        m[f"Error Rate [{row['slice']}]"] = row["error_rate"]
        m[f"Support [{row['slice']}]"] = row["support"]
    return m, pred, bins


def plot_model(name, y, p, pred, bins, out_dir):
    cm = confusion_matrix(y, pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["negative", "positive"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["negative", "positive"])
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"{name}: confusion matrix")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=120)
    plt.close(fig)

    fpr, tpr, _ = roc_curve(y, p)
    prec, rec, _ = precision_recall_curve(y, p)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(fpr, tpr)
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray")
    axes[0].set_xlabel("false positive rate")
    axes[0].set_ylabel("true positive rate")
    axes[0].set_title(f"{name}: ROC curve")
    axes[1].plot(rec, prec)
    axes[1].set_xlabel("recall")
    axes[1].set_ylabel("precision")
    axes[1].set_title(f"{name}: precision-recall curve")
    fig.tight_layout()
    fig.savefig(out_dir / "roc_pr_curves.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot([0.5, 1], [0.5, 1], linestyle="--", color="gray", label="perfect calibration")
    ax.plot([b["confidence"] for b in bins], [b["accuracy"] for b in bins], marker="o", label="model")
    ax.set_xlabel("mean confidence in bin")
    ax.set_ylabel("accuracy in bin")
    ax.set_title(f"{name}: reliability diagram")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "calibration.png", dpi=120)
    plt.close(fig)


def error_review_candidates(y, p, texts, slices, k, min_support):
    pred = (p >= 0.5).astype(np.int64)
    err = pred != y
    chosen = set()

    def take(indices, group):
        out = []
        for i in indices:
            i = int(i)
            if i in chosen:
                continue
            chosen.add(i)
            out.append((group, i))
            if len(out) == k:
                break
        return out

    picks = []
    fp_idx = np.where((y == 0) & (pred == 1))[0]
    picks += take(fp_idx[np.argsort(-p[fp_idx])], "confident_false_positive")
    fn_idx = np.where((y == 1) & (pred == 0))[0]
    picks += take(fn_idx[np.argsort(p[fn_idx])], "confident_false_negative")
    err_idx = np.where(err)[0]
    picks += take(err_idx[np.argsort(np.abs(p[err_idx] - 0.5))], "near_threshold")
    eligible = [r for r in slice_rows(y, pred, slices) if r["support"] >= min_support]
    worst = max(eligible, key=lambda r: r["error_rate"]) if eligible else None
    if worst is not None:
        col, v = worst["slice"].split("=", 1)
        s_idx = np.where((slices[col] == v) & err)[0]
        picks += take(s_idx[np.argsort(-np.abs(p[s_idx] - 0.5))], f"slice_specific[{worst['slice']}]")

    rows = []
    for group, i in picks:
        rows.append({
            "group": group,
            "test_index": i,
            "label": int(y[i]),
            "predicted": int(pred[i]),
            "prob_positive": float(p[i]),
            "n_words": int(slices["n_words"][i]),
            "length_bucket": str(slices["length_bucket"][i]),
            "has_negation": str(slices["has_negation"][i]),
            "has_but": str(slices["has_but"][i]),
            "has_exclamation": str(slices["has_exclamation"][i]),
            "text": texts[i],
        })
    return rows, worst


def write_candidates(rows, out_dir):
    fields = list(rows[0].keys()) if rows else ["group"]
    with open(out_dir / "error_review_candidates.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    lines = ["# Error review candidates", ""]
    for r in rows:
        lines.append(
            f"## {r['group']} | test index {r['test_index']} | label {r['label']} | predicted {r['predicted']} "
            f"| p(positive) = {r['prob_positive']:.3f} | {r['n_words']} words | negation {r['has_negation']} "
            f"| but {r['has_but']} | exclamation {r['has_exclamation']}"
        )
        lines.append("")
        lines.append(r["text"].replace("\\n", " "))
        lines.append("")
    (out_dir / "error_review_candidates.md").write_text("\n".join(lines), encoding="utf-8")


def write_reports(results, out_dir):
    with open(out_dir / "metrics_report.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Model", "Metric", "Value"])
        for name, metrics in results.items():
            for k, v in metrics.items():
                w.writerow([name, k, v])
    keys = []
    for metrics in results.values():
        for k in metrics:
            if k not in keys:
                keys.append(k)
    with open(out_dir / "metrics_wide.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Metric"] + list(results.keys()))
        for k in keys:
            w.writerow([k] + [results[n].get(k, "") for n in results])


def evaluate_run(cfg, data, texts, out_dir, logger):
    ev = cfg["evaluation"]
    out_dir = Path(out_dir)
    y = data["test_y"].astype(np.int64)
    slices = data["slices"]
    results = OrderedDict()
    preds = {}
    probs = {}
    for name in cfg["models"]:
        mdir = out_dir / name
        if not (mdir / "test_probs.npy").exists():
            logger.info("no predictions for %s, skipped", name)
            continue
        p = np.load(mdir / "test_probs.npy").astype(np.float64)
        summary = json.loads((mdir / "train_summary.json").read_text(encoding="utf-8"))
        metrics, pred, bins = model_metrics(y, p, slices, ev, cfg["seed"])
        for label, key in HARDWARE_KEYS:
            metrics[label] = summary[key]
        plot_model(name, y, p, pred, bins, mdir)
        results[name] = metrics
        preds[name] = pred
        probs[name] = p
        logger.info(
            "%s accuracy %.4f macro-F1 %.4f MCC %.4f ROC-AUC %.4f ECE %.4f",
            name, metrics["Accuracy"], metrics["F1 (macro)"], metrics["MCC"], metrics["ROC-AUC"], metrics["ECE"],
        )
    base = ev["baseline"]
    if base in preds:
        for name in results:
            if name == base:
                continue
            mc = mcnemar_test(y, preds[base], preds[name])
            results[name]["McNemar b (baseline right, model wrong)"] = mc["b"]
            results[name]["McNemar c (baseline wrong, model right)"] = mc["c"]
            results[name]["McNemar Statistic"] = mc["statistic"]
            results[name]["McNemar p-value"] = mc["p_value"]
            logger.info("McNemar %s vs %s: b %d c %d statistic %.3f p %.4g", name, base, mc["b"], mc["c"], mc["statistic"], mc["p_value"])
    if results:
        write_reports(results, out_dir)
        (out_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    review = ev["error_review_model"] if ev["error_review_model"] in results else (next(reversed(results)) if results else None)
    if review is not None:
        rows, worst = error_review_candidates(y, probs[review], texts, slices, int(ev["n_review_per_group"]), int(ev["slice_min_support"]))
        write_candidates(rows, out_dir / review)
        logger.info("error review candidates for %s: %d rows, worst slice %s", review, len(rows), json.dumps(worst))
    return results


def main():
    src_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(src_dir))
    from preprocess import load_processed, load_test_texts

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/full.yaml")
    args = ap.parse_args()
    config_path = Path(args.config) if Path(args.config).exists() else src_dir / args.config
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    member_dir = src_dir.parent
    data_dir = member_dir / "data_processed" / cfg["run_name"]
    out_dir = member_dir / "outputs" / cfg["run_name"]
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    results = evaluate_run(cfg, load_processed(data_dir), load_test_texts(data_dir), out_dir, logging.getLogger("evaluate"))
    for name, metrics in results.items():
        print(name, "accuracy", metrics["Accuracy"], "macro-F1", metrics["F1 (macro)"])


if __name__ == "__main__":
    main()
