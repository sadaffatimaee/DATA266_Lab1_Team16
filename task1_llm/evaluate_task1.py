import argparse
import csv
import json
import math
from collections import OrderedDict
from pathlib import Path


def ngrams(tokens, n):
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def distinct_n(texts, n):
    total = 0
    unique = set()
    for text in texts:
        grams = ngrams(text.split(), n)
        total += len(grams)
        unique.update(grams)
    return len(unique) / total if total else 0.0


def repeated_ngram_rate(texts, n=4):
    rates = []
    for text in texts:
        grams = ngrams(text.split(), n)
        if grams:
            rates.append(1.0 - len(set(grams)) / len(grams))
    return sum(rates) / len(rates) if rates else 0.0


def compute_metrics(summary, texts):
    train_loss = summary["final_train_loss"]
    val_loss = summary["final_val_loss"]
    m = OrderedDict()
    m["Training Cross-Entropy Loss"] = train_loss
    m["Validation Cross-Entropy Loss"] = val_loss
    m["Best Validation Cross-Entropy Loss"] = summary["best_val_loss"]
    m["Perplexity"] = math.exp(val_loss)
    m["Bits-Per-Character"] = val_loss / math.log(2)
    m["Generalization Gap"] = val_loss - train_loss
    m["Top-1 Next-Character Accuracy"] = summary["val_top1_accuracy"]
    m["Distinct-1"] = distinct_n(texts, 1)
    m["Distinct-2"] = distinct_n(texts, 2)
    m["Distinct-3"] = distinct_n(texts, 3)
    m["Repeated 4-gram Rate"] = repeated_ngram_rate(texts, 4)
    m["Gradient Norm Mean"] = summary["grad_norm_mean"]
    m["Gradient Norm Max"] = summary["grad_norm_max"]
    m["Gradient Norm Last"] = summary["grad_norm_last"]
    m["Loss Spikes"] = summary["loss_spikes"]
    m["NaN or Inf Losses"] = summary["nan_losses"]
    m["Parameter Count"] = summary["param_count"]
    m["Training Tokens/sec"] = summary["train_tokens_per_sec"]
    m["Generation Tokens/sec"] = summary.get("generation_tokens_per_sec", 0.0)
    m["Peak Memory (MB)"] = summary["peak_memory_mb"]
    m["Peak Memory Source"] = summary["peak_memory_source"]
    m["Total Training Time (sec)"] = summary["train_time_sec"]
    m["Epochs"] = summary["epochs"]
    m["Optimizer Steps"] = summary["optimizer_steps"]
    m["Device"] = summary["device"]
    return m


def write_csv(metrics, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Metric", "Value"])
        for k, v in metrics.items():
            w.writerow([k, v])


def load_texts(path):
    path = Path(path)
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return [s.get("continuation", s["text"]) if isinstance(s, dict) else str(s) for s in data]
    return [t for t in path.read_text(encoding="utf-8").split("\n\n") if t.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    ap.add_argument("--samples", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    metrics = compute_metrics(summary, load_texts(args.samples))
    write_csv(metrics, args.out)
    for k, v in metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
