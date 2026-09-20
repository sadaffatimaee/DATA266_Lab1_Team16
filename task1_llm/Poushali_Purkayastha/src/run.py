import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import torch
import yaml

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(SRC_DIR.parents[1]))

from data import load_arrays, prepare
from evaluate_task1 import compute_metrics, write_csv
from generate import generate_samples, load_model, save_samples
from runlog import (
    MEMBER,
    MEMBER_DIR,
    REPO_DIR,
    TASK,
    git_state,
    hardware_info,
    now_id,
    package_versions,
    setup_logger,
    write_manifest,
)
from train import train


def resolve_config(arg):
    p = Path(arg)
    if p.exists():
        return p.resolve()
    p = SRC_DIR / arg
    if p.exists():
        return p
    raise FileNotFoundError(arg)


def rel(path):
    return str(Path(path).resolve().relative_to(REPO_DIR)).replace("\\", "/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/full.yaml")
    ap.add_argument("--force-prepare", action="store_true")
    ap.add_argument("--bench-steps", type=int, default=None)
    args = ap.parse_args()

    config_path = resolve_config(args.config)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_name = cfg["run_name"]
    prefix = f"bench_{run_name}" if args.bench_steps else run_name
    run_id = f"{prefix}_{now_id()}"
    logger, log_path = setup_logger(run_id)
    t0 = time.time()
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    hardware = hardware_info()
    git = git_state()
    logger.info("run_id %s member %s task %s", run_id, MEMBER, TASK)
    logger.info("config %s %s", rel(config_path), json.dumps(cfg))
    logger.info("hardware %s", json.dumps(hardware))
    logger.info("git %s", json.dumps(git))

    data_dir = MEMBER_DIR / "data_processed" / run_name
    ckpt_dir = MEMBER_DIR / "checkpoints" / run_name
    out_dir = MEMBER_DIR / "outputs" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    source = cfg["data"]["source"]
    if source != "hf" and not Path(source).is_absolute():
        cfg["data"]["source"] = str(REPO_DIR / source)
    if args.force_prepare or not (data_dir / "train.npy").exists():
        prepare(cfg, data_dir, logger)
    else:
        logger.info("using cached data in %s", rel(data_dir))
    tok, train_arr, val_arr = load_arrays(data_dir)
    meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))

    summary = train(cfg, tok, train_arr, val_arr, ckpt_dir, out_dir, logger, bench_steps=args.bench_steps)
    if args.bench_steps:
        (out_dir / f"{run_id}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        logger.info("benchmark finished in %.1f sec", time.time() - t0)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tok, _ = load_model(ckpt_dir / "final.pt", device)
    samples, gen_tps = generate_samples(model, tok, cfg, device, logger)
    save_samples(samples, out_dir)
    summary["generation_tokens_per_sec"] = gen_tps
    summary["run_id"] = run_id
    summary["data"] = meta
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    metrics = compute_metrics(summary, [s["continuation"] for s in samples])
    write_csv(metrics, out_dir / "metrics_report.csv")
    if cfg.get("is_final"):
        shutil.copy(out_dir / "metrics_report.csv", MEMBER_DIR / "metrics_report.csv")
    for k, v in metrics.items():
        logger.info("metric %s = %s", k, v)

    manifest = {
        "member": MEMBER,
        "task": TASK,
        "run_id": run_id,
        "run_name": run_name,
        "is_final": bool(cfg.get("is_final")),
        "started": started,
        "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git": git,
        "config_file": rel(config_path),
        "config": cfg,
        "hardware": hardware,
        "environment_file": "requirements.txt",
        "packages": package_versions(),
        "raw_log": rel(log_path),
        "data": meta,
        "checkpoints": {
            rel(ckpt_dir / name): {**info, "reported_in": rel(out_dir / "metrics_report.csv")}
            for name, info in summary["checkpoints"].items()
        },
        "outputs": {
            "metrics_report": rel(out_dir / "metrics_report.csv"),
            "run_summary": rel(out_dir / "run_summary.json"),
            "history": rel(out_dir / "history.json"),
            "loss_curves": rel(out_dir / "loss_curves.png"),
            "training_dynamics": rel(out_dir / "training_dynamics.png"),
            "samples": rel(out_dir / "samples.txt"),
        },
        "metrics": metrics,
    }
    manifest_path = write_manifest(run_id, manifest)
    logger.info("manifest %s", rel(manifest_path))
    logger.info("run finished in %.1f sec", time.time() - t0)


if __name__ == "__main__":
    main()
