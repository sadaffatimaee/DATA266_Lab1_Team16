import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import yaml

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

from evaluate_task2 import evaluate_run
from preprocess import load_processed, load_test_texts, prepare
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
from train import train_model


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
    ap.add_argument("--models", default=None)
    ap.add_argument("--force-prepare", action="store_true")
    ap.add_argument("--bench-steps", type=int, default=None)
    args = ap.parse_args()

    config_path = resolve_config(args.config)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_name = cfg["run_name"]
    models = [m.strip() for m in args.models.split(",")] if args.models else list(cfg["models"])
    unknown = [m for m in models if m not in cfg["models"]]
    if unknown:
        raise ValueError(f"unknown models {unknown}; config has {list(cfg['models'])}")
    prefix = f"bench_{run_name}" if args.bench_steps else run_name
    run_id = f"{prefix}_{now_id()}"
    logger, log_path = setup_logger(run_id)
    t0 = time.time()
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    hardware = hardware_info()
    git = git_state()
    logger.info("run_id %s member %s task %s models %s", run_id, MEMBER, TASK, ",".join(models))
    logger.info("config %s %s", rel(config_path), json.dumps(cfg))
    logger.info("hardware %s", json.dumps(hardware))
    logger.info("git %s", json.dumps(git))

    data_dir = MEMBER_DIR / "data_processed" / run_name
    ckpt_dir = MEMBER_DIR / "checkpoints" / run_name
    out_dir = MEMBER_DIR / "outputs" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.force_prepare or not (data_dir / "meta.json").exists():
        prepare(cfg, data_dir, out_dir / "eda", logger)
    else:
        logger.info("using cached data in %s", rel(data_dir))
    data = load_processed(data_dir)
    logger.info("data meta %s", json.dumps(data["meta"]))

    summaries = {}
    for name in models:
        summaries[name] = train_model(
            name, cfg["models"][name], cfg["train"], cfg, data, ckpt_dir / name, out_dir / name, logger,
            bench_steps=args.bench_steps,
        )
    if args.bench_steps:
        (out_dir / f"{run_id}.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
        logger.info("benchmark finished in %.1f sec", time.time() - t0)
        return

    results = evaluate_run(cfg, data, load_test_texts(data_dir), out_dir, logger)
    if cfg.get("is_final"):
        shutil.copy(out_dir / "metrics_report.csv", MEMBER_DIR / "metrics_report.csv")

    manifest = {
        "member": MEMBER,
        "task": TASK,
        "run_id": run_id,
        "run_name": run_name,
        "is_final": bool(cfg.get("is_final")),
        "models_trained_this_run": models,
        "started": started,
        "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git": git,
        "config_file": rel(config_path),
        "config": cfg,
        "hardware": hardware,
        "environment_file": "requirements.txt",
        "packages": package_versions(),
        "raw_log": rel(log_path),
        "data": data["meta"],
        "checkpoints": {
            rel(ckpt_dir / name / "best.pt"): {
                "epoch": s["best_epoch"],
                "val_macro_f1": s["best_val_macro_f1"],
                "device": s["device"],
                "reported_in": rel(out_dir / "metrics_report.csv"),
            }
            for name, s in summaries.items()
        },
        "outputs": {
            "metrics_report": rel(out_dir / "metrics_report.csv"),
            "metrics_wide": rel(out_dir / "metrics_wide.csv"),
            "eda": rel(out_dir / "eda"),
            "models": {name: rel(out_dir / name) for name in results},
        },
        "metrics": results,
    }
    manifest_path = write_manifest(run_id, manifest)
    logger.info("manifest %s", rel(manifest_path))
    logger.info("run finished in %.1f sec", time.time() - t0)


if __name__ == "__main__":
    main()
