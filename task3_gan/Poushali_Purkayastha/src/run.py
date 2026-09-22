import argparse
import contextlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

from data import cache_folder, list_images, split_holdout
from metrics import compute_metrics, write_report
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
from translate import load_generators, translate_paths, write_preview, write_submission


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


def read_json(path):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/full.yaml")
    ap.add_argument("--stage", default="all", choices=["all", "train", "translate", "metrics"])
    ap.add_argument("--bench-steps", type=int, default=None)
    args = ap.parse_args()

    config_path = resolve_config(args.config)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_name = cfg["run_name"]
    prefix = f"bench_{run_name}" if args.bench_steps else (run_name if args.stage == "all" else f"{run_name}_{args.stage}")
    run_id = f"{prefix}_{now_id()}"
    logger, log_path = setup_logger(run_id)
    t0 = time.time()
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    hardware = hardware_info()
    git = git_state()
    logger.info("run_id %s member %s task %s stage %s", run_id, MEMBER, TASK, args.stage)
    logger.info("config %s %s", rel(config_path), json.dumps(cfg))
    logger.info("hardware %s", json.dumps(hardware))
    logger.info("git %s", json.dumps(git))

    d = cfg["data"]
    photo_dir = REPO_DIR / d["photo_dir"]
    monet_dir = REPO_DIR / d["monet_dir"]
    data_dir = MEMBER_DIR / "data_processed" / run_name
    ckpt_dir = MEMBER_DIR / "checkpoints" / run_name
    out_dir = MEMBER_DIR / "outputs" / run_name
    data_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    size = int(d["image_size"])

    photos, photo_paths = cache_folder(photo_dir, size, data_dir / f"photo_{size}.npy", d.get("max_photos"))
    monets, monet_paths = cache_folder(monet_dir, size, data_dir / f"monet_{size}.npy", d.get("max_monet"))
    rng = np.random.default_rng(cfg["seed"])
    photo_train_idx, photo_hold_idx = split_holdout(len(photos), d["holdout_photo"], rng)
    monet_train_idx, monet_hold_idx = split_holdout(len(monets), d["holdout_monet"], rng)
    holdout_info = {
        "photo": [photo_paths[i].name for i in photo_hold_idx],
        "monet": [monet_paths[i].name for i in monet_hold_idx],
        "photo_train": int(len(photo_train_idx)),
        "monet_train": int(len(monet_train_idx)),
        "photo_total": int(len(photos)),
        "monet_total": int(len(monets)),
        "image_size": size,
        "seed": cfg["seed"],
    }
    (data_dir / "holdout.json").write_text(json.dumps(holdout_info, indent=1), encoding="utf-8")
    logger.info("data photos %d (train %d holdout %d) monets %d (train %d holdout %d) size %d",
                len(photos), len(photo_train_idx), len(photo_hold_idx), len(monets), len(monet_train_idx), len(monet_hold_idx), size)
    holdout = {"photo": photos[photo_hold_idx], "monet": monets[monet_hold_idx]}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def autocast_ctx():
        if device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return contextlib.nullcontext()

    train_summary = read_json(out_dir / "train_summary.json")
    translate_summary = read_json(out_dir / "translate_summary.json")
    metrics = None

    if args.stage in ("all", "train"):
        train_summary = train(cfg, photos[photo_train_idx], monets[monet_train_idx], ckpt_dir, out_dir, logger, bench_steps=args.bench_steps)
        if args.bench_steps:
            (out_dir / f"{run_id}.json").write_text(json.dumps(train_summary, indent=2), encoding="utf-8")
            logger.info("benchmark finished in %.1f sec", time.time() - t0)
            return
        train_summary["run_id"] = run_id
        (out_dir / "train_summary.json").write_text(json.dumps(train_summary, indent=2), encoding="utf-8")

    gens = None
    if args.stage in ("all", "translate", "metrics"):
        g_ab, g_ba, _ = load_generators(ckpt_dir / "generators.pt", cfg["model"], device)
        gens = (g_ab, g_ba)

    if args.stage in ("all", "translate"):
        tcfg = cfg["translate"]
        infer_size = int(d["infer_size"])
        n_a2b, ips_a2b, t_a2b = translate_paths(g_ab, photo_paths, infer_size, out_dir / "pred_A2B", device, int(tcfg["batch_size"]), autocast_ctx)
        logger.info("translated %d photos to monet style at %dpx, %.1f images/sec", n_a2b, infer_size, ips_a2b)
        n_b2a, ips_b2a, t_b2a = translate_paths(g_ba, monet_paths, infer_size, out_dir / "pred_B2A", device, int(tcfg["batch_size"]), autocast_ctx)
        logger.info("translated %d monets to photo style at %dpx, %.1f images/sec", n_b2a, infer_size, ips_b2a)
        n_prev = write_preview(out_dir / "pred_A2B", out_dir / "pred_A2B_preview", tcfg["preview_images"])
        write_preview(out_dir / "pred_B2A", out_dir / "pred_B2A_preview", tcfg["preview_images"])
        kaggle_dir = out_dir / "kaggle"
        n_sub = write_submission(out_dir / "pred_A2B", kaggle_dir / cfg["kaggle"]["zip_name"], kaggle_dir / "submission.csv")
        if cfg.get("is_final"):
            shutil.copy(kaggle_dir / "submission.csv", MEMBER_DIR / "submission.csv")
        translate_summary = {
            "run_id": run_id,
            "infer_size": infer_size,
            "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            "a2b": {"images": n_a2b, "images_per_sec": ips_a2b, "seconds": t_a2b},
            "b2a": {"images": n_b2a, "images_per_sec": ips_b2a, "seconds": t_b2a},
            "preview_images": n_prev,
            "submission_images": n_sub,
            "submission_zip": rel(kaggle_dir / cfg["kaggle"]["zip_name"]),
        }
        (out_dir / "translate_summary.json").write_text(json.dumps(translate_summary, indent=2), encoding="utf-8")
        logger.info("submission zip with %d images at %s", n_sub, translate_summary["submission_zip"])

    if args.stage in ("all", "metrics"):
        paths = {"photo": photo_paths, "monet": monet_paths}
        metrics = compute_metrics(cfg, paths, holdout, gens, out_dir, train_summary, translate_summary, logger)
        write_report(metrics, out_dir / "full_metrics_report.csv", out_dir / "metrics.json")
        if cfg.get("is_final"):
            shutil.copy(out_dir / "full_metrics_report.csv", MEMBER_DIR / "full_metrics_report.csv")
        for k, v in metrics.items():
            logger.info("metric %s = %s", k, v)

    manifest = {
        "member": MEMBER,
        "task": TASK,
        "run_id": run_id,
        "run_name": run_name,
        "stage": args.stage,
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
        "data": holdout_info,
        "checkpoints": {
            rel(ckpt_dir / "generators.pt"): {
                "epoch": train_summary["epochs"] if train_summary else None,
                "device": train_summary["device"] if train_summary else None,
                "reported_in": rel(out_dir / "full_metrics_report.csv"),
            }
        },
        "outputs": {
            "full_metrics_report": rel(out_dir / "full_metrics_report.csv"),
            "loss_curves": rel(out_dir / "loss_curves.png"),
            "samples": rel(out_dir / "samples"),
            "pred_A2B": rel(out_dir / "pred_A2B"),
            "pred_B2A": rel(out_dir / "pred_B2A"),
            "kaggle": rel(out_dir / "kaggle"),
        },
        "train_summary": train_summary,
        "translate_summary": translate_summary,
        "metrics": metrics,
    }
    manifest_path = write_manifest(run_id, manifest)
    logger.info("manifest %s", rel(manifest_path))
    logger.info("run finished in %.1f sec", time.time() - t0)


if __name__ == "__main__":
    main()
