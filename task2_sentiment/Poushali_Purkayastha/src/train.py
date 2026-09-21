import contextlib
import json
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from models import build_model
from runlog import PeakMemory


def macro_f1(y, pred):
    scores = []
    for c in (0, 1):
        tp = np.sum((pred == c) & (y == c))
        fp = np.sum((pred == c) & (y != c))
        fn = np.sum((pred != c) & (y == c))
        denom = 2 * tp + fp + fn
        scores.append(2 * tp / denom if denom else 0.0)
    return float(np.mean(scores))


def to_device(ids, lengths, idx, device):
    x = torch.from_numpy(ids[idx].astype(np.int64)).to(device, non_blocking=True)
    l = torch.from_numpy(lengths[idx].astype(np.int64)).to(device, non_blocking=True)
    return x, l


@torch.no_grad()
def predict_probs(model, ids, lengths, batch_size, device, autocast_ctx):
    model.eval()
    probs = np.zeros(len(ids), dtype=np.float32)
    for i in range(0, len(ids), batch_size):
        idx = np.arange(i, min(i + batch_size, len(ids)))
        x, l = to_device(ids, lengths, idx, device)
        with autocast_ctx():
            logits = model(x, l)
        probs[idx] = torch.sigmoid(logits.float()).cpu().numpy()
    model.train()
    return probs


def bce(y, p, eps=1e-7):
    p = np.clip(p.astype(np.float64), eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def plot_curves(name, history, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    step_loss = np.array(history["step_loss"])
    axes[0].plot(step_loss, alpha=0.3, label="step loss")
    if len(step_loss) >= 20:
        axes[0].plot(np.convolve(step_loss, np.ones(20) / 20, mode="valid"), label="moving average 20")
    axes[0].set_xlabel("optimizer step")
    axes[0].set_ylabel("training BCE loss")
    axes[0].set_title(f"{name}: training loss")
    axes[0].legend()
    epochs = [e["epoch"] for e in history["epoch"]]
    axes[1].plot(epochs, [e["train_loss"] for e in history["epoch"]], marker="o", label="train loss")
    axes[1].plot(epochs, [e["val_loss"] for e in history["epoch"]], marker="o", label="val loss")
    axes[1].plot(epochs, [e["val_macro_f1"] for e in history["epoch"]], marker="s", label="val macro-F1")
    axes[1].set_xlabel("epoch")
    axes[1].set_title(f"{name}: per epoch")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "curves.png", dpi=120)
    plt.close(fig)


def train_model(name, mcfg, tcfg, cfg, data, ckpt_dir, out_dir, logger, bench_steps=None):
    ckpt_dir, out_dir = Path(ckpt_dir), Path(out_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(tcfg["amp"]) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp and amp_dtype == torch.float16)

    def autocast_ctx():
        if use_amp:
            return torch.autocast(device_type="cuda", dtype=amp_dtype)
        return contextlib.nullcontext()

    vocab_size = len(data["itos"])
    model = build_model(mcfg, vocab_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(mcfg["lr"]), weight_decay=float(tcfg["weight_decay"]))
    n_params = sum(p.numel() for p in model.parameters())
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    train_ids, train_len, train_y = data["train_ids"], data["train_len"], data["train_y"]
    val_ids, val_len, val_y = data["val_ids"], data["val_len"], data["val_y"]
    test_ids, test_len = data["test_ids"], data["test_len"]
    B = int(mcfg["batch_size"])
    N = len(train_ids)
    steps_per_epoch = math.ceil(N / B)
    epochs = int(mcfg["epochs"])
    total_steps = steps_per_epoch * epochs
    eval_bs = int(tcfg["eval_batch_size"])
    logger.info("%s device %s amp %s params %d config %s", name, device_name, use_amp, n_params, json.dumps(mcfg))
    logger.info(
        "%s train %d val %d test %d steps/epoch %d total steps %d",
        name, N, len(val_ids), len(test_ids), steps_per_epoch, total_steps,
    )

    mem = PeakMemory(device)
    history = {"step_loss": [], "step_grad_norm": [], "epoch": []}
    best = {"val_macro_f1": -1.0, "epoch": 0}
    step = 0
    nans = 0
    train_time = 0.0
    examples_seen = 0
    model.train()

    for epoch in range(1, epochs + 1):
        perm = rng.permutation(N)
        epoch_losses = []
        epoch_examples = 0
        t_epoch = time.time()
        for i in range(0, N, B):
            idx = perm[i : i + B]
            x, l = to_device(train_ids, train_len, idx, device)
            y = torch.from_numpy(train_y[idx].astype(np.float32)).to(device, non_blocking=True)
            with autocast_ctx():
                logits = model(x, l)
            loss = F.binary_cross_entropy_with_logits(logits.float(), y)
            loss_val = loss.item()
            if not math.isfinite(loss_val):
                nans += 1
                optimizer.zero_grad(set_to_none=True)
                logger.warning("%s non-finite loss at step %d, batch skipped", name, step)
                step += 1
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(tcfg["grad_clip"])).item()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            history["step_loss"].append(loss_val)
            history["step_grad_norm"].append(grad_norm)
            epoch_losses.append(loss_val)
            epoch_examples += len(idx)
            step += 1
            if step == 1 or step % int(tcfg["log_interval"]) == 0:
                mem.update()
                logger.info("%s epoch %d step %d/%d loss %.4f grad_norm %.3f", name, epoch, step, total_steps, loss_val, grad_norm)
            if bench_steps and step >= bench_steps:
                break
        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_time = time.time() - t_epoch
        train_time += epoch_time
        examples_seen += epoch_examples

        if bench_steps:
            est = total_steps * (train_time / step)
            result = {
                "model": name,
                "bench_steps": step,
                "device": device_name,
                "param_count": n_params,
                "train_examples_per_sec": examples_seen / train_time,
                "sec_per_step": train_time / step,
                "total_steps": total_steps,
                "estimated_train_sec": est,
                "estimated_train_min": est / 60,
                **mem.result(),
            }
            logger.info("benchmark %s", json.dumps(result))
            return result

        val_probs = predict_probs(model, val_ids, val_len, eval_bs, device, autocast_ctx)
        val_pred = (val_probs >= 0.5).astype(np.int64)
        rec = {
            "epoch": epoch,
            "train_loss": float(np.mean(epoch_losses)),
            "val_loss": bce(val_y, val_probs),
            "val_accuracy": float((val_pred == val_y).mean()),
            "val_macro_f1": macro_f1(val_y, val_pred),
            "epoch_time_sec": epoch_time,
            "examples_per_sec": epoch_examples / epoch_time,
        }
        history["epoch"].append(rec)
        logger.info(
            "%s epoch %d done train_loss %.4f val_loss %.4f val_acc %.4f val_macro_f1 %.4f time %.1fs examples/sec %.0f",
            name, epoch, rec["train_loss"], rec["val_loss"], rec["val_accuracy"], rec["val_macro_f1"],
            epoch_time, rec["examples_per_sec"],
        )
        if rec["val_macro_f1"] > best["val_macro_f1"]:
            best = {"val_macro_f1": rec["val_macro_f1"], "epoch": epoch}
            torch.save(
                {
                    "model": model.state_dict(),
                    "name": name,
                    "model_config": mcfg,
                    "vocab_size": vocab_size,
                    "epoch": epoch,
                    "val_macro_f1": rec["val_macro_f1"],
                },
                ckpt_dir / "best.pt",
            )
        (out_dir / "history.json").write_text(json.dumps(history), encoding="utf-8")
        plot_curves(name, history, out_dir)

    ckpt = torch.load(ckpt_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    val_probs = predict_probs(model, val_ids, val_len, eval_bs, device, autocast_ctx)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    test_probs = predict_probs(model, test_ids, test_len, eval_bs, device, autocast_ctx)
    if device.type == "cuda":
        torch.cuda.synchronize()
    infer_time = max(time.time() - t0, 1e-9)
    np.save(out_dir / "test_probs.npy", test_probs)
    np.save(out_dir / "val_probs.npy", val_probs)
    grad_norms = np.array(history["step_grad_norm"])
    summary = {
        "model": name,
        "type": mcfg["type"],
        "model_config": mcfg,
        "device": device_name,
        "amp_dtype": str(amp_dtype) if use_amp else "float32",
        "param_count": n_params,
        "epochs": epochs,
        "steps_per_epoch": steps_per_epoch,
        "optimizer_steps": step,
        "best_epoch": best["epoch"],
        "best_val_macro_f1": best["val_macro_f1"],
        "final_train_loss": history["epoch"][-1]["train_loss"],
        "train_examples": examples_seen,
        "train_time_sec": train_time,
        "train_examples_per_sec": examples_seen / train_time,
        "test_inference_time_sec": infer_time,
        "test_inference_examples_per_sec": len(test_ids) / infer_time,
        "grad_norm_mean": float(grad_norms.mean()),
        "grad_norm_max": float(grad_norms.max()),
        "nan_losses": nans,
        "checkpoint": "best.pt",
        **mem.result(),
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
