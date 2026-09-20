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

from model import CharGPT
from runlog import PeakMemory


def lr_at(step, total_steps, tc):
    lr, min_lr, warmup = tc["lr"], tc["min_lr"], tc["warmup_steps"]
    if step < warmup:
        return lr * (step + 1) / warmup
    if tc["schedule"] == "constant":
        return lr
    progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
    if tc["schedule"] == "linear":
        return min_lr + (lr - min_lr) * (1.0 - progress)
    return min_lr + 0.5 * (lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def build_optimizer(model, tc, device):
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    groups = [
        {"params": decay, "weight_decay": tc["weight_decay"]},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=tc["lr"], betas=tuple(tc["betas"]), fused=device.type == "cuda")


def batch_to_xy(arr, device):
    batch = torch.from_numpy(arr.astype(np.int64)).to(device, non_blocking=True)
    return batch[:, :-1], batch[:, 1:]


@torch.no_grad()
def evaluate(model, arr, batch_size, device, autocast_ctx):
    model.eval()
    total_loss, total_correct, total_tokens = 0.0, 0, 0
    for i in range(0, len(arr), batch_size):
        x, y = batch_to_xy(arr[i : i + batch_size], device)
        with autocast_ctx():
            logits, loss = model(x, y)
        n = y.numel()
        total_loss += loss.item() * n
        total_correct += (logits.argmax(dim=-1) == y).sum().item()
        total_tokens += n
    model.train()
    return total_loss / total_tokens, total_correct / total_tokens


def save_checkpoint(path, model, cfg, tok, epoch, val_loss, optimizer=None):
    payload = {
        "model": model.state_dict(),
        "config": cfg,
        "chars": tok.chars,
        "epoch": epoch,
        "val_loss": val_loss,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, path)


def plot_curves(history, out_dir):
    epochs = [e["epoch"] for e in history["epoch"]]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, [e["train_loss"] for e in history["epoch"]], marker="o", label="train")
    axes[0].plot(epochs, [e["val_loss"] for e in history["epoch"]], marker="o", label="validation")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("cross-entropy loss")
    axes[0].set_title("loss per epoch")
    axes[0].legend()
    step_loss = np.array(history["step_loss"])
    axes[1].plot(step_loss, alpha=0.3, label="step loss")
    if len(step_loss) >= 50:
        kernel = np.ones(50) / 50
        axes[1].plot(np.convolve(step_loss, kernel, mode="valid"), label="moving average 50")
    axes[1].set_xlabel("optimizer step")
    axes[1].set_ylabel("training loss")
    axes[1].set_title("loss per step")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "loss_curves.png", dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["step_lr"])
    axes[0].set_xlabel("optimizer step")
    axes[0].set_ylabel("learning rate")
    axes[0].set_title("learning rate schedule")
    axes[1].plot(history["step_grad_norm"], alpha=0.6)
    axes[1].set_xlabel("optimizer step")
    axes[1].set_ylabel("gradient norm before clipping")
    axes[1].set_title("gradient norm")
    fig.tight_layout()
    fig.savefig(out_dir / "training_dynamics.png", dpi=120)
    plt.close(fig)


def train(cfg, tok, train_arr, val_arr, ckpt_dir, out_dir, logger, bench_steps=None):
    tc, mc, dc = cfg["train"], cfg["model"], cfg["data"]
    ckpt_dir, out_dir = Path(ckpt_dir), Path(out_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(tc["amp"]) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp and amp_dtype == torch.float16)

    def autocast_ctx():
        if use_amp:
            return torch.autocast(device_type="cuda", dtype=amp_dtype)
        return contextlib.nullcontext()

    model = CharGPT(vocab_size=tok.vocab_size, block_size=dc["block_size"], **mc).to(device)
    optimizer = build_optimizer(model, tc, device)
    n_params = model.count_params()
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    logger.info("device %s amp %s dtype %s", device_name, use_amp, amp_dtype if use_amp else "float32")
    logger.info("model %s params %d", json.dumps(mc), n_params)

    B = tc["batch_size"]
    N = len(train_arr)
    steps_per_epoch = math.ceil(N / B)
    total_steps = steps_per_epoch * tc["epochs"]
    logger.info(
        "train sequences %d val sequences %d steps/epoch %d total steps %d",
        N, len(val_arr), steps_per_epoch, total_steps,
    )

    rng = np.random.default_rng(cfg["seed"])
    mem = PeakMemory(device)
    history = {"step_loss": [], "step_lr": [], "step_grad_norm": [], "epoch": []}
    ema = None
    spikes = 0
    nans = 0
    step = 0
    tokens_seen = 0
    train_time = 0.0
    best_val = float("inf")
    best_epoch = 0
    t_start = time.time()
    model.train()

    for epoch in range(1, tc["epochs"] + 1):
        perm = rng.permutation(N)
        epoch_losses = []
        epoch_tokens = 0
        t_epoch = time.time()
        for i in range(0, N, B):
            lr = lr_at(step, total_steps, tc)
            for group in optimizer.param_groups:
                group["lr"] = lr
            x, y = batch_to_xy(train_arr[perm[i : i + B]], device)
            with autocast_ctx():
                _, loss = model(x, y)
            loss_val = loss.item()
            if not math.isfinite(loss_val):
                nans += 1
                optimizer.zero_grad(set_to_none=True)
                logger.warning("non-finite loss at step %d, batch skipped", step)
                step += 1
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"]).item()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if ema is None:
                ema = loss_val
            else:
                if step >= tc["warmup_steps"] and loss_val > 1.5 * ema:
                    spikes += 1
                    logger.warning("loss spike at step %d: %.4f vs ema %.4f", step, loss_val, ema)
                ema = 0.98 * ema + 0.02 * loss_val
            history["step_loss"].append(loss_val)
            history["step_lr"].append(lr)
            history["step_grad_norm"].append(grad_norm)
            epoch_losses.append(loss_val)
            epoch_tokens += y.numel()
            step += 1
            if step == 1 or step % tc["log_interval"] == 0:
                mem.update()
                logger.info(
                    "epoch %d step %d/%d loss %.4f lr %.2e grad_norm %.3f",
                    epoch, step, total_steps, loss_val, lr, grad_norm,
                )
            if bench_steps and step >= bench_steps:
                break
        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_time = time.time() - t_epoch
        train_time += epoch_time
        tokens_seen += epoch_tokens

        if bench_steps:
            est = total_steps * (train_time / step)
            result = {
                "bench_steps": step,
                "device": device_name,
                "param_count": n_params,
                "train_tokens_per_sec": tokens_seen / train_time,
                "sec_per_step": train_time / step,
                "total_steps": total_steps,
                "estimated_full_train_sec": est,
                "estimated_full_train_min": est / 60,
                **mem.result(),
            }
            logger.info("benchmark %s", json.dumps(result))
            return result

        val_loss, val_acc = evaluate(model, val_arr, tc["eval_batch_size"], device, autocast_ctx)
        train_loss = float(np.mean(epoch_losses))
        history["epoch"].append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_top1_accuracy": val_acc,
                "epoch_time_sec": epoch_time,
                "tokens_per_sec": epoch_tokens / epoch_time,
            }
        )
        logger.info(
            "epoch %d done train_loss %.4f val_loss %.4f val_top1 %.4f time %.1fs tokens/sec %.0f",
            epoch, train_loss, val_loss, val_acc, epoch_time, epoch_tokens / epoch_time,
        )
        save_checkpoint(ckpt_dir / f"epoch_{epoch:02d}.pt", model, cfg, tok, epoch, val_loss)
        save_checkpoint(ckpt_dir / "last.pt", model, cfg, tok, epoch, val_loss, optimizer)
        if val_loss < best_val:
            best_val, best_epoch = val_loss, epoch
            save_checkpoint(ckpt_dir / "best.pt", model, cfg, tok, epoch, val_loss)
        (out_dir / "history.json").write_text(json.dumps(history), encoding="utf-8")
        plot_curves(history, out_dir)

    last = history["epoch"][-1]
    save_checkpoint(ckpt_dir / "final.pt", model, cfg, tok, tc["epochs"], last["val_loss"])
    grad_norms = np.array(history["step_grad_norm"])
    summary = {
        "run_name": cfg["run_name"],
        "device": device_name,
        "amp_dtype": str(amp_dtype) if use_amp else "float32",
        "param_count": n_params,
        "vocab_size": tok.vocab_size,
        "block_size": dc["block_size"],
        "batch_size": B,
        "epochs": tc["epochs"],
        "steps_per_epoch": steps_per_epoch,
        "optimizer_steps": step,
        "final_train_loss": last["train_loss"],
        "final_val_loss": last["val_loss"],
        "best_val_loss": best_val,
        "best_epoch": best_epoch,
        "val_top1_accuracy": last["val_top1_accuracy"],
        "train_tokens": tokens_seen,
        "train_time_sec": train_time,
        "train_tokens_per_sec": tokens_seen / train_time,
        "grad_norm_mean": float(grad_norms.mean()),
        "grad_norm_max": float(grad_norms.max()),
        "grad_norm_last": float(grad_norms[-1]),
        "loss_spikes": spikes,
        "nan_losses": nans,
        "wall_time_sec": time.time() - t_start,
        "checkpoints": {
            "final.pt": {"epoch": tc["epochs"], "val_loss": last["val_loss"]},
            "best.pt": {"epoch": best_epoch, "val_loss": best_val},
        },
        **mem.result(),
    }
    return summary
