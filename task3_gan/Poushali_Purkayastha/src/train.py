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
from PIL import Image

from data import to_tensor, to_uint8
from models import ImagePool, build_models, count_params
from runlog import PeakMemory

LOSS_KEYS = ["loss_g", "loss_d_a", "loss_d_b", "loss_gan_ab", "loss_gan_ba", "loss_cycle", "loss_identity"]


def lr_at(epoch, tc):
    start = int(tc["decay_start_epoch"])
    total = int(tc["epochs"])
    if epoch <= start:
        return float(tc["lr"])
    return float(tc["lr"]) * max(0.0, 1.0 - (epoch - start) / (total - start + 1))


def gan_loss(pred, real, mode):
    target = torch.ones_like(pred) if real else torch.zeros_like(pred)
    if mode == "lsgan":
        return F.mse_loss(pred, target)
    return F.binary_cross_entropy_with_logits(pred, target)


def set_requires_grad(nets, flag):
    for net in nets:
        for p in net.parameters():
            p.requires_grad_(flag)


def random_flip(x, rng):
    mask = torch.from_numpy(rng.random(x.size(0)) < 0.5).to(x.device)
    if mask.any():
        x = x.clone()
        x[mask] = x[mask].flip(3)
    return x


def save_grid(rows, path):
    grid = np.concatenate([np.concatenate(list(r), axis=1) for r in rows], axis=0)
    Image.fromarray(grid).save(path, quality=92)


@torch.no_grad()
def sample_grid(g_ab, g_ba, real_a, real_b, path):
    fake_b = g_ab(real_a)
    rec_a = g_ba(fake_b)
    fake_a = g_ba(real_b)
    rec_b = g_ab(fake_a)
    rows = [to_uint8(real_a), to_uint8(fake_b), to_uint8(rec_a), to_uint8(real_b), to_uint8(fake_a), to_uint8(rec_b)]
    save_grid(rows, path)


def smooth(values, k=20):
    values = np.asarray(values, dtype=np.float64)
    if len(values) < k:
        return values
    return np.convolve(values, np.ones(k) / k, mode="valid")


def plot_curves(history, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(17, 4))
    for k in ["loss_g", "loss_d_a", "loss_d_b"]:
        axes[0].plot(smooth(history[k]), label=k)
    axes[0].set_title("generator and discriminator losses")
    axes[0].set_xlabel("step")
    axes[0].legend()
    for k in ["loss_gan_ab", "loss_gan_ba", "loss_cycle", "loss_identity"]:
        axes[1].plot(smooth(history[k]), label=k)
    axes[1].set_title("generator loss components")
    axes[1].set_xlabel("step")
    axes[1].legend()
    axes[2].plot(history["grad_norm_g"], alpha=0.5, label="grad norm G")
    axes[2].plot(history["grad_norm_d"], alpha=0.5, label="grad norm D")
    axes[2].set_title("gradient norms before clipping")
    axes[2].set_xlabel("step")
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "loss_curves.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(history["lr"])
    ax.set_xlabel("step")
    ax.set_ylabel("learning rate")
    ax.set_title("learning rate schedule")
    fig.tight_layout()
    fig.savefig(out_dir / "lr_schedule.png", dpi=120)
    plt.close(fig)


def train(cfg, photos, monets, ckpt_dir, out_dir, logger, bench_steps=None):
    tc, mc = cfg["train"], cfg["model"]
    ckpt_dir, out_dir = Path(ckpt_dir), Path(out_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    sample_dir = out_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = {"amp": bool(tc["amp"]) and device.type == "cuda"}
    amp_dtype = torch.bfloat16 if state["amp"] and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler(device.type, enabled=state["amp"] and amp_dtype == torch.float16)

    def autocast_ctx():
        if state["amp"]:
            return torch.autocast(device_type="cuda", dtype=amp_dtype)
        return contextlib.nullcontext()

    g_ab, g_ba, d_a, d_b = build_models(mc)
    for net in (g_ab, g_ba, d_a, d_b):
        net.to(device)
    params = {"g_ab": count_params(g_ab), "g_ba": count_params(g_ba), "d_a": count_params(d_a), "d_b": count_params(d_b)}
    params["total"] = sum(params.values())
    g_params = list(g_ab.parameters()) + list(g_ba.parameters())
    d_params = list(d_a.parameters()) + list(d_b.parameters())
    betas = (float(tc["beta1"]), 0.999)
    opt_g = torch.optim.Adam(g_params, lr=float(tc["lr"]), betas=betas)
    opt_d = torch.optim.Adam(d_params, lr=float(tc["lr"]), betas=betas)
    pool_a, pool_b = ImagePool(tc["pool_size"], rng), ImagePool(tc["pool_size"], rng)
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    mode = tc["gan_mode"]
    lam_c, lam_i = float(tc["lambda_cycle"]), float(tc["lambda_identity"])
    B = int(tc["batch_size"])
    spe = int(tc["steps_per_epoch"])
    epochs = int(tc["epochs"])
    total_steps = spe * epochs
    logger.info("device %s amp %s dtype %s", device_name, state["amp"], amp_dtype if state["amp"] else "float32")
    logger.info("model %s params %s", json.dumps(mc), json.dumps(params))
    logger.info("train photos %d monets %d batch %d steps/epoch %d epochs %d total steps %d", len(photos), len(monets), B, spe, epochs, total_steps)

    n_fixed = min(4, len(photos), len(monets))
    fixed_a = to_tensor(photos[rng.choice(len(photos), n_fixed, replace=False)], device)
    fixed_b = to_tensor(monets[rng.choice(len(monets), n_fixed, replace=False)], device)

    mem = PeakMemory(device)
    history = {k: [] for k in LOSS_KEYS + ["grad_norm_g", "grad_norm_d", "lr"]}
    epoch_log = []
    step = 0
    nans = 0
    consecutive = 0
    amp_disabled_step = None
    train_time = 0.0
    images_seen = 0

    def handle_nan(where):
        nonlocal nans, consecutive, scaler, amp_disabled_step
        nans += 1
        consecutive += 1
        opt_g.zero_grad(set_to_none=True)
        opt_d.zero_grad(set_to_none=True)
        logger.warning("non-finite %s loss at step %d, batch skipped", where, step)
        if state["amp"] and consecutive >= int(tc["amp_nan_limit"]):
            state["amp"] = False
            scaler = torch.amp.GradScaler(device.type, enabled=False)
            amp_disabled_step = step
            logger.warning("mixed precision disabled at step %d after %d consecutive non-finite losses", step, consecutive)

    for epoch in range(1, epochs + 1):
        lr = lr_at(epoch, tc)
        for opt in (opt_g, opt_d):
            for group in opt.param_groups:
                group["lr"] = lr
        epoch_vals = {k: [] for k in LOSS_KEYS}
        t_epoch = time.time()
        for _ in range(spe):
            step += 1
            real_a = random_flip(to_tensor(photos[rng.integers(0, len(photos), B)], device), rng)
            real_b = random_flip(to_tensor(monets[rng.integers(0, len(monets), B)], device), rng)

            set_requires_grad([d_a, d_b], False)
            with autocast_ctx():
                fake_b = g_ab(real_a)
                rec_a = g_ba(fake_b)
                fake_a = g_ba(real_b)
                rec_b = g_ab(fake_a)
                idt_b = g_ab(real_b)
                idt_a = g_ba(real_a)
                loss_gan_ab = gan_loss(d_b(fake_b).float(), True, mode)
                loss_gan_ba = gan_loss(d_a(fake_a).float(), True, mode)
                loss_cycle = F.l1_loss(rec_a.float(), real_a) + F.l1_loss(rec_b.float(), real_b)
                loss_identity = F.l1_loss(idt_b.float(), real_b) + F.l1_loss(idt_a.float(), real_a)
                loss_g = loss_gan_ab + loss_gan_ba + lam_c * loss_cycle + lam_i * loss_identity
            if not math.isfinite(loss_g.item()):
                handle_nan("generator")
                continue
            opt_g.zero_grad(set_to_none=True)
            scaler.scale(loss_g).backward()
            scaler.unscale_(opt_g)
            grad_norm_g = torch.nn.utils.clip_grad_norm_(g_params, float(tc["grad_clip"])).item()
            scaler.step(opt_g)

            set_requires_grad([d_a, d_b], True)
            fake_a_q = pool_a.query(fake_a.detach().float())
            fake_b_q = pool_b.query(fake_b.detach().float())
            with autocast_ctx():
                loss_d_a = 0.5 * (gan_loss(d_a(real_a).float(), True, mode) + gan_loss(d_a(fake_a_q).float(), False, mode))
                loss_d_b = 0.5 * (gan_loss(d_b(real_b).float(), True, mode) + gan_loss(d_b(fake_b_q).float(), False, mode))
                loss_d = loss_d_a + loss_d_b
            if not math.isfinite(loss_d.item()):
                scaler.update()
                handle_nan("discriminator")
                continue
            opt_d.zero_grad(set_to_none=True)
            scaler.scale(loss_d).backward()
            scaler.unscale_(opt_d)
            grad_norm_d = torch.nn.utils.clip_grad_norm_(d_params, float(tc["grad_clip"])).item()
            scaler.step(opt_d)
            scaler.update()

            consecutive = 0
            images_seen += 2 * B
            values = {
                "loss_g": loss_g.item(), "loss_d_a": loss_d_a.item(), "loss_d_b": loss_d_b.item(),
                "loss_gan_ab": loss_gan_ab.item(), "loss_gan_ba": loss_gan_ba.item(),
                "loss_cycle": loss_cycle.item(), "loss_identity": loss_identity.item(),
            }
            for k, v in values.items():
                history[k].append(v)
                epoch_vals[k].append(v)
            history["grad_norm_g"].append(grad_norm_g)
            history["grad_norm_d"].append(grad_norm_d)
            history["lr"].append(lr)
            if step == 1 or step % int(tc["log_interval"]) == 0:
                mem.update()
                logger.info(
                    "epoch %d step %d/%d loss_g %.3f loss_d_a %.3f loss_d_b %.3f gan_ab %.3f gan_ba %.3f cycle %.3f idt %.3f grad_g %.2f grad_d %.2f lr %.1e",
                    epoch, step, total_steps, values["loss_g"], values["loss_d_a"], values["loss_d_b"],
                    values["loss_gan_ab"], values["loss_gan_ba"], values["loss_cycle"], values["loss_identity"],
                    grad_norm_g, grad_norm_d, lr,
                )
            if bench_steps and step >= bench_steps:
                break
        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_time = time.time() - t_epoch
        train_time += epoch_time

        if bench_steps:
            est = total_steps * (train_time / step)
            result = {
                "bench_steps": step, "device": device_name, "param_count": params,
                "train_images_per_sec": images_seen / train_time, "sec_per_step": train_time / step,
                "total_steps": total_steps, "estimated_train_sec": est, "estimated_train_min": est / 60,
                **mem.result(),
            }
            logger.info("benchmark %s", json.dumps(result))
            return result

        means = {k: (float(np.mean(v)) if v else float("nan")) for k, v in epoch_vals.items()}
        epoch_log.append({"epoch": epoch, "lr": lr, "epoch_time_sec": epoch_time, **means})
        logger.info(
            "epoch %d done loss_g %.3f loss_d_a %.3f loss_d_b %.3f cycle %.3f idt %.3f time %.1fs images/sec %.1f",
            epoch, means["loss_g"], means["loss_d_a"], means["loss_d_b"], means["loss_cycle"], means["loss_identity"],
            epoch_time, 2 * B * spe / epoch_time,
        )
        if int(tc["sample_every"]) and epoch % int(tc["sample_every"]) == 0:
            sample_grid(g_ab, g_ba, fixed_a, fixed_b, sample_dir / f"epoch_{epoch:03d}.jpg")
        torch.save(
            {
                "g_ab": g_ab.state_dict(), "g_ba": g_ba.state_dict(), "d_a": d_a.state_dict(), "d_b": d_b.state_dict(),
                "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict(), "epoch": epoch, "config": cfg,
            },
            ckpt_dir / "last.pt",
        )
        (out_dir / "history.json").write_text(json.dumps({"steps": history, "epochs": epoch_log}), encoding="utf-8")
        plot_curves(history, out_dir)

    torch.save(
        {
            "g_ab": {k: v.half() for k, v in g_ab.state_dict().items()},
            "g_ba": {k: v.half() for k, v in g_ba.state_dict().items()},
            "epoch": epochs, "config": cfg,
        },
        ckpt_dir / "generators.pt",
    )
    gn_g, gn_d = np.array(history["grad_norm_g"]), np.array(history["grad_norm_d"])
    summary = {
        "device": device_name,
        "amp_dtype": str(amp_dtype) if bool(tc["amp"]) and device.type == "cuda" else "float32",
        "amp_disabled_step": amp_disabled_step,
        "param_count": params,
        "image_size": int(cfg["data"]["image_size"]),
        "epochs": epochs,
        "steps_per_epoch": spe,
        "optimizer_steps": step,
        "batch_size": B,
        "train_time_sec": train_time,
        "train_images_per_sec": images_seen / train_time,
        "final_epoch_losses": epoch_log[-1],
        "grad_norm_g_mean": float(gn_g.mean()) if len(gn_g) else None,
        "grad_norm_g_max": float(gn_g.max()) if len(gn_g) else None,
        "grad_norm_d_mean": float(gn_d.mean()) if len(gn_d) else None,
        "grad_norm_d_max": float(gn_d.max()) if len(gn_d) else None,
        "nan_losses": nans,
        "checkpoints": {"generators.pt": {"epoch": epochs}, "last.pt": {"epoch": epochs}},
        **mem.result(),
    }
    return summary
