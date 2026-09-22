import json
import logging
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import psutil
import torch

SRC_DIR = Path(__file__).resolve().parent
MEMBER_DIR = SRC_DIR.parent
TASK_DIR = MEMBER_DIR.parent
REPO_DIR = TASK_DIR.parent
MEMBER = MEMBER_DIR.name
TASK = TASK_DIR.name
RAW_LOG_DIR = REPO_DIR / "reproducibility" / "raw_logs"
MANIFEST_DIR = REPO_DIR / "reproducibility" / "manifests"


def now_id():
    return time.strftime("%Y%m%d_%H%M%S")


def setup_logger(run_id):
    RAW_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = RAW_LOG_DIR / f"{MEMBER}_{TASK}_{run_id}.log"
    logger = logging.getLogger(run_id)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger, log_path


def _run(cmd):
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        return out.stdout.strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def cpu_name():
    if platform.system() == "Linux":
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except Exception:
            pass
    return platform.processor() or platform.machine()


def hardware_info():
    return {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cpu": cpu_name(),
        "cpu_cores": os.cpu_count(),
        "ram_gb": round(psutil.virtual_memory().total / 1e9, 1),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "nvidia_smi": _run(["nvidia-smi"]) if torch.cuda.is_available() else "no gpu",
    }


def package_versions():
    return _run([sys.executable, "-m", "pip", "freeze"]).splitlines()


def git_state():
    return {
        "commit": _run(["git", "-C", str(REPO_DIR), "rev-parse", "HEAD"]),
        "branch": _run(["git", "-C", str(REPO_DIR), "rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": bool(_run(["git", "-C", str(REPO_DIR), "status", "--porcelain"])),
    }


class PeakMemory:
    def __init__(self, device):
        self.cuda = device.type == "cuda"
        self.proc = psutil.Process()
        self.peak_rss = 0
        if self.cuda:
            torch.cuda.reset_peak_memory_stats(device)
        self.update()

    def update(self):
        rss = self.proc.memory_info().rss
        if rss > self.peak_rss:
            self.peak_rss = rss

    def result(self):
        if self.cuda:
            return {
                "peak_memory_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1),
                "peak_memory_source": "cuda_max_memory_allocated",
            }
        self.update()
        return {
            "peak_memory_mb": round(self.peak_rss / 2**20, 1),
            "peak_memory_source": "process_rss",
        }


def write_manifest(run_id, payload):
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    path = MANIFEST_DIR / f"{MEMBER}_{TASK}_{run_id}.json"
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path
