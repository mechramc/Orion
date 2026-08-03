"""
M4 MacBook Air 8GB – LLM inference benchmark - Viraj Shah (virajsh4h on GitHub).
"""

from __future__ import annotations

import ast
import gc
import json
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class BenchmarkInterrupted(Exception):
    pass

def install_sigint_handler():
    def handler(signum, frame):
        log("⚠️  INTERRUPT received — breaking all loops now...")
        raise BenchmarkInterrupted("User pressed Ctrl+C")
    return signal.signal(signal.SIGINT, handler)

# --------------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------------

BURST_ITERATIONS = 10
SUSTAINED_MINUTES = 5
COOLDOWN_SECONDS = 60
MAX_TOKENS = 256
TEMPERATURE = 0.7
PROMPT = (
    "Write a Python function that implements the quicksort algorithm.\n\n"
    "Return only the function."
)

ORION_BIN = "../Orion/orion"
MODELS: Dict[str, Dict[str, Optional[str]]] = {
    "llama-3.2-3b": {
        "gguf": "models/llama-3.2-3b.Q4_K_M.gguf",
        "mlx": "mlx-community/Llama-3.2-3B-Instruct-4bit",
        "orion": None,
    },
    "phi-3-mini": {
        "gguf": "models/phi-3-mini.Q4_K_M.gguf",
        "mlx": "mlx-community/Phi-3-mini-4k-instruct-4bit",
        "orion": None,
    },
    "gpt2-124m": {
        "gguf": None,
        "mlx": None,
        "orion": "gpt2_124m",
    },
}

# --------------------------------------------------------------------
# Data structures
# --------------------------------------------------------------------

@dataclass
class PowerSample:
    timestamp: float
    cpu_power_w: float = 0.0
    gpu_power_w: float = 0.0
    ane_power_w: float = 0.0
    package_power_w: float = 0.0
    combined_power_w: float = 0.0
    thermal_pressure: str = "Unknown"
    cpu_freq_fraction: float = 0.0
    raw: str = ""

@dataclass
class RunRecord:
    run: int
    elapsed_sec: float
    tok_s: float
    n_tokens: int
    timestamp: float
    exit_code: Optional[int] = None
    stdout_excerpt: str = ""
    stderr_excerpt: str = ""
    energy_j: Optional[float] = None
    avg_power_w: Optional[float] = None
    cpu_energy_j: Optional[float] = None
    gpu_energy_j: Optional[float] = None
    ane_energy_j: Optional[float] = None
    package_energy_j: Optional[float] = None
    p50_latency_ms: Optional[float] = None
    p90_latency_ms: Optional[float] = None
    p99_latency_ms: Optional[float] = None
    rss_mb: Optional[float] = None
    thermal_pressure_max: str = "Unknown"
    cpu_freq_min_fraction: float = 0.0
    notes: List[str] = field(default_factory=list)

# --------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------

def which(cmd: str) -> Optional[str]:
    return shutil.which(cmd)

def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def get_power_source() -> str:
    try:
        out = subprocess.check_output(["pmset", "-g", "batt"], text=True, timeout=5)
        if "AC Power" in out: return "AC"
        if "Battery Power" in out: return "Battery"
    except Exception:
        pass
    return "Unknown"

def get_wifi_interface() -> Optional[str]:
    try:
        out = subprocess.check_output(["networksetup", "-listallhardwareports"], text=True, timeout=5)
        lines = out.splitlines()
        for i, line in enumerate(lines):
            if "Wi-Fi" in line or "AirPort" in line:
                if i + 1 < len(lines):
                    dev_line = lines[i + 1]
                    if "Device:" in dev_line:
                        return dev_line.split("Device:")[-1].strip()
    except Exception:
        pass
    return None

def check_wifi_off() -> Optional[bool]:
    iface = get_wifi_interface()
    if not iface: return None
    try:
        out = subprocess.check_output(["networksetup", "-getairportpower", iface], text=True, timeout=5)
        if "Off" in out: return True
        if "On" in out: return False
    except Exception:
        pass
    return None

def safe_loadavg() -> float:
    try:
        return os.getloadavg()[0]
    except Exception:
        return -1.0

def get_memory_pressure() -> str:
    try:
        out = subprocess.check_output(["memory_pressure"], text=True, timeout=5)
        for line in out.splitlines():
            if "System-wide memory free percentage" in line:
                return line.strip()
    except Exception:
        pass
    return "Unknown"

def parse_time_l_memory(stderr: str) -> Optional[float]:
    if not stderr: return None
    m = re.search(r"maximum resident set size \(kbytes\):\s*(\d+)", stderr)
    if m:
        return int(m.group(1)) / 1024.0
    return None

# --------------------------------------------------------------------
# Preflight validation
# --------------------------------------------------------------------

def preflight(orion_bin: str) -> List[str]:
    errors = []
    if not which("llama-cli"):
        errors.append("llama-cli not found. Install: brew install llama.cpp OR build from source")
    else:
        try:
            subprocess.run(["llama-cli", "--help"], capture_output=True, timeout=5)
        except Exception as e:
            errors.append(f"llama-cli smoke test failed: {e}")

    orion_path = os.path.abspath(orion_bin)
    if not os.path.exists(orion_path):
        errors.append(f"Orion binary not found at {orion_path}. Run 'make' in Orion repo.")
    else:
        orion_dir = os.path.dirname(orion_path)
        gpt2_blobs = os.path.join(orion_dir, "model", "blobs", "gpt2_124m")
        if not os.path.exists(gpt2_blobs):
            errors.append(f"Orion GPT-2 weights not found at {gpt2_blobs}. Run: python model/convert/hf_to_blobs_gpt2.py")
        tokenizer_json = os.path.join(orion_dir, "model", "tokenizer", "data", "vocab.json")
        if not os.path.exists(tokenizer_json):
            errors.append(f"Orion tokenizer not found at {tokenizer_json}.")

    try:
        import mlx_lm
        from mlx_lm.sample_utils import make_sampler
        _ = make_sampler(temp=0.7)
    except ImportError as e:
        errors.append(f"mlx-lm not installed or API mismatch: {e}. Run: pip install mlx-lm")

    for model_name, paths in MODELS.items():
        if paths.get("gguf") and not os.path.exists(paths["gguf"]):
            errors.append(f"GGUF missing: {paths['gguf']} (model: {model_name})")

    if not which("powermetrics"):
        errors.append("powermetrics not found (should be built into macOS)")
    else:
        try:
            test = subprocess.run(
                ["sudo", "-n", "powermetrics", "--show-all", "-n", "1", "-i", "1000"],
                capture_output=True, text=True, timeout=10
            )
            if test.returncode != 0:
                errors.append(f"powermetrics test failed: {(test.stderr or '')[:200]}")
            elif "CPU Power" not in test.stdout:
                errors.append("powermetrics output missing CPU Power")
        except Exception as e:
            errors.append(f"powermetrics test failed: {e}")

    return errors

# --------------------------------------------------------------------
# Energy + Thermal monitor
# --------------------------------------------------------------------

class EnergyMonitor:
    # Tuned to 2.5s to reduce overhead on sustained runs while maintaining burst accuracy
    def __init__(self, sample_interval: float = 2.5):
        self.sample_interval = sample_interval
        self.samples: List[PowerSample] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.enabled = False
        self.reason = ""

    def start(self):
        if not which("powermetrics"):
            self.reason = "powermetrics not found"
            return
        try:
            test = subprocess.run(
                ["sudo", "-n", "powermetrics", "--show-all", "-n", "1", "-i", "1000"],
                capture_output=True, text=True, timeout=10
            )
            if test.returncode != 0:
                self.reason = f"powermetrics test failed: {(test.stderr or '')[:200]}"
                log(f"[energy] {self.reason}")
                return
            if "CPU Power" not in test.stdout:
                self.reason = "powermetrics output missing CPU Power"
                log(f"[energy] {self.reason}")
                return
        except Exception as e:
            self.reason = f"powermetrics exception: {e}"
            log(f"[energy] {self.reason}")
            return

        self.enabled = True
        self._stop.clear()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        log("[energy] Monitor started")

    def _sample_once(self):
        cmd = ["sudo", "-n", "powermetrics", "--show-all", "-n", "1", "-i", "1000"]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            if proc.returncode != 0:
                if "Second underflow" not in proc.stderr:
                    log(f"    [energy] powermetrics error: {proc.stderr[:200]}")
                    return
            sample = self._parse(proc.stdout)
            if sample.combined_power_w > 0 or sample.cpu_power_w > 0:
                self.samples.append(sample)
        except subprocess.TimeoutExpired:
            log("    [energy] powermetrics timed out (15s) — skipping sample")
        except Exception as e:
            log(f"    [energy] Exception: {e}")

    def _parse(self, text: str) -> PowerSample:
        s = PowerSample(timestamp=time.time(), raw=text[:2000])
        for line in text.splitlines():
            if "CPU Power" in line and "HW active" not in line and "SW" not in line:
                m = re.search(r"([\d.]+)\s*(mW|W)", line)
                if m:
                    val = float(m.group(1))
                    s.cpu_power_w = val / 1000.0 if m.group(2) == "mW" else val
            if "GPU Power" in line and "HW active" not in line and "SW" not in line:
                m = re.search(r"([\d.]+)\s*(mW|W)", line)
                if m:
                    val = float(m.group(1))
                    s.gpu_power_w = val / 1000.0 if m.group(2) == "mW" else val
            if "ANE Power" in line:
                m = re.search(r"([\d.]+)\s*(mW|W)", line)
                if m:
                    val = float(m.group(1))
                    s.ane_power_w = val / 1000.0 if m.group(2) == "mW" else val
            if "Package Power" in line:
                m = re.search(r"([\d.]+)\s*(mW|W)", line)
                if m:
                    val = float(m.group(1))
                    s.package_power_w = val / 1000.0 if m.group(2) == "mW" else val
            if "Combined Power" in line:
                m = re.search(r"([\d.]+)\s*(mW|W)", line)
                if m:
                    val = float(m.group(1))
                    s.combined_power_w = val / 1000.0 if m.group(2) == "mW" else val
            if "Current pressure level" in line:
                s.thermal_pressure = line.split(":")[-1].strip()
            if "CPU Average frequency as fraction of nominal" in line:
                m = re.search(r"(\d+\.?\d*)%", line)
                if m:
                    s.cpu_freq_fraction = float(m.group(1)) / 100.0
        if s.combined_power_w == 0.0:
            s.combined_power_w = s.cpu_power_w + s.gpu_power_w + s.ane_power_w
        return s

    def _worker(self):
        while not self._stop.is_set():
            self._sample_once()
            deadline = time.time() + self.sample_interval
            while time.time() < deadline and not self._stop.is_set():
                time.sleep(0.25)

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=self.sample_interval + 3)
        log("[energy] Monitor stopped")

    def energy_between(self, t0: float, t1: float) -> Dict[str, Any]:
        fields = ["cpu_power_w", "gpu_power_w", "ane_power_w", "package_power_w", "combined_power_w"]
        results = {}
        for f in fields:
            joules = 0.0
            pts = [(s.timestamp, getattr(s, f, 0.0) or 0.0) for s in self.samples if t0 <= s.timestamp <= t1]
            if not pts:
                results[f] = 0.0
                continue
            if pts[0][0] > t0: pts.insert(0, (t0, pts[0][1]))
            if pts[-1][0] < t1: pts.append((t1, pts[-1][1]))
            for i in range(len(pts)-1):
                dt = pts[i+1][0] - pts[i][0]
                if dt > 0:
                    joules += (pts[i][1] + pts[i+1][1]) / 2 * dt
            results[f] = joules

        total = results["combined_power_w"] or results["package_power_w"] or sum(
            results[k] for k in ["cpu_power_w", "gpu_power_w", "ane_power_w"]
        )
        duration = max(0.0, t1 - t0)
        avg_power_w = total / duration if duration > 0 else 0.0

        window_samples = [s for s in self.samples if t0 <= s.timestamp <= t1]
        thermal_pressures = [s.thermal_pressure for s in window_samples if s.thermal_pressure != "Unknown"]
        thermal_max = max(set(thermal_pressures), key=thermal_pressures.count) if thermal_pressures else "Unknown"
        freq_values = [s.cpu_freq_fraction for s in window_samples if s.cpu_freq_fraction > 0]
        freq_min = min(freq_values) if freq_values else 0.0

        return {
            "energy_j": total, "avg_power_w": avg_power_w,
            "cpu_energy_j": results["cpu_power_w"], "gpu_energy_j": results["gpu_power_w"],
            "ane_energy_j": results["ane_power_w"], "package_energy_j": results["package_power_w"],
            "thermal_pressure_max": thermal_max, "cpu_freq_min_fraction": freq_min,
        }

    def summary(self) -> Dict[str, Any]:
        if not self.samples: return {"enabled": self.enabled, "sample_count": 0}
        ts = [s.timestamp for s in self.samples]
        pressures = [s.thermal_pressure for s in self.samples if s.thermal_pressure != "Unknown"]
        return {
            "enabled": self.enabled, "sample_count": len(self.samples),
            "start_ts": min(ts), "end_ts": max(ts),
            "window_sec": max(ts) - min(ts) if len(ts) > 1 else 0.0,
            "thermal_states": list(set(pressures)),
        }

# --------------------------------------------------------------------
# Backend runners
# --------------------------------------------------------------------

def run_llamacpp(model_path: str, prompt: str, max_tokens: int, temperature: float,
                 iterations: int, duration_minutes: int, device: str = "mps") -> List[RunRecord]:
    if not which("llama-cli"):
        raise RuntimeError("llama-cli not found")

    ngl = "0" if device == "cpu" else "99"
    records = []
    if duration_minutes <= 0 and iterations <= 0: return records
    
    end = time.time() + duration_minutes * 60 if duration_minutes > 0 else None
    run = 0
    model_abs = os.path.abspath(model_path)
    if not os.path.exists(model_abs):
        raise RuntimeError(f"Model not found: {model_abs}")

    while True:
        if end and time.time() >= end: break
        if not end and run >= iterations: break
        run += 1

        log(f"  [llama.cpp] Run {run}/{iterations if not end else '∞'} ({device})...")
        start = time.time()
        
        cmd = ["llama-cli", "-m", model_abs, "-p", prompt, "-n", str(max_tokens),
               "--temp", str(temperature), "-ngl", ngl,
               "--no-display-prompt", "--no-cnv", "--mlock"]
        
        if which("/usr/bin/time"):
            cmd = ["/usr/bin/time", "-l"] + cmd

        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        except subprocess.TimeoutExpired:
            log("    [llama.cpp] TIMEOUT after 300s")
            records.append(RunRecord(
                run=run, elapsed_sec=300.0, tok_s=0.0, n_tokens=0,
                timestamp=start, exit_code=-1, stderr_excerpt="Timeout after 300s",
                notes=["llama-cli timed out"]
            ))
            if run >= 3 and all(r.exit_code != 0 for r in records[-3:]): break
            continue

        elapsed = time.time() - start
        rss_mb = parse_time_l_memory(proc.stderr) if proc.stderr else None

        if proc.returncode != 0:
            err = proc.stderr[:500] if proc.stderr else proc.stdout[:500]
            log(f"    [llama.cpp] FAILED (code {proc.returncode}): {err}")
            records.append(RunRecord(
                run=run, elapsed_sec=elapsed, tok_s=0.0, n_tokens=0,
                timestamp=start, exit_code=proc.returncode, stderr_excerpt=err[:400],
                rss_mb=rss_mb, notes=["llama-cli failed — see stderr"]
            ))
            if run >= 3 and all(r.exit_code != 0 for r in records[-3:]): break
            continue

        combined = proc.stdout + proc.stderr
        tok_s = None
        n_tokens = 0

        m = re.search(r"eval time =\s*[\d.]+\s*ms\s*/\s*\d+\s*runs\s*\([^)]*?([\d.]+)\s*tokens per second\)", combined)
        if m: tok_s = float(m.group(1))

        m2 = re.search(r"eval count =\s*(\d+)", combined)
        if m2: n_tokens = int(m2.group(1))
        else: n_tokens = len(proc.stdout.split())

        if tok_s is None and elapsed > 0: tok_s = n_tokens / elapsed

        records.append(RunRecord(
            run=run, elapsed_sec=elapsed, tok_s=tok_s or 0.0, n_tokens=n_tokens,
            timestamp=start, exit_code=0, stdout_excerpt=proc.stdout[:400], 
            stderr_excerpt=proc.stderr[:400], rss_mb=rss_mb
        ))
    return records


def run_mlx(model_repo: str, prompt: str, max_tokens: int, temperature: float,
            iterations: int, duration_minutes: int) -> List[RunRecord]:
    try:
        import mlx.core as mx
        from mlx_lm import load
        from mlx_lm.generate import generate_step
        from mlx_lm.sample_utils import make_sampler
    except ImportError as e:
        raise RuntimeError(f"mlx-lm not installed correctly: {e}")

    try:
        reset_peak = mx.reset_peak_memory
        get_peak = mx.get_peak_memory
    except AttributeError:
        reset_peak = mx.metal.reset_peak_memory
        get_peak = mx.metal.get_peak_memory

    log(f"  [MLX] Loading {model_repo} (30-60s on first run)...")
    model, tokenizer = load(model_repo)
    log(f"  [MLX] Loaded. Benchmarking...")

    sampler = make_sampler(temp=temperature)
    records = []
    if duration_minutes <= 0 and iterations <= 0: return records
    
    end = time.time() + duration_minutes * 60 if duration_minutes > 0 else None
    run = 0

    while True:
        if end and time.time() >= end: break
        if not end and run >= iterations: break
        run += 1

        log(f"  [MLX] Run {run}/{iterations if not end else '∞'}...")
        prompt_tokens = mx.array(tokenizer.encode(prompt))

        latencies = []
        start = time.time()
        
        state = {"n_tokens": 0}
        heartbeat_stop = threading.Event()
        def heartbeat():
            while not heartbeat_stop.is_set():
                for _ in range(5):  # 5 intervals of 1s to allow responsive exit
                    if heartbeat_stop.is_set(): break
                    time.sleep(1)
                if not heartbeat_stop.is_set():
                    log(f"    [MLX] Alive... {state['n_tokens']} tokens")
        hb = threading.Thread(target=heartbeat, daemon=True)
        hb.start()

        try:
            for token, _ in generate_step(prompt_tokens, model, max_tokens=max_tokens, sampler=sampler):
                latencies.append(time.time())
                state["n_tokens"] += 1
                if state["n_tokens"] >= max_tokens or token == tokenizer.eos_token_id:
                    break
        finally:
            heartbeat_stop.set()

        elapsed = time.time() - start
        n_tokens = state["n_tokens"]

        if len(latencies) > 2:
            inter = [latencies[i] - latencies[i-1] for i in range(2, len(latencies))]
            inter_sorted = sorted(inter)
            p50 = inter_sorted[len(inter)//2] * 1000
            p90 = inter_sorted[int(len(inter)*0.9)] * 1000 if len(inter) > 1 else p50
            p99 = inter_sorted[int(len(inter)*0.99)] * 1000 if len(inter) > 1 else p90
        elif len(latencies) > 1:
            p50 = p90 = p99 = (latencies[-1] - latencies[0]) * 1000 / max(len(latencies)-1, 1)
        else:
            p50 = p90 = p99 = None

        tok_s = n_tokens / elapsed if elapsed > 0 else 0.0
        peak_bytes = get_peak()
        rss_mb = peak_bytes / (1024 * 1024) if peak_bytes else None
        
        records.append(RunRecord(
            run=run, elapsed_sec=elapsed, tok_s=tok_s, n_tokens=n_tokens,
            timestamp=start, p50_latency_ms=p50, p90_latency_ms=p90, 
            p99_latency_ms=p99, rss_mb=rss_mb
        ))

        reset_peak()

    del model
    del tokenizer
    gc.collect()
    reset_peak()
    return records


def run_orion(model_key: str, prompt: str, max_tokens: int, temperature: float,
              iterations: int, duration_minutes: int, orion_bin: str, ane: bool = True) -> List[RunRecord]:
    if not os.path.exists(orion_bin):
        raise RuntimeError(f"Orion binary not found: {orion_bin}")

    orion_dir = os.path.dirname(os.path.abspath(orion_bin))
    model_registry_name = MODELS[model_key]["orion"]
    
    # Pass absolute weights path instead of --model flag
    weights_path = os.path.join(orion_dir, "model", "blobs", model_registry_name)
    
    records = []
    if duration_minutes <= 0 and iterations <= 0: return records
    
    end = time.time() + duration_minutes * 60 if duration_minutes > 0 else None
    run = 0

    while True:
        if end and time.time() >= end: break
        if not end and run >= iterations: break
        run += 1

        log(f"  [Orion] Run {run}/{iterations if not end else '∞'} ({'ANE' if ane else 'CPU'})...")

        cmd = [orion_bin, "bench", "inference", 
               "--weights", weights_path,
               "--prompt", prompt, "--max_tokens", str(max_tokens)]
        if ane: cmd.append("--ane")
        
        if which("/usr/bin/time"):
            cmd = ["/usr/bin/time", "-l"] + cmd

        start = time.time()
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120, cwd=orion_dir)
        except subprocess.TimeoutExpired:
            log("    [Orion] TIMEOUT after 120s")
            records.append(RunRecord(
                run=run, elapsed_sec=120.0, tok_s=0.0, n_tokens=0,
                timestamp=start, exit_code=-1, stderr_excerpt="Timeout after 120s",
                notes=["Orion timed out"]
            ))
            if run >= 3 and all(r.exit_code != 0 for r in records[-3:]): break
            continue

        elapsed = time.time() - start
        combined = proc.stdout + proc.stderr
        rss_mb = parse_time_l_memory(proc.stderr) if proc.stderr else None

        if proc.returncode != 0:
            err = combined[:500]
            log(f"    [Orion] FAILED (code {proc.returncode}): {err}")
            records.append(RunRecord(
                run=run, elapsed_sec=elapsed, tok_s=0.0, n_tokens=0,
                timestamp=start, exit_code=proc.returncode, stderr_excerpt=err[:400],
                rss_mb=rss_mb, notes=["Orion bench inference failed"]
            ))
            if run >= 3 and all(r.exit_code != 0 for r in records[-3:]): break
            continue

        tok_s = p50 = p90 = n_tokens = None
        m = re.search(r"Decode p50:\s*([0-9.]+)\s*ms", combined)
        if m:
            p50 = float(m.group(1))
            tok_s = 1000.0 / p50 if p50 > 0 else 0.0
        m = re.search(r"Decode p90:\s*([0-9.]+)\s*ms", combined)
        if m: p90 = float(m.group(1))
        m = re.search(r"Prefill:\s*[\d.]+\s*ms\s*\((\d+)\s*tokens?\)", combined)
        if m: n_tokens = int(m.group(1))
        if n_tokens is None: n_tokens = max_tokens

        records.append(RunRecord(
            run=run, elapsed_sec=elapsed, tok_s=tok_s or 0.0, n_tokens=n_tokens,
            timestamp=start, exit_code=0, stdout_excerpt=proc.stdout[:400], 
            stderr_excerpt=proc.stderr[:400], p50_latency_ms=p50, p90_latency_ms=p90,
            rss_mb=rss_mb
        ))
    return records


def compute_statistics(records: List[RunRecord]) -> Dict[str, Any]:
    if not records: return {}
    tok_s = [r.tok_s for r in records]
    energy = [r.energy_j for r in records if r.energy_j is not None]
    p50s = [r.p50_latency_ms for r in records if r.p50_latency_ms is not None]
    p90s = [r.p90_latency_ms for r in records if r.p90_latency_ms is not None]
    p99s = [r.p99_latency_ms for r in records if r.p99_latency_ms is not None]
    rss = [r.rss_mb for r in records if r.rss_mb is not None]

    stats = {
        "count": len(records),
        "mean_tok_s": statistics.fmean(tok_s),
        "median_tok_s": statistics.median(tok_s),
        "min_tok_s": min(tok_s),
        "max_tok_s": max(tok_s),
        "stdev_tok_s": statistics.stdev(tok_s) if len(tok_s) > 1 else 0.0,
    }
    if p50s: stats["mean_p50_ms"] = statistics.fmean(p50s)
    if p90s: stats["mean_p90_ms"] = statistics.fmean(p90s)
    if p99s: stats["mean_p99_ms"] = statistics.fmean(p99s)
    if rss: stats["mean_rss_mb"] = statistics.fmean(rss)
    if energy:
        stats["mean_energy_j"] = statistics.fmean(energy)
        stats["total_energy_j"] = sum(energy)
        total_tokens = sum(r.n_tokens for r in records)
        stats["mean_energy_per_token_j"] = sum(energy) / total_tokens if total_tokens else None
        thermal_pressures = [r.thermal_pressure_max for r in records if r.thermal_pressure_max != "Unknown"]
        if thermal_pressures:
            stats["thermal_pressure_mode"] = max(set(thermal_pressures), key=thermal_pressures.count)
        freq_mins = [r.cpu_freq_min_fraction for r in records if r.cpu_freq_min_fraction > 0]
        if freq_mins:
            stats["cpu_freq_min_fraction"] = min(freq_mins)
    return stats

def attach_energy(records: List[RunRecord], monitor: EnergyMonitor):
    if not monitor.enabled or not records: return
    for rec in records:
        e = monitor.energy_between(rec.timestamp, rec.timestamp + rec.elapsed_sec)
        rec.energy_j = e["energy_j"]
        rec.avg_power_w = e["avg_power_w"]
        rec.cpu_energy_j = e["cpu_energy_j"]
        rec.gpu_energy_j = e["gpu_energy_j"]
        rec.ane_energy_j = e["ane_energy_j"]
        rec.package_energy_j = e["package_energy_j"]
        rec.thermal_pressure_max = e["thermal_pressure_max"]
        rec.cpu_freq_min_fraction = e["cpu_freq_min_fraction"]

# --------------------------------------------------------------------
# Experiment runner
# --------------------------------------------------------------------

def run_experiment(backend: str, model_key: str, prompt: str, max_tokens: int,
                   temperature: float, burst_iter: int, sustained_min: int,
                   orion_bin: str, ane: bool, use_energy: bool) -> Dict[str, Any]:
    records: List[RunRecord] = []
    monitor = EnergyMonitor() if use_energy else None
    if monitor: monitor.start()

    original_handler = signal.getsignal(signal.SIGINT)

    try:
        install_sigint_handler()

        if backend.startswith("llama.cpp"):
            gguf = MODELS[model_key]["gguf"]
            if not gguf or not os.path.exists(gguf):
                raise RuntimeError(f"GGUF missing: {gguf}")
            device = "cpu" if "cpu" in backend else "mps"
            records = run_llamacpp(gguf, prompt, max_tokens, temperature, burst_iter, sustained_min, device)
        elif backend == "mlx":
            repo = MODELS[model_key]["mlx"]
            if not repo: raise RuntimeError(f"No MLX repo for {model_key}")
            records = run_mlx(repo, prompt, max_tokens, temperature, burst_iter, sustained_min)
        elif backend == "orion":
            if model_key != "gpt2-124m":
                raise RuntimeError(f"Orion only supports gpt2-124m, not {model_key}")
            records = run_orion(model_key, prompt, max_tokens, temperature, burst_iter, sustained_min, orion_bin, ane)
        else:
            raise ValueError(f"Unknown backend: {backend}")

    except BenchmarkInterrupted:
        log("Interrupted during experiment — saving partial results...")
    finally:
        signal.signal(signal.SIGINT, original_handler)
        if monitor: monitor.stop()

    if monitor: attach_energy(records, monitor)

    return {
        "statistics": compute_statistics(records),
        "energy_monitor": monitor.summary() if monitor else {},
        "runs": [asdict(r) for r in records],
    }

# --------------------------------------------------------------------
# Main orchestrator
# --------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-energy", action="store_true")
    parser.add_argument("--orion-bin", default=ORION_BIN)
    parser.add_argument("--burst-iter", type=int, default=BURST_ITERATIONS)
    parser.add_argument("--sustained-min", type=int, default=SUSTAINED_MINUTES)
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--backends", default="orion,mlx,llama.cpp-mps,llama.cpp-cpu")
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--allow-mismatched-power", action="store_true", 
                        help="Run battery phase on AC (or vice-versa) and tag with actual source")
    args = parser.parse_args()

    if not args.skip_preflight:
        log("Running preflight checks...")
        errors = preflight(args.orion_bin)
        if errors:
            log("❌ PREFLIGHT FAILED. Fix these before running:")
            for e in errors: log(f"   - {e}")
            sys.exit(1)
        log("✅ Preflight passed. All binaries, models, and permissions verified.")
    else:
        log("⚠️  Preflight skipped.")

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    progress_file = out_root / "progress.json"
    if args.resume and progress_file.exists():
        progress = json.loads(progress_file.read_text())
        log(f"Resuming from {len([k for k,v in progress.items() if v == 'done'])} completed experiments")
    else:
        progress = {}

    env_info = {
        "timestamp": datetime.now().isoformat(),
        "power_source": get_power_source(),
        "wifi_off": check_wifi_off(),
        "loadavg": safe_loadavg(),
        "memory_pressure": get_memory_pressure(),
        "mlx_version": None,
        "llamacpp_path": which("llama-cli"),
        "orion_path": os.path.abspath(args.orion_bin),
    }
    try:
        import mlx_lm
        env_info["mlx_version"] = mlx_lm.__version__
    except: pass

    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    power_phases = ["AC", "Battery"]
    
    modes = ["burst"]
    if args.sustained_min > 0:
        modes.append("sustained")
        
    models = list(MODELS.keys())

    for power in power_phases:
        current_power = get_power_source()
        if current_power != power:
            log(f"⚠️  Expected {power} but detected {current_power}.")
            if not args.allow_mismatched_power:
                log(f"⏸  Skipping {power} phase. Unplug/plug in and re-run, or pass --allow-mismatched-power to force.")
                for mode in modes:
                    for model in models:
                        for backend in backends:
                            if (backend.startswith("llama.cpp") and not MODELS[model].get("gguf")): continue
                            if (backend == "mlx"      and not MODELS[model].get("mlx")):    continue
                            if (backend == "orion"    and not MODELS[model].get("orion")):  continue
                            progress[f"{mode}_{model}_{backend}_{power}"] = "skipped: power mismatch"
                progress_file.write_text(json.dumps(progress, indent=2))
                continue
            else:
                log(f"⚠️  --allow-mismatched-power flag is set. Proceeding but tagging results as {current_power}.")
                time.sleep(5)

        for mode in modes:
            for model in models:
                for backend in backends:
                    if backend.startswith("llama.cpp") and not MODELS[model].get("gguf"): continue
                    if backend == "mlx" and not MODELS[model].get("mlx"): continue
                    if backend == "orion" and not MODELS[model].get("orion"): continue

                    exp_key = f"{mode}_{model}_{backend}_{power}"
                    if args.resume and exp_key in progress and progress[exp_key] == "done":
                        log(f"Skipping completed: {exp_key}")
                        continue

                    log(f"\n{'='*60}")
                    log(f"EXPERIMENT: {exp_key}")
                    log(f"Actual power: {current_power} | Memory: {get_memory_pressure()}")
                    log(f"{'='*60}")

                    dur = args.sustained_min if mode == "sustained" else 0
                    iter_count = 0 if mode == "sustained" else args.burst_iter

                    try:
                        result = run_experiment(
                            backend, model, PROMPT, args.max_tokens, TEMPERATURE,
                            iter_count, dur, args.orion_bin, True, not args.no_energy
                        )

                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        fname = f"{exp_key}_{ts}.json"
                        out_file = out_root / fname
                        out_file.write_text(json.dumps({
                            "config": {
                                "model": model, "backend": backend, "mode": mode,
                                "power_source": current_power,
                                "timestamp": datetime.now().isoformat(),
                            },
                            "environment": env_info,
                            **result
                        }, indent=2))

                        progress[exp_key] = "done"
                        stats = result["statistics"]
                        log(f"✅ Mean tok/s: {stats.get('mean_tok_s',0):.1f} "
                            f"(p50: {stats.get('mean_p50_ms',0):.1f}ms) "
                            f"RSS: {stats.get('mean_rss_mb', 0):.0f}MB")
                        if result["energy_monitor"].get("enabled"):
                            log(f"✅ Mean energy/run: {stats.get('mean_energy_j',0):.2f} J")
                        if stats.get("thermal_pressure_mode"):
                            log(f"✅ Thermal state: {stats['thermal_pressure_mode']} "
                                f"(CPU freq min: {stats.get('cpu_freq_min_fraction',0):.0%})")

                    except Exception as e:
                        log(f"❌ FAILED: {e}")
                        traceback.print_exc()
                        progress[exp_key] = f"failed: {e}"
                        (out_root / f"{exp_key}_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json").write_text(
                            json.dumps({"error": str(e), "traceback": traceback.format_exc()}, indent=2)
                        )

                    progress_file.write_text(json.dumps(progress, indent=2))

                    if mode == "sustained" and progress.get(exp_key) == "done":
                        log(f"Cooling down {COOLDOWN_SECONDS}s...")
                        time.sleep(COOLDOWN_SECONDS)

    summary = {
        "environment": env_info,
        "progress": progress,
        "generated_at": datetime.now().isoformat(),
    }
    (out_root / "final_summary.json").write_text(json.dumps(summary, indent=2))
    log(f"\nAll results in {out_root}/")
    log("Done.")

if __name__ == "__main__":
    main()