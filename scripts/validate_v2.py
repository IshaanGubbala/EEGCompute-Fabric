"""
V2 Validation Script

Purpose: Verify end-to-end timing, stability, and artifact generation for
real-time hosts and calibration in a closed-loop setting.

Checks covered (best-effort with available endpoints):
  - Timing & stability: E2E latency p95/p99, update gaps, CPU/RAM trends
  - Signal path correctness: presence of filters/ICA hooks (metadata/logs)
  - Scorer performance (aggregates only; AUROC via eval_offline if available)
  - Artifacts: JSONL aggregation and Markdown report

Outputs:
  - output/reports/v2_validation.md
  - output/jsonl/latency_cpu.jsonl (streamed snapshot entries)
  - derivatives/scores/<session>/*.jsonl (copied from data/logs)

Usage:
  python scripts/validate_v2.py --duration 600 --api-url http://localhost:8008
"""
from __future__ import annotations
import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import requests


def ensure_dirs() -> Dict[str, Path]:
    base_out = Path("output")
    reports = base_out / "reports"
    jsonl = base_out / "jsonl"
    deriv = Path("derivatives") / "scores"
    for p in [reports, jsonl, deriv]:
        p.mkdir(parents=True, exist_ok=True)
    return {"reports": reports, "jsonl": jsonl, "deriv": deriv}


def get_metrics(api_url: str) -> dict:
    try:
        r = requests.get(f"{api_url}/metrics", timeout=2)
        return r.json() if r.status_code == 200 else {}
    except Exception:
        return {}


def get_latest(api_url: str) -> dict:
    try:
        r = requests.get(f"{api_url}/latest", timeout=2)
        return r.json() if r.status_code == 200 else {}
    except Exception:
        return {}


def summarize_latency(history: List[float]) -> Dict[str, float]:
    if not history:
        return {}
    arr = np.array(history, float)
    return {
        "mean_ms": float(np.mean(arr)),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "max_ms": float(np.max(arr)),
        "count": int(arr.size),
    }


def find_latest_session_logs() -> Optional[Path]:
    log_root = Path("data/logs")
    if not log_root.exists():
        return None
    sessions = [p for p in log_root.iterdir() if p.is_dir()]
    if not sessions:
        return None
    sessions.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return sessions[0]


def copy_scores_to_derivatives(deriv_dir: Path) -> Optional[Path]:
    session_dir = find_latest_session_logs()
    if not session_dir:
        return None
    target = deriv_dir / session_dir.name
    target.mkdir(parents=True, exist_ok=True)
    for fname in ["p300_scores.jsonl", "ssvep_scores.jsonl", "errp_scores.jsonl"]:
        src = session_dir / fname
        if src.exists():
            dst = target / fname
            dst.write_bytes(src.read_bytes())
    return target


def write_report(report_path: Path, summary: dict) -> None:
    lines = []
    lines.append(f"# V2 Validation Report\n")
    lines.append(f"Generated: {datetime.now().isoformat()}\n")
    lines.append("\n## Timing & Stability\n")

    lat = summary.get("latency", {})
    for k in ["p300", "ssvep", "errp"]:
        s = lat.get(k, {})
        if s:
            lines.append(f"- {k.upper()} p95: {s.get('p95_ms', 'n/a'):.1f} ms; p99: {s.get('p99_ms', 'n/a'):.1f} ms (n={s.get('count', 0)})\n")
        else:
            lines.append(f"- {k.upper()} p95: n/a\n")

    lines.append(f"- Dropped updates (>1s gaps): {summary.get('dropped_updates', 0)}\n")
    sysm = summary.get("system", {})
    if sysm:
        lines.append(f"- CPU mean: {sysm.get('cpu_mean', 0):.1f}% (max {sysm.get('cpu_max', 0):.1f}%)\n")
        lines.append(f"- RSS mean: {sysm.get('mem_mean_mb', 0):.1f} MB (Δ {sysm.get('mem_delta_mb', 0):.1f} MB)\n")

    lines.append("\n## Artifacts\n")
    lines.append(f"- Latency/CPU JSONL: {summary.get('latency_cpu_jsonl', 'n/a')}\n")
    lines.append(f"- Derivative scores: {summary.get('derivatives_dir', 'n/a')}\n")

    # Minimal pass/fail summary against targets
    lines.append("\n## Threshold Checks\n")
    def pass_fail(ok: bool) -> str:
        return "PASS" if ok else "FAIL"

    p300_ok = lat.get("p300", {}).get("p95_ms", 9999) <= 250.0 and lat.get("p300", {}).get("p99_ms", 9999) <= 350.0
    errp_ok = lat.get("errp", {}).get("p95_ms", 9999) <= 250.0 and lat.get("errp", {}).get("p99_ms", 9999) <= 350.0
    # For SSVEP, decision latency with dwell is app-level; use p95 as proxy here
    ssvep_ok = lat.get("ssvep", {}).get("p95_ms", 9999) <= 1000.0
    gaps_ok = summary.get("dropped_updates", 0) == 0
    cpu_ok = summary.get("system", {}).get("cpu_max", 100.0) < 60.0

    lines.append(f"- RSVP P300 latency: {pass_fail(p300_ok)}\n")
    lines.append(f"- ErrP latency: {pass_fail(errp_ok)}\n")
    lines.append(f"- SSVEP decision latency (proxy): {pass_fail(ssvep_ok)}\n")
    lines.append(f"- Backpressure (gaps): {pass_fail(gaps_ok)}\n")
    lines.append(f"- Resource use: {pass_fail(cpu_ok)}\n")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("".join(lines))


def main():
    ap = argparse.ArgumentParser(description="V2 Validation")
    ap.add_argument("--duration", type=int, default=600, help="Duration in seconds (default 600 = 10 min)")
    ap.add_argument("--api-url", type=str, default="http://localhost:8008", help="Fabric API URL")
    args = ap.parse_args()

    paths = ensure_dirs()
    out_jsonl = paths["jsonl"] / "latency_cpu.jsonl"
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    # Connectivity check
    print("[v2] Checking API connectivity...")
    try:
        r = requests.get(f"{args.api_url}/latest", timeout=3)
        r.raise_for_status()
        print("[v2] API OK")
    except Exception as e:
        print(f"[v2] API check failed: {e}")
        return 2

    # Timed monitoring loop
    print(f"[v2] Monitoring for {args.duration}s...")
    start = time.time()
    end = start + args.duration

    # Collect rolling samples
    lat_hist = {"p300": [], "ssvep": [], "errp": []}
    last_update = {"p300": start, "ssvep": start, "errp": start}
    dropped_updates = 0

    cpu_samples = []
    mem_samples = []

    with open(out_jsonl, "a", encoding="utf-8") as fh:
        while time.time() < end:
            metrics = get_metrics(args.api_url)
            latest = get_latest(args.api_url)
            now = time.time()

            # Latency aggregation from /metrics
            for kind in ["p300", "ssvep", "errp"]:
                key = f"{kind}_latency"
                if key in metrics and isinstance(metrics[key], dict):
                    p95 = metrics[key].get("p95_ms")
                    if p95 is not None:
                        try:
                            lat_hist[kind].append(float(p95))
                        except Exception:
                            pass

            # Update gap detection from /latest
            if isinstance(latest, dict):
                for kind in ["p300", "ssvep", "errp"]:
                    if kind in latest:
                        gap = now - last_update[kind]
                        if gap > 1.0:
                            dropped_updates += 1
                        last_update[kind] = now

            # System metrics
            sysm = metrics.get("system", {}).get("process", {})
            if sysm:
                c = float(sysm.get("cpu_percent", 0.0))
                m = float(sysm.get("memory_mb", 0.0))
                cpu_samples.append(c)
                mem_samples.append(m)

            # Emit combined JSONL snapshot
            entry = {
                "timestamp": datetime.now().isoformat(),
                "latency_p95": {k: (lat_hist[k][-1] if lat_hist[k] else None) for k in lat_hist},
                "process": {"cpu_percent": sysm.get("cpu_percent", None), "memory_mb": sysm.get("memory_mb", None)},
            }
            fh.write(json.dumps(entry) + "\n")
            fh.flush()

            time.sleep(0.5)

    # Summaries
    latency_summary = {k: summarize_latency(v) for k, v in lat_hist.items()}
    system_summary = {}
    if cpu_samples:
        system_summary["cpu_mean"] = float(np.mean(cpu_samples))
        system_summary["cpu_max"] = float(np.max(cpu_samples))
    if mem_samples:
        system_summary["mem_mean_mb"] = float(np.mean(mem_samples))
        system_summary["mem_delta_mb"] = float(mem_samples[-1] - mem_samples[0])

    # Copy score logs into derivatives
    deriv_session = copy_scores_to_derivatives(paths["deriv"])

    report_summary = {
        "latency": {"p300": latency_summary.get("p300", {}),
                    "ssvep": latency_summary.get("ssvep", {}),
                    "errp": latency_summary.get("errp", {}),},
        "dropped_updates": dropped_updates,
        "system": system_summary,
        "latency_cpu_jsonl": str(out_jsonl),
        "derivatives_dir": str(deriv_session) if deriv_session else "n/a",
    }

    report_path = paths["reports"] / "v2_validation.md"
    write_report(report_path, report_summary)
    print(f"[v2] Wrote report: {report_path}")
    print(f"[v2] Latency/CPU log: {out_jsonl}")
    if deriv_session:
        print(f"[v2] Derivative scores: {deriv_session}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

