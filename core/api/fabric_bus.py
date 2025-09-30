from __future__ import annotations
import os, asyncio, json, time
from dataclasses import asdict
from typing import Dict, List
from collections import deque
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import numpy as np

from .schema import ScoreVector
from ..logging.score_logger import ScoreLogger
from ..monitoring.system_monitor import SystemMonitor

app = FastAPI(title="EEGCompute Fabric Bus v1.5")

# v1.5: No web dashboard - use PyQt6 GUI instead

score_queue: "asyncio.Queue[ScoreVector]" = asyncio.Queue(maxsize=1024)
latest_scores: Dict[str, ScoreVector] = {}

# v1.5: Latency tracking
latency_history: Dict[str, deque] = {
    "p300": deque(maxlen=1000),
    "ssvep": deque(maxlen=1000),
    "errp": deque(maxlen=1000),
}

# v1.5: Logging and monitoring
score_logger: ScoreLogger = None
system_monitor: SystemMonitor = None

@app.on_event("startup")
async def startup():
    global score_logger, system_monitor
    score_logger = ScoreLogger()
    system_monitor = SystemMonitor()
    print(f"[v1.5] Session ID: {score_logger.session_id}")
    print(f"[v1.5] Logs: {score_logger.session_dir}")

@app.on_event("shutdown")
async def shutdown():
    if score_logger:
        score_logger.close()
    if system_monitor:
        summary = system_monitor.get_summary()
        print(f"[v1.5] Session summary: {summary}")
        system_monitor.close()

@app.get("/")
def index():
    """v1.5: API-only, no web dashboard. Use PyQt6 GUI: python scripts/launcher_gui_qt.py"""
    return {
        "name": "EEGCompute Fabric Bus v1.5",
        "status": "running",
        "endpoints": {
            "GET /latest": "Latest scores for all signal types",
            "GET /metrics": "Latency and system metrics",
            "GET /events": "Server-Sent Events stream",
            "POST /publish": "Publish a new score",
        },
        "gui": "Launch PyQt6 GUI: python scripts/launcher_gui_qt.py"
    }

@app.post("/publish")
async def publish(req: Request):
    payload = await req.json()

    # Add publish timestamp
    payload['publish_time'] = time.time()

    sv = ScoreVector(**payload)
    latest_scores[sv.kind] = sv

    # Track latency
    latencies = sv.compute_latency()
    if 'e2e_ms' in latencies:
        latency_history[sv.kind].append(latencies['e2e_ms'])

    # Log to JSONL
    if score_logger:
        score_logger.log_score(sv)

    try:
        score_queue.put_nowait(sv)
    except asyncio.QueueFull:
        pass

    return {"ok": True, "latency": latencies}

@app.get("/latest")
def latest():
    return {k: asdict(v) for k, v in latest_scores.items()}

@app.get("/stream")
async def stream(kind: str = "p300"):
    sv = latest_scores.get(kind)
    return asdict(sv) if sv else None

@app.get("/metrics")
def metrics():
    """v1.5: System metrics and latency statistics"""
    metrics_data = {}

    # Latency statistics
    for kind, history in latency_history.items():
        if history:
            latencies = list(history)
            metrics_data[f"{kind}_latency"] = {
                "mean_ms": float(np.mean(latencies)),
                "p50_ms": float(np.percentile(latencies, 50)),
                "p95_ms": float(np.percentile(latencies, 95)),
                "p99_ms": float(np.percentile(latencies, 99)),
                "max_ms": float(np.max(latencies)),
                "count": len(latencies),
            }

    # System metrics
    if system_monitor:
        system_metrics = system_monitor.log_metrics()
        if system_metrics:
            metrics_data["system"] = system_metrics

    return metrics_data

@app.get("/events")
async def events():
    """Server-Sent Events endpoint for real-time updates"""
    async def event_generator():
        while True:
            try:
                # Wait for new data from the queue
                sv = await asyncio.wait_for(score_queue.get(), timeout=1.0)

                # Send the new score as an SSE event
                data = json.dumps(asdict(sv))
                yield f"data: {data}\n\n"

            except asyncio.TimeoutError:
                # Send a heartbeat to keep connection alive
                yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"
            except Exception as e:
                print(f"SSE error: {e}")
                break

    return StreamingResponse(
        event_generator(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
        }
    )
