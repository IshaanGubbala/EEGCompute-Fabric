"""v1 host demo: build epochs around RSVP markers and publish P300 score (stubbed training).

Can be executed as:
  - `python scripts/host_rsvp_demo.py` (from repo root), or
  - `python -m scripts.host_rsvp_demo` (from repo root)
"""
import os, sys, time, requests, numpy as np

# Ensure project root is importable when run as a script
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.host.windows import Ring, epoch_relative
from core.scoring.p300_riemann import P300Riemann

fs = 512
ring = Ring(maxlen=fs*10)
clf = P300Riemann()  # Will return random values for demo

def fake_stream_batch(n=64):
    t0 = time.time()
    ts = np.linspace(t0, t0 + (n-1)/fs, n)
    x = np.random.randn(n)
    return ts, x

def fake_marker_time(rate_hz=8):
    return time.time()

while True:
    ts, x = fake_stream_batch()
    ring.extend(ts, x)
    evt = fake_marker_time()

    # v1.5: Track EEG sample time
    eeg_last_sample_time = ts[-1]

    # v1.5: Track processing start
    processing_start_time = time.time()

    X = epoch_relative(ring, evt, pre_s=0.2, post_s=0.8, fs=fs)
    if X is None:
        continue
    if X.ndim == 1:
        X = X[None, :]  # (n_ch=1, n_times)
    p = clf.score_epoch(X)  # Will return random values
    quality = np.random.uniform(0.7, 0.95)  # Variable quality score

    # v1.5: Payload with latency tracking
    payload = {
        "t0": float(evt-0.2),
        "dt": 1.0,
        "kind":"p300",
        "value": float(p),
        "quality": float(quality),
        "meta":{"item_id":"id_001"},
        "eeg_last_sample_time": float(eeg_last_sample_time),
        "processing_start_time": float(processing_start_time),
    }

    try:
        requests.post("http://localhost:8008/publish", json=payload, timeout=1.0)
    except Exception:
        pass
    time.sleep(0.2)
