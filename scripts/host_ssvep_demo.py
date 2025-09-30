"""v1 host demo: windowed SSVEP correlation and publish.

Run from repo root as:
  - `python scripts/host_ssvep_demo.py` or `python -m scripts.host_ssvep_demo`
"""
import os, sys, time, requests, numpy as np

# Ensure project root on sys.path when run as script
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.scoring.ssvep_csp import fbcca_score

fs = 512
def epoch(n_ch=8, dur=0.8):
    return np.random.randn(n_ch, int(dur*fs))

while True:
    # v1.5: Enhanced SSVEP with margin, scores, confidence
    eeg_last_sample_time = time.time()
    processing_start_time = time.time()

    X = epoch()
    k, s, ssvep_meta = fbcca_score(X, fs)

    # Build payload with enhanced metadata
    payload = {
        "t0": time.time(),
        "dt": 0.8,
        "kind": "ssvep",
        "value": float(s),
        "quality": 0.9,
        "meta": {
            "target_idx": k,
            **ssvep_meta  # includes scores, margin, target_freq, confidence
        },
        "eeg_last_sample_time": float(eeg_last_sample_time),
        "processing_start_time": float(processing_start_time),
    }

    try:
        requests.post("http://localhost:8008/publish", json=payload, timeout=1.0)
    except Exception:
        pass
    time.sleep(0.4)
