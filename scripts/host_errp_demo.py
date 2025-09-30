"""v1 host demo: post-response windows -> LDA score (placeholder training).

Run from repo root as:
  - `python scripts/host_errp_demo.py` or `python -m scripts.host_errp_demo`
"""
import os, sys, time, requests, numpy as np

# Ensure project root on sys.path when run as script
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.scoring.errp_decoder import ErrPDecoder

dec = ErrPDecoder()
X = np.random.randn(60, 16); y = np.r_[np.ones(30), np.zeros(30)]
dec.fit(X, y)

while True:
    x1 = np.random.randn(16)
    p = dec.score(x1)
    payload = {"t0": time.time(), "dt": 0.6, "kind":"errp","value": float(p), "quality": 0.85, "meta":{"action_id": 0}}
    try:
        requests.post("http://localhost:8008/publish", json=payload, timeout=1.0)
    except Exception:
        pass
    time.sleep(0.5)
