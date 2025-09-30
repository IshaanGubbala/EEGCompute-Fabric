from __future__ import annotations
from collections import deque
import numpy as np

class Ring:
    def __init__(self, maxlen: int):
        self.ts = deque(maxlen=maxlen); self.x = deque(maxlen=maxlen)
    def extend(self, ts_batch, x_batch):
        self.ts.extend(ts_batch); self.x.extend(x_batch)
    def window(self, t0, t1):
        ts = np.array(self.ts, float); x = np.array(self.x, float)
        if ts.size == 0: return None, None
        sel = (ts>=t0) & (ts<=t1); 
        # For multi-channel, store as columns in x; in v1 demo, x is 1D stream.
        return ts[sel], x[sel]

def epoch_relative(ring: Ring, t_event: float, pre_s: float, post_s: float, fs: int):
    t0, t1 = t_event - pre_s, t_event + post_s
    ts, x = ring.window(t0, t1)
    if ts is None or x is None or ts.size == 0: 
        return None
    n = int((pre_s + post_s) * fs)
    if n <= 0: return None
    grid = np.linspace(t0, t1, n, endpoint=False)
    # 1D demo epoch; v1.1 will stack channels
    X = np.interp(grid, ts, x)
    return X
