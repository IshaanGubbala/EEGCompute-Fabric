from __future__ import annotations
import numpy as np

def flatline_rate(x: np.ndarray, tol=1e-6):
    return float(np.mean(np.abs(np.diff(x)) < tol))

def jump_events(x: np.ndarray, z=6.0):
    dx = np.diff(x); s = np.std(dx) + 1e-9
    return int(np.sum(np.abs(dx) > z*s))

def channel_quality(x: np.ndarray):
    fr = flatline_rate(x); jumps = jump_events(x)
    qual = max(0.0, 1.0 - 0.5*fr - 0.02*jumps)
    return float(np.clip(qual, 0.0, 1.0))
