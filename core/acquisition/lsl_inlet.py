from __future__ import annotations
from pylsl import StreamInlet, resolve_stream
import numpy as np

def connect_eeg(name="EEG", stype="EEG", timeout=10):
    streams = resolve_stream('type', stype, timeout=timeout)
    if not streams:
        raise RuntimeError(f"No LSL stream found for type={stype}")
    return StreamInlet(streams[0], max_chunklen=512, recover=True)

def connect_markers(name="Markers", stype="Markers", timeout=10):
    streams = resolve_stream('type', stype, timeout=timeout)
    if not streams:
        raise RuntimeError("No LSL marker stream found")
    return StreamInlet(streams[0], max_chunklen=16, recover=True)

def pull_chunk(inlet, max_samples=512):
    chunk, ts = inlet.pull_chunk(timeout=0.0, max_samples=max_samples)
    if not chunk: return None, None
    return np.asarray(chunk), np.asarray(ts)
