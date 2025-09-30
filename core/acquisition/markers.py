from __future__ import annotations
import time, random
from pylsl import StreamInfo, StreamOutlet

def start_marker_outlet(name="Markers", stype="Markers", source_id="markers"):
    info = StreamInfo(name, stype, 1, 0, 'string', source_id)
    outlet = StreamOutlet(info); return outlet

def emit_loop_rsvp(outlet, rate_hz=8.0):
    isi = 1.0/rate_hz
    vocab = [f"id_{i:03d}" for i in range(1,101)]
    while True:
        outlet.push_sample([random.choice(vocab)])
        time.sleep(isi)

def emit_loop_ssvep(outlet, rate_hz=2.0):
    isi = 1.0/rate_hz
    targets = ["31.0","32.0","33.0","34.5"]
    while True:
        outlet.push_sample([random.choice(targets)])
        time.sleep(isi)
