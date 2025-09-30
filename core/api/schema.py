from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, Dict, Any, Optional
import time

@dataclass
class ScoreVector:
    t0: float
    dt: float
    kind: Literal["p300","ssvep","errp"]
    value: float
    quality: float
    meta: Dict[str, Any] = field(default_factory=dict)

    # v1.5: Latency tracking
    eeg_last_sample_time: Optional[float] = None  # When last EEG sample was received
    processing_start_time: Optional[float] = None  # When processing started
    publish_time: Optional[float] = None  # When published to API

    def compute_latency(self) -> Dict[str, float]:
        """Compute E2E and processing latencies in ms"""
        latencies = {}
        if self.eeg_last_sample_time and self.publish_time:
            latencies['e2e_ms'] = (self.publish_time - self.eeg_last_sample_time) * 1000
        if self.processing_start_time and self.publish_time:
            latencies['processing_ms'] = (self.publish_time - self.processing_start_time) * 1000
        return latencies
