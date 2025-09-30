"""JSONL logger for ScoreVector persistence (v1.5)"""
from __future__ import annotations
import json
import os
from pathlib import Path
from datetime import datetime
from dataclasses import asdict
from typing import Optional
from ..api.schema import ScoreVector


class ScoreLogger:
    """Logs ScoreVectors to JSONL files with timestamps"""

    def __init__(self, output_dir: str = "data/logs", session_id: Optional[str] = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Generate session ID if not provided
        if session_id is None:
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_id = session_id

        # Create session directory
        self.session_dir = self.output_dir / session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)

        # Log files for each signal type
        self.log_files = {
            "p300": self.session_dir / "p300_scores.jsonl",
            "ssvep": self.session_dir / "ssvep_scores.jsonl",
            "errp": self.session_dir / "errp_scores.jsonl",
            "latency": self.session_dir / "latency_metrics.jsonl",
        }

        # Open file handles
        self.handles = {}
        for kind, path in self.log_files.items():
            self.handles[kind] = open(path, "a", encoding="utf-8")

    def log_score(self, score: ScoreVector):
        """Log a ScoreVector to its corresponding JSONL file"""
        # Convert to dict
        score_dict = asdict(score)

        # Add timestamp
        score_dict['log_timestamp'] = datetime.now().isoformat()

        # Compute and log latency
        latencies = score.compute_latency()
        if latencies:
            score_dict['latencies'] = latencies

            # Also log to latency file
            latency_entry = {
                'timestamp': datetime.now().isoformat(),
                'kind': score.kind,
                **latencies
            }
            self.handles['latency'].write(json.dumps(latency_entry) + "\n")
            self.handles['latency'].flush()

        # Write to appropriate log file
        if score.kind in self.handles:
            self.handles[score.kind].write(json.dumps(score_dict) + "\n")
            self.handles[score.kind].flush()

    def close(self):
        """Close all file handles"""
        for handle in self.handles.values():
            handle.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
