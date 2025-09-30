"""System resource monitoring (v1.5)"""
from __future__ import annotations
import psutil
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Optional
import os


class SystemMonitor:
    """Monitor CPU, memory, and system resources"""

    def __init__(self, output_dir: str = "data/logs", session_id: Optional[str] = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if session_id is None:
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_id = session_id

        self.session_dir = self.output_dir / session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)

        self.log_file = self.session_dir / "system_metrics.jsonl"
        self.handle = open(self.log_file, "a", encoding="utf-8")

        # Track current process
        self.process = psutil.Process(os.getpid())
        self.start_time = time.time()
        self.initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB

    def log_metrics(self):
        """Log current system metrics"""
        try:
            # CPU usage
            cpu_percent = self.process.cpu_percent(interval=0.1)

            # Memory usage
            mem_info = self.process.memory_info()
            memory_mb = mem_info.rss / 1024 / 1024
            memory_growth_mb = memory_mb - self.initial_memory

            # System-wide
            system_cpu = psutil.cpu_percent(interval=0.1)
            system_memory = psutil.virtual_memory()

            metrics = {
                'timestamp': datetime.now().isoformat(),
                'uptime_seconds': time.time() - self.start_time,
                'process': {
                    'cpu_percent': cpu_percent,
                    'memory_mb': memory_mb,
                    'memory_growth_mb': memory_growth_mb,
                    'num_threads': self.process.num_threads(),
                },
                'system': {
                    'cpu_percent': system_cpu,
                    'memory_percent': system_memory.percent,
                    'memory_available_mb': system_memory.available / 1024 / 1024,
                }
            }

            self.handle.write(json.dumps(metrics) + "\n")
            self.handle.flush()

            return metrics

        except Exception as e:
            print(f"Error logging metrics: {e}")
            return None

    def get_summary(self) -> dict:
        """Get summary statistics"""
        mem_info = self.process.memory_info()
        memory_mb = mem_info.rss / 1024 / 1024
        memory_growth_mb = memory_mb - self.initial_memory

        return {
            'uptime_seconds': time.time() - self.start_time,
            'cpu_percent': self.process.cpu_percent(interval=0.1),
            'memory_mb': memory_mb,
            'memory_growth_mb': memory_growth_mb,
            'passes_stability': memory_growth_mb < 50 and self.process.cpu_percent(interval=0.1) < 60,
        }

    def close(self):
        """Close log file"""
        self.handle.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
