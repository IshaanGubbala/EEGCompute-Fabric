"""
v1.5 Latency & CPU Logger

Continuously polls /metrics endpoint and logs latency/CPU data to JSONL.
Useful for long-term monitoring and offline analysis.

Usage:
    python scripts/latency_cpu_logger.py --duration 60 --output data/logs/metrics.jsonl
"""
import time
import requests
import json
import argparse
from pathlib import Path
from datetime import datetime


def log_metrics(duration_seconds: int, api_url: str, output_file: str, poll_interval: float = 1.0):
    """
    Poll /metrics endpoint and log to JSONL.

    Args:
        duration_seconds: How long to run (0 = infinite)
        api_url: Base URL of fabric bus API
        output_file: Path to output JSONL file
        poll_interval: Seconds between polls
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[latency_cpu_logger] Starting")
    print(f"  API URL: {api_url}")
    print(f"  Output: {output_file}")
    print(f"  Duration: {duration_seconds}s (0 = infinite)")
    print(f"  Poll interval: {poll_interval}s")

    start_time = time.time()
    sample_count = 0

    with open(output_path, 'a') as f:
        while True:
            # Check duration
            if duration_seconds > 0:
                elapsed = time.time() - start_time
                if elapsed >= duration_seconds:
                    print(f"\n[latency_cpu_logger] Completed {duration_seconds}s run")
                    break

            try:
                # Poll /metrics
                response = requests.get(f"{api_url}/metrics", timeout=2.0)
                if response.status_code == 200:
                    metrics = response.json()

                    # Add timestamp
                    log_entry = {
                        "timestamp": datetime.now().isoformat(),
                        "epoch_time": time.time(),
                        "metrics": metrics
                    }

                    # Write to JSONL
                    f.write(json.dumps(log_entry) + '\n')
                    f.flush()

                    sample_count += 1

                    # Progress indicator
                    if sample_count % 10 == 0:
                        elapsed = time.time() - start_time
                        print(f"  [{elapsed:.0f}s] Logged {sample_count} samples")

                else:
                    print(f"  Warning: API returned status {response.status_code}")

            except requests.exceptions.ConnectionError:
                print(f"  Warning: Cannot connect to API at {api_url}")
                time.sleep(5)  # Wait longer before retry
            except Exception as e:
                print(f"  Error: {e}")

            time.sleep(poll_interval)

    print(f"\n[latency_cpu_logger] Done")
    print(f"  Total samples: {sample_count}")
    print(f"  Output file: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="v1.5 Latency & CPU Logger")
    parser.add_argument("--duration", type=int, default=0,
                        help="Duration in seconds (0 = infinite)")
    parser.add_argument("--api-url", type=str, default="http://localhost:8008",
                        help="API base URL")
    parser.add_argument("--output", type=str, default="data/logs/metrics_live.jsonl",
                        help="Output JSONL file")
    parser.add_argument("--poll-interval", type=float, default=1.0,
                        help="Seconds between polls")

    args = parser.parse_args()

    try:
        log_metrics(
            duration_seconds=args.duration,
            api_url=args.api_url,
            output_file=args.output,
            poll_interval=args.poll_interval
        )
    except KeyboardInterrupt:
        print("\n[latency_cpu_logger] Interrupted by user")


if __name__ == "__main__":
    main()
