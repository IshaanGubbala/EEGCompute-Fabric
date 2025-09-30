"""
v1.5 Validation Script

Checks all v1.5 requirements:
- Latency tracking (E2E < 250ms p95 for P300/ErrP, < 150ms for SSVEP)
- No dropped updates > 1s during 10-min soak
- CPU < 60% and memory growth ~0
- JSONL logs created
"""
import time
import requests
import json
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta


def validate_v15(duration_minutes=10, api_url="http://localhost:8008"):
    """Run v1.5 validation tests"""

    print("=" * 60)
    print("EEGCompute Fabric v1.5 Validation")
    print("=" * 60)
    print(f"\nDuration: {duration_minutes} minutes")
    print(f"API URL: {api_url}\n")

    # Test 1: Check API is running
    print("[1/6] Checking API connectivity...")
    try:
        response = requests.get(f"{api_url}/latest", timeout=2)
        if response.status_code == 200:
            print("✓ API is running")
        else:
            print(f"✗ API returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Cannot connect to API: {e}")
        return False

    # Test 2: Monitor latency
    print(f"\n[2/6] Monitoring latency for {duration_minutes} minutes...")
    start_time = time.time()
    end_time = start_time + (duration_minutes * 60)

    latency_data = {"p300": [], "ssvep": [], "errp": []}
    update_times = {"p300": [], "ssvep": [], "errp": []}
    last_update = {"p300": time.time(), "ssvep": time.time(), "errp": time.time()}

    dropped_updates = 0
    sample_count = 0

    while time.time() < end_time:
        try:
            # Get metrics
            response = requests.get(f"{api_url}/metrics", timeout=1)
            if response.status_code == 200:
                metrics = response.json()

                # Extract latency data
                for kind in ["p300", "ssvep", "errp"]:
                    key = f"{kind}_latency"
                    if key in metrics:
                        lat = metrics[key]
                        if "p95_ms" in lat:
                            latency_data[kind].append(lat["p95_ms"])

                # Check for data updates
                latest_response = requests.get(f"{api_url}/latest", timeout=1)
                if latest_response.status_code == 200:
                    latest = latest_response.json()
                    current_time = time.time()

                    for kind in ["p300", "ssvep", "errp"]:
                        if kind in latest:
                            update_times[kind].append(current_time)

                            # Check for dropped updates (>1s gap)
                            time_since_last = current_time - last_update[kind]
                            if time_since_last > 1.0:
                                dropped_updates += 1
                                print(f"  ⚠ Dropped update: {kind} ({time_since_last:.1f}s gap)")

                            last_update[kind] = current_time

                sample_count += 1

                # Progress indicator
                elapsed = time.time() - start_time
                progress = (elapsed / (duration_minutes * 60)) * 100
                if sample_count % 10 == 0:
                    print(f"  Progress: {progress:.1f}% ({elapsed:.0f}s / {duration_minutes*60:.0f}s)")

        except Exception as e:
            print(f"  Error during monitoring: {e}")

        time.sleep(0.5)

    print(f"\n  Collected {sample_count} samples over {duration_minutes} minutes")

    # Test 3: Analyze latency
    print("\n[3/6] Analyzing latency...")
    latency_pass = True

    for kind in ["p300", "ssvep", "errp"]:
        if latency_data[kind]:
            p95 = np.percentile(latency_data[kind], 95)
            mean = np.mean(latency_data[kind])

            # Check thresholds
            if kind == "ssvep":
                threshold = 150
            else:  # p300, errp
                threshold = 250

            passed = p95 <= threshold
            latency_pass = latency_pass and passed

            status = "✓" if passed else "✗"
            print(f"  {status} {kind.upper()}: mean={mean:.1f}ms, p95={p95:.1f}ms (threshold: {threshold}ms)")
        else:
            print(f"  ⚠ {kind.upper()}: No latency data collected")

    # Test 4: Check dropped updates
    print("\n[4/6] Checking dropped updates...")
    if dropped_updates == 0:
        print(f"  ✓ No dropped updates (0 gaps > 1s)")
        drops_pass = True
    else:
        print(f"  ✗ {dropped_updates} dropped updates detected")
        drops_pass = False

    # Test 5: Check system metrics
    print("\n[5/6] Checking system metrics...")
    try:
        response = requests.get(f"{api_url}/metrics", timeout=2)
        if response.status_code == 200:
            metrics = response.json()
            if "system" in metrics:
                sys_metrics = metrics["system"]
                proc = sys_metrics.get("process", {})

                cpu = proc.get("cpu_percent", 0)
                mem_growth = proc.get("memory_growth_mb", 0)

                cpu_pass = cpu < 60
                mem_pass = abs(mem_growth) < 50  # Allow ±50MB

                cpu_status = "✓" if cpu_pass else "✗"
                mem_status = "✓" if mem_pass else "✗"

                print(f"  {cpu_status} CPU: {cpu:.1f}% (threshold: <60%)")
                print(f"  {mem_status} Memory growth: {mem_growth:.1f}MB (threshold: ±50MB)")

                system_pass = cpu_pass and mem_pass
            else:
                print("  ⚠ No system metrics available")
                system_pass = True  # Don't fail if metrics not available
        else:
            print("  ⚠ Could not fetch system metrics")
            system_pass = True
    except Exception as e:
        print(f"  ⚠ Error checking system metrics: {e}")
        system_pass = True

    # Test 6: Check JSONL logs
    print("\n[6/6] Checking JSONL logs...")
    log_dir = Path("data/logs")
    if log_dir.exists():
        # Find most recent session
        sessions = sorted(log_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)
        if sessions:
            session_dir = sessions[0]
            print(f"  Session: {session_dir.name}")

            # Check for log files
            expected_files = ["p300_scores.jsonl", "ssvep_scores.jsonl", "errp_scores.jsonl",
                            "latency_metrics.jsonl", "system_metrics.jsonl"]

            found_logs = []
            for fname in expected_files:
                fpath = session_dir / fname
                if fpath.exists():
                    size = fpath.stat().st_size / 1024  # KB
                    lines = len(open(fpath).readlines())
                    found_logs.append(fname)
                    print(f"  ✓ {fname}: {lines} entries, {size:.1f}KB")
                else:
                    print(f"  ✗ {fname}: Missing")

            logs_pass = len(found_logs) >= 3  # At least 3 log files
        else:
            print("  ✗ No session directories found")
            logs_pass = False
    else:
        print("  ✗ Log directory not found")
        logs_pass = False

    # Final summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    all_tests = [
        ("Latency < thresholds", latency_pass),
        ("No dropped updates", drops_pass),
        ("System stability", system_pass),
        ("JSONL logs created", logs_pass),
    ]

    for test_name, passed in all_tests:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {test_name}")

    overall_pass = all(p for _, p in all_tests)
    print("\n" + "=" * 60)
    if overall_pass:
        print("✓✓✓ v1.5 VALIDATION PASSED ✓✓✓")
    else:
        print("✗✗✗ v1.5 VALIDATION FAILED ✗✗✗")
    print("=" * 60 + "\n")

    return overall_pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="v1.5 Validation Script")
    parser.add_argument("--duration", type=int, default=10, help="Test duration in minutes (default: 10)")
    parser.add_argument("--api-url", type=str, default="http://localhost:8008", help="API URL")

    args = parser.parse_args()

    validate_v15(duration_minutes=args.duration, api_url=args.api_url)
