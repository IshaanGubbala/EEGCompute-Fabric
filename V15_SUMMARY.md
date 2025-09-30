# EEGCompute Fabric v1.5 — Implementation Summary

## 🎯 Overview

v1.5 adds production-ready features for latency tracking, signal quality control, data persistence, and system monitoring.

## ✅ Implemented Features

### 1. Latency & Stability Tracking

**Files:**
- `core/api/schema.py` - Enhanced ScoreVector with latency fields
- `core/api/fabric_bus.py` - `/metrics` endpoint with p50/p95/p99 latency stats
- `scripts/host_rsvp_demo.py` - Updated to track EEG sample and processing times

**Features:**
- E2E latency tracking (last EEG sample → /publish)
- Processing latency tracking
- Real-time p50/p95/p99 percentile calculation
- Latency history buffer (last 1000 samples per signal type)

**Thresholds:**
- P300/ErrP: ≤ 250ms p95
- SSVEP: ≤ 150ms p95

### 2. System Resource Monitoring

**Files:**
- `core/monitoring/system_monitor.py` - CPU & memory tracking with psutil

**Features:**
- Process CPU usage monitoring
- Memory usage and growth tracking
- System-wide resource metrics
- JSONL logging of system metrics
- Automatic stability checks (CPU < 60%, memory growth < 50MB)

### 3. Data Persistence

**Files:**
- `core/logging/score_logger.py` - JSONL persistence for ScoreVectors

**Features:**
- Per-session logging to `data/logs/{session_id}/`
- Separate JSONL files for each signal type (p300, ssvep, errp)
- Latency metrics log
- System metrics log
- Automatic session ID generation with timestamps

**Log Structure:**
```
data/logs/20250930_120000/
├── p300_scores.jsonl
├── ssvep_scores.jsonl
├── errp_scores.jsonl
├── latency_metrics.jsonl
└── system_metrics.jsonl
```

### 4. Modern PyQt6 GUI with Live Visualization

**Files:**
- `scripts/launcher_gui_qt.py` - Enhanced with pyqtgraph integration

**Features:**
- **Tabbed interface**: Metrics cards + Live plot
- **Rolling window graph**: All 3 signals (P300, SSVEP, ErrP) on one plot
- **Real-time updates**: 500ms polling with automatic graph updates
- **Color-coded signals**:
  - P300: Cyan (#06B6D4)
  - SSVEP: Purple (#8B5CF6)
  - ErrP: Orange (#F97316)
- **100-point history buffer**: Shows last ~50 seconds of data
- **Dark theme with pyqtgraph**: Modern visualization

### 5. API Enhancements

**New Endpoints:**
- `GET /metrics` - System and latency metrics
  - Per-signal latency stats (mean, p50, p95, p99, max)
  - Process metrics (CPU%, memory, threads)
  - System metrics (CPU%, available memory)

**Enhanced Endpoints:**
- `POST /publish` - Now logs latency and persists to JSONL
- Returns latency info in response

### 6. Validation Script

**Files:**
- `scripts/validate_v15.py`

**Features:**
- Automated 10-minute soak test
- Latency threshold validation
- Dropped update detection (>1s gaps)
- System stability checks
- JSONL log verification
- Detailed progress reporting
- Pass/fail summary

**Usage:**
```bash
python scripts/validate_v15.py --duration 10
```

## 📊 v1.5 Definition of Done Checklist

### Latency & Stability ✅
- [x] E2E latency tracking implemented
- [x] p95 latency < 250ms for P300/ErrP (validated via /metrics)
- [x] p95 latency < 150ms for SSVEP (validated via /metrics)
- [x] No dropped updates >1s (monitored via validation script)
- [x] CPU < 60% tracked
- [x] Memory growth ~0 tracked
- [x] psutil logging added

### Signal Path & QC 🚧 (Stub for future)
- [ ] Live bandpass (1–40 Hz) - TODO in v1.6
- [ ] Notch filter (50/60 Hz) - TODO in v1.6
- [ ] ICLabel auto-masking - TODO in v1.6
- [ ] Channel quality scores - TODO in v1.6

### Scorers & Thresholds 🚧 (Stub for future)
- [ ] P300 calibration with AUROC ≥ 0.80 - TODO in v1.6
- [ ] SSVEP margin logic - TODO in v1.6
- [ ] ErrP threshold calibration - TODO in v1.6

### Contracts & Persistence ✅
- [x] ScoreVector logs persisted as JSONL
- [x] Timestamps included
- [x] Session-based organization
- [ ] EEG-BIDS export - TODO in v1.6
- [ ] Config system (devices/tasks) - TODO in v1.6

### Apps Sanity 🚧 (Demos for future)
- [ ] Re-ranker demo with measurable lift - TODO in v1.6
- [ ] SSVEP softkeys with dwell logic - TODO in v1.6
- [ ] ErrP bandit with regret tracking - TODO in v1.6

### Visualization ✅
- [x] Rolling window graph with all 3 signals
- [x] Real-time updates
- [x] Modern PyQt6 interface

## 🚀 Quick Start Guide

### 1. Launch the Stack

```bash
# Start the PyQt6 GUI (recommended)
python scripts/launcher_gui_qt.py

# In the GUI:
# 1. Click "Start EEG" to start LSL simulator
# 2. Click "Start Markers" to start event stream
# 3. Click "Start Bus" to start API server
# 4. Click "Start RSVP", "Start SSVEP", "Start ErrP" for demos
```

### 2. View Live Data

- **Metrics Tab**: See current values and quality scores
- **Live Plot Tab**: Watch rolling window graph of all 3 signals
- **Connection Status**: Green = connected, Red = disconnected

### 3. Access API Metrics

```bash
# Get latest scores
curl http://localhost:8008/latest

# Get latency and system metrics
curl http://localhost:8008/metrics
```

### 4. Run Validation

```bash
# 10-minute soak test
python scripts/validate_v15.py --duration 10

# Quick 1-minute test
python scripts/validate_v15.py --duration 1
```

### 5. Check Logs

```bash
# Logs are saved to data/logs/{session_id}/
ls -lh data/logs/$(ls -t data/logs/ | head -1)/

# View P300 scores
cat data/logs/$(ls -t data/logs/ | head -1)/p300_scores.jsonl | head

# View latency metrics
cat data/logs/$(ls -t data/logs/ | head -1)/latency_metrics.jsonl | head
```

## 📈 Performance Targets

### Achieved in v1.5:
- ✅ E2E latency tracking infrastructure
- ✅ System monitoring (CPU, memory)
- ✅ JSONL data persistence
- ✅ Live visualization with rolling graph
- ✅ Automated validation script

### TODO for v1.6:
- Signal QC (bandpass, notch, ICLabel)
- Calibration routines (P300, SSVEP, ErrP)
- EEG-BIDS export
- Config system for reproducibility
- Demo applications with metrics

## 🔧 Technical Details

### Latency Calculation

```python
# In ScoreVector
e2e_ms = (publish_time - eeg_last_sample_time) * 1000
processing_ms = (publish_time - processing_start_time) * 1000
```

### JSONL Format

```json
{
  "t0": 1759191271.526304,
  "dt": 1.0,
  "kind": "p300",
  "value": 0.4689995518879646,
  "quality": 0.721702287458619,
  "meta": {"item_id": "id_001"},
  "eeg_last_sample_time": 1759191271.526,
  "processing_start_time": 1759191271.526,
  "publish_time": 1759191271.527,
  "log_timestamp": "2025-09-30T12:34:56.789123",
  "latencies": {
    "e2e_ms": 1.2,
    "processing_ms": 0.8
  }
}
```

### System Metrics Format

```json
{
  "timestamp": "2025-09-30T12:34:56",
  "uptime_seconds": 600.5,
  "process": {
    "cpu_percent": 45.2,
    "memory_mb": 234.5,
    "memory_growth_mb": 12.3,
    "num_threads": 8
  },
  "system": {
    "cpu_percent": 32.1,
    "memory_percent": 65.4,
    "memory_available_mb": 8192.0
  }
}
```

## 🎨 GUI Features

### New in v1.5:
1. **Tabbed Data Panel**:
   - Metrics tab: Data cards with live values
   - Live Plot tab: Rolling window graph

2. **Rolling Graph**:
   - PyQt graph integration
   - 100-point history buffer
   - Auto-scaling axes
   - Color-coded legends

3. **Enhanced Status**:
   - Connection indicator with color
   - Live timestamps on cards
   - System metrics display

## 📝 Notes

- **v1.5 Focus**: Infrastructure for monitoring and validation
- **Production Ready**: Latency tracking, logging, system monitoring
- **Future Work**: Signal QC, calibration, apps in v1.6+

## 🐛 Known Limitations

1. **Signal QC**: Not yet implemented (v1.6)
2. **Calibration**: Using random values for demo (v1.6)
3. **EEG-BIDS**: Export not yet implemented (v1.6)
4. **Config System**: Hardcoded parameters (v1.6)

## 🎯 Next Steps (v1.6)

1. Implement real signal processing:
   - Bandpass filtering (1-40 Hz)
   - Notch filter (50/60 Hz)
   - ICLabel for artifact rejection

2. Add calibration routines:
   - P300: 2-3 min calibration with AUROC ≥ 0.80
   - SSVEP: Per-target margins with false activation < 5%
   - ErrP: Threshold at ~10% FPR

3. Complete data persistence:
   - EEG-BIDS export
   - Config files for reproducibility

4. Demo applications:
   - Re-ranker with precision lift
   - SSVEP softkeys with dwell
   - ErrP bandit with regret tracking
