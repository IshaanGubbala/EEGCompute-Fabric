# EEGCompute Fabric — v1.5+

Real-time Brain-Computer Interface framework with production-ready monitoring, logging, visualization, and calibration.

## ✨ Features

### v1.5+ — Production Ready + Calibration
- ✅ **Latency Tracking**: E2E and processing latency with p50/p95/p99 statistics
- ✅ **System Monitoring**: CPU & memory tracking with psutil
- ✅ **Data Persistence**: JSONL logs for all scores, latencies, and system metrics
- ✅ **Modern PyQt6 GUI**: Beautiful dark theme with tabbed interface + QC indicators
- ✅ **Live Rolling Graph**: Real-time visualization of all 3 signals (P300, SSVEP, ErrP)
- ✅ **Validation Suite**: Automated soak tests with pass/fail criteria
- ✅ **Calibration Scripts**: P300 & ErrP model training with AUROC/AUPRC evaluation
- ✅ **SSVEP Enhancements**: Margin, confidence, and multi-target scoring
- ✅ **BIDS Export**: EEG-BIDS compliant session export
- ✅ **Offline Evaluation**: Post-session AUROC/AUPRC analysis and latency reports

### Core BCI Capabilities
- **Multi-paradigm BCI**: P300 (RSVP), SSVEP, Error Potentials (ErrP)
- **Real-time Processing**: Low-latency signal processing pipeline
- **LSL Integration**: Lab Streaming Layer for EEG data acquisition
- **REST API**: `/latest`, `/metrics`, `/events` endpoints
- **Fabric Bus**: Central message bus with session management

## 🚀 Quick Start

### 1. Install Dependencies

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Launch the GUI

```bash
python scripts/launcher_gui_qt.py
```

### 3. Start Services (in GUI)

Click buttons in this order:
1. **Start EEG** → LSL simulator (512 Hz, 8 channels)
2. **Start Markers** → Event stream for RSVP/SSVEP
3. **Start Bus** → API server on port 8008
4. **Start RSVP/SSVEP/ErrP** → Demo publishers

### 4. View Live Data

- **📊 Metrics Tab**: Current values in beautiful cards
- **📈 Live Plot Tab**: Rolling window graph with all 3 signals
- **🟢 Status**: Green = connected, Red = disconnected

### 5. Run Validation

```bash
# 10-minute full test
python scripts/validate_v15.py --duration 10

# Quick 1-minute test
python scripts/validate_v15.py --duration 1
```

## 📁 Project Structure

```
EEGCompute-Fabric/
├── core/
│   ├── acquisition/        # LSL data acquisition
│   ├── api/               # FastAPI fabric bus + schema
│   ├── host/              # Signal processing (filters, ICA, QC, epoching)
│   ├── logging/           # JSONL score logger
│   ├── monitoring/        # System resource monitoring
│   └── scoring/           # P300, SSVEP, ErrP classifiers
├── scripts/
│   ├── launcher_gui_qt.py       # PyQt6 GUI launcher with QC
│   ├── host_rsvp_demo.py        # P300 RSVP demo
│   ├── host_ssvep_demo.py       # SSVEP demo (enhanced with margin)
│   ├── host_errp_demo.py        # Error potential demo
│   ├── p300_fit.py              # P300 calibration script (NEW)
│   ├── errp_fit.py              # ErrP calibration script (NEW)
│   ├── latency_cpu_logger.py    # Continuous latency/CPU logger (NEW)
│   ├── eval_offline.py          # Offline AUROC/AUPRC evaluation (NEW)
│   ├── simulate_lsl_eeg.py      # LSL EEG simulator
│   ├── simulate_markers.py      # Event marker simulator
│   ├── validate_v15.py          # v1.5 validation suite
│   └── reranking_demo.py        # Search re-ranking app
├── data/
│   ├── logs/{session_id}/      # Session logs (auto-created)
│   │   ├── p300_scores.jsonl
│   │   ├── ssvep_scores.jsonl
│   │   ├── errp_scores.jsonl
│   │   ├── latency_metrics.jsonl
│   │   ├── system_metrics.jsonl
│   │   └── evaluation_report.json  # Offline analysis (NEW)
│   └── bids/                   # EEG-BIDS export (NEW)
│       └── sub-{subject}/ses-{session}/eeg/
├── models/                # Trained model cards (NEW)
│   ├── p300_sub-{subject}_{timestamp}.json
│   ├── p300_sub-{subject}_{timestamp}.pkl
│   ├── errp_sub-{subject}_{timestamp}.json
│   └── errp_sub-{subject}_{timestamp}.pkl
├── core/
│   ├── export/            # BIDS writer module (NEW)
│   └── ...
├── app_logic/             # Application demos
├── configs/               # Device and task configs (TODO v1.6)
├── docs/                  # Documentation
├── eval/                  # Evaluation metrics
├── tests/                 # Unit tests
├── README.md              # This file
├── V15_SUMMARY.md         # Detailed v1.5 documentation
└── requirements.txt       # Python dependencies
```

## 📊 API Endpoints

### GET /latest
Returns latest scores for all signal types:
```json
{
  "p300": {"t0": 1759191271.5, "value": 0.468, "quality": 0.72, ...},
  "ssvep": {"t0": 1759191271.6, "value": 0.024, "quality": 0.90, ...},
  "errp": {"t0": 1759191271.5, "value": 0.868, "quality": 0.85, ...}
}
```

### GET /metrics
Returns latency and system metrics:
```json
{
  "p300_latency": {
    "mean_ms": 45.2,
    "p50_ms": 42.1,
    "p95_ms": 78.5,
    "p99_ms": 95.3
  },
  "system": {
    "process": {"cpu_percent": 45.2, "memory_mb": 234.5},
    "uptime_seconds": 600.5
  }
}
```

### GET /events
Server-Sent Events stream for real-time updates.

## 📈 Data Logs

All data is automatically logged to `data/logs/{session_id}/`:

- **p300_scores.jsonl**: All P300 scores with timestamps
- **ssvep_scores.jsonl**: All SSVEP scores
- **errp_scores.jsonl**: All ErrP scores
- **latency_metrics.jsonl**: E2E and processing latencies
- **system_metrics.jsonl**: CPU, memory, uptime

Each entry is a JSON line with full metadata.

## 🎨 GUI Features

### PyQt6 Modern Interface
- **Dark gradient theme**: Beautiful space-themed colors
- **Tabbed layout**: Metrics cards + Live plot
- **Real-time updates**: 500ms polling rate
- **Connection status**: Live indicator with color coding

### Rolling Window Graph
- **3 signals on one plot**: P300 (cyan), SSVEP (purple), ErrP (orange)
- **100-point history**: ~50 seconds of data
- **Auto-scaling**: Adjusts to data range
- **Legend**: Color-coded signal names

## 🔧 Manual Mode (No GUI)

If you prefer terminals:

```bash
# Terminal 1: Simulators
python scripts/simulate_lsl_eeg.py --fs 512 --channels 8 &
python scripts/simulate_markers.py --task rsvp --rate_hz 8 &

# Terminal 2: API Server
uvicorn core.api.fabric_bus:app --host 0.0.0.0 --port 8008 --reload

# Terminal 3: Demos (pick one or more)
python scripts/host_rsvp_demo.py &
python scripts/host_ssvep_demo.py &
python scripts/host_errp_demo.py &

# Check metrics
curl http://localhost:8008/latest
curl http://localhost:8008/metrics
```

## 🎯 Calibration Workflows

### P300 Calibration

```bash
# Run 3-minute calibration (180 seconds)
python scripts/p300_fit.py --duration 180 --subject 001

# Output:
# - models/p300_sub-001_{timestamp}.json  (model card with AUROC/AUPRC)
# - models/p300_sub-001_{timestamp}.pkl   (trained model)
```

**Thresholds:**
- AUROC ≥ 0.80 (required for production)
- Saves decision thresholds at 50% and 80% sensitivity

### ErrP Calibration

```bash
# Run 3-minute calibration
python scripts/errp_fit.py --duration 180 --subject 001

# Output:
# - models/errp_sub-001_{timestamp}.json  (model card)
# - models/errp_sub-001_{timestamp}.pkl   (trained model)
```

**Thresholds:**
- AUROC ≥ 0.70 (acceptable for error detection)
- Threshold tuned at 10% FPR (high specificity)

## 📊 Offline Evaluation

After running a session, analyze the results:

```bash
# Evaluate session logs
python scripts/eval_offline.py --session data/logs/20250930_120000

# Generates:
# - data/logs/20250930_120000/evaluation_report.json
# - Latency histograms per signal type
# - AUROC/AUPRC metrics
# - Quality statistics
```

## 💾 BIDS Export

Export session data to EEG-BIDS format:

```python
from core.export import export_session_to_bids

# Export EEG data, events, and scores
session_path = export_session_to_bids(
    eeg_data=eeg_array,          # (n_channels, n_samples)
    events=[...],                # Event markers
    scores=[...],                # ScoreVector list
    fs=512.0,
    subject="001",
    task="bci"
)
# Output: data/bids/sub-001/ses-{timestamp}/eeg/
```

## 📈 Continuous Monitoring

Run latency/CPU logger for long-term monitoring:

```bash
# Monitor for 1 hour
python scripts/latency_cpu_logger.py --duration 3600 --output data/logs/metrics_live.jsonl

# Monitor indefinitely (until Ctrl+C)
python scripts/latency_cpu_logger.py --duration 0
```

## ✅ v1.5+ Validation Checklist

### Implemented ✅
- [x] E2E latency tracking (< 250ms p95 for P300/ErrP)
- [x] System monitoring (CPU < 60%, memory growth < 50MB)
- [x] JSONL persistence for all scores
- [x] PyQt6 GUI with rolling graph + QC indicators
- [x] Automated validation script
- [x] P300/ErrP calibration routines with AUROC ≥ 0.80/0.70
- [x] SSVEP margin and confidence scoring
- [x] EEG-BIDS export module
- [x] Offline AUROC/AUPRC evaluation
- [x] Continuous latency/CPU logger

### Deferred to v1.6 🚧
- [ ] Signal QC (bandpass, notch, ICLabel) in live path
- [ ] SSVEP calibration routine with margin tuning
- [ ] Config system for reproducibility
- [ ] Closed-loop demo applications (re-ranker, ErrP-bandit)
- [ ] Real MNE/mne-lsl epochers (replace demo loops)

See `V15_SUMMARY.md` for detailed implementation notes.

## 📚 Documentation

- **V15_SUMMARY.md**: Complete v1.5 feature documentation
- **docs/design.md**: Architecture and design decisions
- **docs/**: Additional technical documentation

## 🐛 Troubleshooting

### GUI won't start
```bash
# Check Qt installation
python -c "from PyQt6.QtWidgets import QApplication; print('PyQt6 OK')"

# Reinstall if needed
pip install --force-reinstall PyQt6
```

### API connection failed
- Ensure port 8008 is not in use
- Check firewall settings
- Try: `curl http://localhost:8008/latest`

### No data appearing
1. Start services in correct order (EEG → Markers → Bus → Demos)
2. Check logs in terminal/GUI
3. Verify simulators are running: `ps aux | grep simulate`

## 🎯 Performance Targets

| Metric | Target | Status |
|--------|--------|--------|
| P300/ErrP latency (p95) | < 250ms | ✅ Tracking |
| SSVEP latency (p95) | < 150ms | ✅ Tracking |
| CPU usage | < 60% | ✅ Monitored |
| Memory growth | < 50MB/10min | ✅ Monitored |
| Dropped updates | 0 gaps >1s | ✅ Validated |

## 🔮 Roadmap

### v1.6 (Next)
- Real signal processing (bandpass, notch, ICLabel)
- Calibration routines with AUROC tracking
- EEG-BIDS export
- Config system for reproducible runs

### v2.0 (Future)
- Multi-device support
- Advanced artifact rejection
- Online learning
- Cloud deployment

## 📝 License

MIT License - See LICENSE file for details

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## 📞 Support

- Issues: GitHub Issues
- Documentation: See `docs/` directory
- Validation: Run `python scripts/validate_v15.py`
