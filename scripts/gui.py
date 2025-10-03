#!/usr/bin/env python3
import sys
import os
from pathlib import Path

from PyQt6.QtCore import Qt, QProcess, QTimer
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QFileDialog, QSpinBox,
    QGroupBox, QPlainTextEdit, QMessageBox, QCheckBox, QComboBox
)

# Optional plotting
try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    import matplotlib.pyplot as plt
except Exception:
    FigureCanvas = None
    plt = None


class MinimalV3GUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EEGCompute V3 – Minimal Control")
        self.resize(900, 600)
        self.proc: QProcess | None = None
        self._build()

    def _build(self):
        root = QWidget(); self.setCentralWidget(root)
        lay = QVBoxLayout(root)

        # Config row
        cfg_box = QGroupBox("Config")
        cfg_lay = QHBoxLayout(cfg_box)
        self.cfg_edit = QLineEdit(str(Path("configs/v3.yaml").resolve()))
        self.btn_browse = QPushButton("Browse…")
        cfg_lay.addWidget(QLabel("Path:"))
        cfg_lay.addWidget(self.cfg_edit, 1)
        cfg_lay.addWidget(self.btn_browse)
        lay.addWidget(cfg_box)

        # Controls
        ctl_box = QGroupBox("Controls")
        ctl_lay = QHBoxLayout(ctl_box)
        self.btn_dl = QPushButton("Download COCO")
        self.btn_cal = QPushButton("Calibrate")
        self.btn_prep = QPushButton("Prepare")
        self.btn_train = QPushButton("Train")
        # Legacy buttons removed; benchmark is the main path
        self.btn_bench = QPushButton("Run Benchmark")
        ctl_lay.addWidget(self.btn_dl)
        ctl_lay.addWidget(self.btn_cal)
        ctl_lay.addWidget(self.btn_prep)
        ctl_lay.addWidget(self.btn_train)
        # ctl_lay.addWidget(self.btn_eval)
        # ctl_lay.addWidget(self.btn_all)
        ctl_lay.addWidget(self.btn_bench)
        ctl_lay.addStretch(1)
        ctl_lay.addWidget(QLabel("K:"))
        self.spin_k = QSpinBox(); self.spin_k.setRange(1, 1000); self.spin_k.setValue(10)
        ctl_lay.addWidget(self.spin_k)
        self.chk_locked = QCheckBox("Strict Locked")
        self.chk_locked.setChecked(True)
        ctl_lay.addWidget(self.chk_locked)
        # Download options
        self.chk_dl_ann = QCheckBox("Annotations")
        self.chk_dl_ann.setChecked(True)
        self.chk_dl_val = QCheckBox("Val Images (~1GB)")
        self.chk_dl_val.setChecked(True)
        self.chk_dl_train = QCheckBox("Train Images (~18GB)")
        self.chk_dl_train.setChecked(False)
        self.chk_full = QCheckBox("Full Dataset for Prepare")
        lay.addWidget(self.chk_dl_ann)
        lay.addWidget(self.chk_dl_val)
        lay.addWidget(self.chk_dl_train)
        lay.addWidget(self.chk_full)
        lay.addWidget(ctl_box)

        # Report buttons
        rep_box = QGroupBox("Reports")
        rep_lay = QHBoxLayout(rep_box)
        self.btn_open_results = QPushButton("Open results.json")
        self.btn_open_report = QPushButton("Open v3.md")
        self.lbl_status = QLabel("Status: –")
        rep_lay.addStretch(1)
        rep_lay.addWidget(self.lbl_status)
        rep_lay.addWidget(self.btn_open_results)
        rep_lay.addWidget(self.btn_open_report)
        lay.addWidget(rep_box)

        # EEG Scores panel
        eeg_box = QGroupBox("EEG Scores (P300)")
        eeg_lay = QHBoxLayout(eeg_box)
        eeg_lay.addWidget(QLabel('Mode:'))
        self.sel_mode = QComboBox(); self.sel_mode.addItems(['brainflow','synth']); self.sel_mode.setCurrentText('brainflow')
        eeg_lay.addWidget(self.sel_mode)
        eeg_lay.addWidget(QLabel('Key:'))
        self.edit_key = QLineEdit('dog'); eeg_lay.addWidget(self.edit_key)
        eeg_lay.addWidget(QLabel('Match:'))
        self.sel_match = QComboBox(); self.sel_match.addItems(['classes','item_id'])
        eeg_lay.addWidget(self.sel_match)
        # BrainFlow options (inline, optional)
        eeg_lay.addWidget(QLabel('Board:'))
        self.edit_board = QLineEdit('synthetic'); eeg_lay.addWidget(self.edit_board)
        eeg_lay.addWidget(QLabel('Serial:'))
        self.edit_serial = QLineEdit(''); eeg_lay.addWidget(self.edit_serial)
        eeg_lay.addWidget(QLabel('Dur(s):'))
        self.edit_duration = QLineEdit('30'); eeg_lay.addWidget(self.edit_duration)
        eeg_lay.addWidget(QLabel('FS:'))
        self.edit_fs = QLineEdit(''); eeg_lay.addWidget(self.edit_fs)
        eeg_lay.addWidget(QLabel('Chans:'))
        self.edit_chans = QLineEdit('0,1,2,3'); eeg_lay.addWidget(self.edit_chans)
        self.btn_compute = QPushButton('Compute P300')
        eeg_lay.addWidget(self.btn_compute)
        lay.addWidget(eeg_box)

        # Live Training panel
        live_box = QGroupBox("Live Training (Loss)")
        live_lay = QVBoxLayout(live_box)
        btn_row = QHBoxLayout();
        self.btn_start_live = QPushButton("Start Live Plot"); self.btn_stop_live = QPushButton("Stop")
        self.btn_open_csv = QPushButton("Open CSV")
        btn_row.addWidget(self.btn_start_live); btn_row.addWidget(self.btn_stop_live); btn_row.addWidget(self.btn_open_csv); btn_row.addStretch(1)
        live_lay.addLayout(btn_row)
        self.live_note = QLabel("Logs at output/logs/train_loss.csv")
        live_lay.addWidget(self.live_note)
        self.canvas = None
        if FigureCanvas and plt:
            self.fig = plt.figure(figsize=(6,2.5))
            self.canvas = FigureCanvas(self.fig)
            live_lay.addWidget(self.canvas)
        else:
            self.lbl_live = QLabel("matplotlib not available. Showing text-only updates.")
            live_lay.addWidget(self.lbl_live)
        lay.addWidget(live_box)

        # Log
        log_box = QGroupBox("Logs")
        log_lay = QVBoxLayout(log_box)
        self.log = QPlainTextEdit(); self.log.setReadOnly(True)
        log_lay.addWidget(self.log)
        lay.addWidget(log_box, 1)

        # Wiring
        self.btn_browse.clicked.connect(self._pick_cfg)
        self.btn_dl.clicked.connect(self._do_download)
        self.btn_cal.clicked.connect(self._do_cal)
        self.btn_prep.clicked.connect(self._do_prep)
        self.btn_train.clicked.connect(self._do_train)
        # self.btn_eval.clicked.connect(self._do_eval)
        # self.btn_all.clicked.connect(self._do_all)
        self.btn_bench.clicked.connect(self._do_bench)
        self.btn_open_results.clicked.connect(self._open_results)
        self.btn_open_report.clicked.connect(self._open_report)
        self.btn_compute.clicked.connect(self._do_compute)
        self.btn_start_live.clicked.connect(self._start_live)
        self.btn_stop_live.clicked.connect(self._stop_live)
        self.btn_open_csv.clicked.connect(self._open_csv)
        self.live_timer = QTimer(self); self.live_timer.setInterval(1500); self.live_timer.timeout.connect(self._refresh_plot)

    # Helpers
    def _append(self, text: str):
        self.log.appendPlainText(text.rstrip("\n"))
        sb = self.log.verticalScrollBar(); sb.setValue(sb.maximum())

    def _start(self, args: list[str]):
        if self.proc is not None:
            QMessageBox.warning(self, "Busy", "Another process is running.")
            return
        self._append("Running: "+" ".join(args))
        self.proc = QProcess(self)
        self.proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.proc.readyReadStandardOutput.connect(self._read)
        self.proc.finished.connect(self._done)
        py = sys.executable
        self.proc.start(py, args)
        self._set_enabled(False)

    def _read(self):
        if self.proc:
            out = bytes(self.proc.readAllStandardOutput()).decode(errors='ignore')
            if out:
                self._append(out)

    def _done(self):
        code = self.proc.exitCode() if self.proc else -1
        self._append(f"\nProcess finished with exit code {code}")
        self.proc = None
        self._set_enabled(True)
        # Try to parse results.json for PASS summary
        try:
            import json
            p = Path("output/reports/results.json")
            if p.exists():
                with p.open('r') as f:
                    res = json.load(f)
                overall = res.get('PASS',{}).get('overall')
                rsvp = res.get('PASS',{}).get('RSVP')
                txt = f"Status: RSVP={'✅' if rsvp else '❌'} • Overall={'✅' if overall else '❌'}"
                # If checklist exists, show compact counts
                ch = res.get('checklist',{})
                if ch:
                    passed = sum(1 for v in ch.values() if v.get('pass'))
                    total = len(ch)
                    txt += f" • Checklist: {passed}/{total}"
                self.lbl_status.setText(txt)
        except Exception as e:
            self._append(f"Could not parse results.json: {e}")

    def _set_enabled(self, enabled: bool):
        # Enable/disable only widgets that exist (legacy controls may be absent)
        names = [
            'btn_browse','btn_cal','btn_prep','btn_train','btn_bench',
            'btn_open_results','btn_open_report','spin_k','chk_locked',
            'btn_compute','sel_mode','sel_match','edit_key',
            'btn_start_live','btn_stop_live','btn_open_csv','chk_full',
            'edit_board','edit_serial','edit_duration','edit_fs','edit_chans'
        ]
        for name in names:
            w = getattr(self, name, None)
            if w is not None:
                w.setEnabled(enabled)

    # Actions
    def _pick_cfg(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select config", str(Path("configs").resolve()), "Config (*.yaml *.yml *.json);;All Files (*)")
        if path:
            self.cfg_edit.setText(path)

    def _do_cal(self):
        self._start(["scripts/control.py","calibrate"]) 

    def _do_download(self):
        # Build download command according to checkboxes
        args=["scripts/control.py","download"]
        args += ["--ann", "1" if self.chk_dl_ann.isChecked() else "0"]
        args += ["--val", "1" if self.chk_dl_val.isChecked() else "0"]
        args += ["--train", "1" if self.chk_dl_train.isChecked() else "0"]
        self._start(args) 

    def _do_prep(self):
        cfg=self.cfg_edit.text().strip()
        args=["scripts/control.py","prepare","--config",cfg]
        if self.chk_full.isChecked():
            args += ["--full","1"]
        self._start(args)

    def _do_train(self):
        cfg=self.cfg_edit.text().strip()
        self._start(["scripts/control.py","train","--config",cfg])

    def _do_eval(self):
        cfg=self.cfg_edit.text().strip()
        k=str(self.spin_k.value())
        # control.py currently always uses strict_locked=1; if unchecked, call eval.py directly
        if self.chk_locked.isChecked():
            self._start(["scripts/control.py","eval","--config",cfg,"--k",k])
        else:
            self._start(["scripts/eval.py","--config",cfg,"--k",k,"--strict_locked","0"]) 

    def _do_all(self):
        # chain: calibrate → prepare → train → eval
        # run as a single shell process to keep it simple
        cfg=self.cfg_edit.text().strip(); k=str(self.spin_k.value())
        cmd = (
            f"{sys.executable} scripts/control.py calibrate && "
            f"{sys.executable} scripts/control.py prepare --config {cfg} && "
            f"{sys.executable} scripts/control.py train --config {cfg} && "
            f"{sys.executable} scripts/control.py eval --config {cfg} --k {k}"
        )
        # Use QProcess with shell
        self._append("Running pipeline…")
        self.proc = QProcess(self)
        self.proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.proc.readyReadStandardOutput.connect(self._read)
        self.proc.finished.connect(self._done)
        # macOS/Linux /bin/sh
        self.proc.start("/bin/sh", ["-lc", cmd])
        self._set_enabled(False)

    def _do_bench(self):
        cfg=self.cfg_edit.text().strip()
        # Use full dataset if checked
        full = '1' if getattr(self,'chk_full',None) and self.chk_full.isChecked() else '0'
        args=["scripts/pipeline.py","--config",cfg,"--full",full]
        self._start(args)

    def _do_compute(self):
        cfg=self.cfg_edit.text().strip()
        mode=self.sel_mode.currentText().strip()
        key=self.edit_key.text().strip() or 'dog'
        match=self.sel_match.currentText().strip()
        args=["scripts/compute_p300.py","--config",cfg,"--mode",mode,"--key",key,"--match",match,"--out","data/processed/scores_p300.jsonl"]
        if mode=="brainflow":
            board=self.edit_board.text().strip() or 'synthetic'
            serial=self.edit_serial.text().strip()
            dur=self.edit_duration.text().strip() or '30'
            fs=self.edit_fs.text().strip()
            chans=self.edit_chans.text().strip()
            args += ["--board-id", board, "--duration", dur]
            if serial:
                args += ["--serial", serial]
            if fs:
                args += ["--fs", fs]
            if chans:
                args += ["--channels", chans]
        self._start(args)

    # Live training helpers
    def _start_live(self):
        self.live_timer.start()
        self._append("Live plot started")

    def _stop_live(self):
        self.live_timer.stop()
        self._append("Live plot stopped")

    def _open_csv(self):
        p = Path("output/logs/train_loss.csv").resolve()
        if not p.exists():
            QMessageBox.information(self, "train_loss.csv", "Not found. Start training to generate logs.")
            return
        import webbrowser
        webbrowser.open(p.as_uri())

    def _refresh_plot(self):
        try:
            p = Path("output/logs/train_loss.csv")
            if not p.exists():
                return
            rows = p.read_text().strip().splitlines()
            if len(rows) <= 1:
                return
            hdr = rows[0].split(',')
            data = [r.split(',') for r in rows[1:]]
            # expect: epoch,split,model,loss
            series = {}
            for ep, split, model, loss in data:
                key = f"{model}-{split}"
                series.setdefault(key, []).append((int(ep), float(loss)))
            # plot if possible
            if self.canvas and plt:
                self.fig.clear(); ax = self.fig.add_subplot(111)
                for k, pts in series.items():
                    pts.sort(key=lambda x: x[0])
                    xs=[x for x,_ in pts]; ys=[y for _,y in pts]; ax.plot(xs, ys, label=k)
                ax.set_xlabel('Epoch'); ax.set_ylabel('Loss'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
                self.canvas.draw()
            else:
                # text-only summary
                parts=[f"{k}: last={pts[-1][1]:.4f} at ep{pts[-1][0]}" for k,pts in series.items()]
                self.lbl_live.setText(' | '.join(parts))
        except Exception as e:
            self._append(f"Plot refresh error: {e}")

    def _open_results(self):
        p = Path("output/reports/results.json").resolve()
        if not p.exists():
            QMessageBox.information(self, "results.json", "Not found. Run Evaluate first.")
            return
        import webbrowser
        webbrowser.open(p.as_uri())
        self._append(f"Opened {p}")

    def _open_report(self):
        p = Path("output/reports/v3.md").resolve()
        if not p.exists():
            QMessageBox.information(self, "v3.md", "Not found. Run Evaluate first.")
            return
        import webbrowser
        webbrowser.open(p.as_uri())
        self._append(f"Opened {p}")


def main():
    app = QApplication(sys.argv)
    w = MinimalV3GUI(); w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
