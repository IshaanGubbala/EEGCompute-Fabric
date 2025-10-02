#!/usr/bin/env python3
import sys
import os
from pathlib import Path

from PyQt6.QtCore import Qt, QProcess
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QFileDialog, QSpinBox,
    QGroupBox, QPlainTextEdit, QMessageBox, QCheckBox
)


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
        self.btn_eval = QPushButton("Evaluate")
        self.btn_all = QPushButton("Run All")
        ctl_lay.addWidget(self.btn_dl)
        ctl_lay.addWidget(self.btn_cal)
        ctl_lay.addWidget(self.btn_prep)
        ctl_lay.addWidget(self.btn_train)
        ctl_lay.addWidget(self.btn_eval)
        ctl_lay.addWidget(self.btn_all)
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
        lay.addWidget(self.chk_dl_ann)
        lay.addWidget(self.chk_dl_val)
        lay.addWidget(self.chk_dl_train)
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
        self.btn_eval.clicked.connect(self._do_eval)
        self.btn_all.clicked.connect(self._do_all)
        self.btn_open_results.clicked.connect(self._open_results)
        self.btn_open_report.clicked.connect(self._open_report)

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
        for w in [self.btn_browse,self.btn_cal,self.btn_prep,self.btn_train,self.btn_eval,self.btn_all,self.btn_open_results,self.btn_open_report,self.spin_k,self.chk_locked]:
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
        self._start(["scripts/control.py","prepare","--config",cfg])

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
