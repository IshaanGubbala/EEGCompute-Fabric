"""EEGCompute Fabric Launcher GUI (PyQt6)

Modern, sleek launcher with live data visualization.

Usage:
  source .venv/bin/activate
  python scripts/launcher_gui_qt.py
"""
import os
import sys
import subprocess
import threading
import webbrowser
import requests
from datetime import datetime
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QPushButton, QLabel, QLineEdit, QComboBox,
    QTextEdit, QGroupBox, QFrame, QScrollArea, QTabWidget
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt6.QtGui import QFont, QPalette, QColor
import pyqtgraph as pg
from collections import deque

ROOT = os.path.dirname(os.path.dirname(__file__))


class Proc:
    def __init__(self, name: str, cmd: list[str]):
        self.name = name
        self.cmd = cmd
        self.p: subprocess.Popen | None = None
        self._t: threading.Thread | None = None

    def start(self, log_cb):
        if self.p and self.p.poll() is None:
            log_cb(f"[{self.name}] already running\n")
            return
        try:
            self.p = subprocess.Popen(
                self.cmd,
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except FileNotFoundError as e:
            log_cb(f"[{self.name}] failed to start: {e}\n")
            self.p = None
            return
        log_cb(f"[{self.name}] started\n")
        self._t = threading.Thread(target=self._pump, args=(log_cb,), daemon=True)
        self._t.start()

    def _pump(self, log_cb):
        assert self.p and self.p.stdout
        for line in self.p.stdout:
            log_cb(f"[{self.name}] {line}")
        rc = self.p.wait()
        log_cb(f"[{self.name}] exited with code {rc}\n")

    def stop(self, log_cb):
        if not self.p or self.p.poll() is not None:
            return
        log_cb(f"[{self.name}] stopping...\n")
        try:
            self.p.terminate()
            try:
                self.p.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.p.kill()
        finally:
            log_cb(f"[{self.name}] stopped\n")


class DataCard(QFrame):
    """Modern card widget for displaying signal data"""
    def __init__(self, title, icon, color, parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.Shape.StyledPanel)
        self.setStyleSheet(f"""
            DataCard {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(30, 30, 40, 240),
                    stop:1 rgba(40, 40, 50, 240));
                border: 1px solid rgba(100, 100, 120, 100);
                border-radius: 15px;
                padding: 20px;
            }}
        """)

        layout = QVBoxLayout()
        layout.setSpacing(12)

        # Title with icon
        title_layout = QHBoxLayout()
        title_label = QLabel(f"{icon} {title}")
        title_label.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        title_label.setStyleSheet(f"color: {color}; background: transparent;")
        title_layout.addWidget(title_label)
        title_layout.addStretch()
        layout.addLayout(title_layout)

        # Value display
        self.value_label = QLabel("--")
        self.value_label.setFont(QFont("Segoe UI", 32, QFont.Weight.Bold))
        self.value_label.setStyleSheet(f"color: {color}; background: transparent;")
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.value_label)

        # Secondary info
        self.info_label = QLabel("")
        self.info_label.setFont(QFont("Segoe UI", 11))
        self.info_label.setStyleSheet("color: rgba(255, 255, 255, 180); background: transparent;")
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.info_label)

        # QC indicator (NEW v1.5+)
        self.qc_label = QLabel("")
        self.qc_label.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        self.qc_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.qc_label)

        # Timestamp
        self.time_label = QLabel("Never updated")
        self.time_label.setFont(QFont("Segoe UI", 9))
        self.time_label.setStyleSheet("color: rgba(255, 255, 255, 100); background: transparent;")
        self.time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.time_label)

        layout.addStretch()
        self.setLayout(layout)

    def update_data(self, value, info="", quality=None):
        """
        Update card with value, info, and QC status.

        Args:
            value: Main value to display
            info: Secondary info text
            quality: Quality score 0-1 (for QC color coding)
        """
        self.value_label.setText(value)
        self.info_label.setText(info)
        self.time_label.setText(f"Updated: {datetime.now().strftime('%H:%M:%S')}")

        # Update QC indicator with color coding
        if quality is not None:
            if quality >= 0.8:
                qc_color = "#10B981"  # Green
                qc_status = "🟢 Excellent"
            elif quality >= 0.6:
                qc_color = "#F59E0B"  # Yellow/Orange
                qc_status = "🟡 Good"
            else:
                qc_color = "#EF4444"  # Red
                qc_status = "🔴 Poor"

            self.qc_label.setText(f"Quality: {qc_status} ({quality:.2f})")
            self.qc_label.setStyleSheet(f"color: {qc_color}; background: transparent;")


class LauncherWindow(QMainWindow):
    log_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("EEGCompute Fabric — Neural Interface Launcher")
        self.setGeometry(100, 100, 1400, 900)

        self.python = sys.executable
        self.procs = {}

        # Data history for graphs
        self.max_points = 100
        self.time_data = deque(maxlen=self.max_points)
        self.p300_data = deque(maxlen=self.max_points)
        self.ssvep_data = deque(maxlen=self.max_points)
        self.errp_data = deque(maxlen=self.max_points)
        self.start_time = datetime.now()

        # Setup dark theme
        self.setup_theme()

        # Main widget
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout()
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # Left panel - Controls
        left_panel = self.create_control_panel()
        main_layout.addWidget(left_panel, stretch=2)

        # Right panel - Live Data
        right_panel = self.create_data_panel()
        main_layout.addWidget(right_panel, stretch=1)

        main_widget.setLayout(main_layout)

        # Connect log signal
        self.log_signal.connect(self.append_log)

        # Start live updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_data)
        self.timer.start(500)  # Update every 500ms

    def setup_theme(self):
        """Setup modern dark theme"""
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, QColor(20, 20, 28))
        palette.setColor(QPalette.ColorRole.WindowText, QColor(255, 255, 255))
        palette.setColor(QPalette.ColorRole.Base, QColor(30, 30, 40))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor(40, 40, 50))
        palette.setColor(QPalette.ColorRole.Text, QColor(255, 255, 255))
        palette.setColor(QPalette.ColorRole.Button, QColor(50, 50, 65))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor(255, 255, 255))
        palette.setColor(QPalette.ColorRole.Highlight, QColor(99, 102, 241))
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
        self.setPalette(palette)

        # Global stylesheet
        self.setStyleSheet("""
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgb(20, 20, 28),
                    stop:1 rgb(30, 30, 40));
            }
            QGroupBox {
                font-size: 13px;
                font-weight: bold;
                color: rgba(255, 255, 255, 200);
                border: 1px solid rgba(100, 100, 120, 100);
                border-radius: 10px;
                margin-top: 10px;
                padding-top: 20px;
                background: rgba(40, 40, 50, 150);
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 5px 15px;
                background: rgba(60, 60, 75, 200);
                border-radius: 5px;
            }
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(99, 102, 241, 200),
                    stop:1 rgba(79, 82, 221, 200));
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px 20px;
                font-size: 11px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(119, 122, 255, 220),
                    stop:1 rgba(99, 102, 241, 220));
            }
            QPushButton:pressed {
                background: rgba(79, 82, 221, 250);
            }
            QLineEdit, QComboBox {
                background: rgba(50, 50, 65, 200);
                color: white;
                border: 1px solid rgba(100, 100, 120, 100);
                border-radius: 6px;
                padding: 8px;
                font-size: 11px;
            }
            QLineEdit:focus, QComboBox:focus {
                border: 1px solid rgba(99, 102, 241, 200);
            }
            QTextEdit {
                background: rgba(25, 25, 35, 250);
                color: #A9B7C6;
                border: 1px solid rgba(100, 100, 120, 100);
                border-radius: 8px;
                padding: 10px;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 10px;
            }
            QLabel {
                color: rgba(255, 255, 255, 200);
            }
        """)

    def create_control_panel(self):
        """Create left control panel"""
        panel = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(15)

        # Simulators
        sim_group = QGroupBox("🎛️ Simulators (LSL)")
        sim_layout = QGridLayout()
        sim_layout.setSpacing(10)

        # EEG
        sim_layout.addWidget(QLabel("EEG:"), 0, 0)
        self.eeg_fs = QLineEdit("512")
        self.eeg_fs.setMaximumWidth(60)
        sim_layout.addWidget(QLabel("fs:"), 0, 1)
        sim_layout.addWidget(self.eeg_fs, 0, 2)
        self.eeg_ch = QLineEdit("8")
        self.eeg_ch.setMaximumWidth(50)
        sim_layout.addWidget(QLabel("#ch:"), 0, 3)
        sim_layout.addWidget(self.eeg_ch, 0, 4)

        btn_start_eeg = QPushButton("Start EEG")
        btn_start_eeg.clicked.connect(self.start_eeg)
        sim_layout.addWidget(btn_start_eeg, 0, 5)

        btn_stop_eeg = QPushButton("Stop EEG")
        btn_stop_eeg.clicked.connect(self.stop_eeg)
        sim_layout.addWidget(btn_stop_eeg, 0, 6)

        # Markers
        sim_layout.addWidget(QLabel("Markers:"), 1, 0)
        self.mark_task = QComboBox()
        self.mark_task.addItems(["rsvp", "ssvep"])
        sim_layout.addWidget(QLabel("task:"), 1, 1)
        sim_layout.addWidget(self.mark_task, 1, 2)
        self.mark_rate = QLineEdit("8.0")
        self.mark_rate.setMaximumWidth(50)
        sim_layout.addWidget(QLabel("rate:"), 1, 3)
        sim_layout.addWidget(self.mark_rate, 1, 4)

        btn_start_mark = QPushButton("Start Markers")
        btn_start_mark.clicked.connect(self.start_markers)
        sim_layout.addWidget(btn_start_mark, 1, 5)

        btn_stop_mark = QPushButton("Stop Markers")
        btn_stop_mark.clicked.connect(self.stop_markers)
        sim_layout.addWidget(btn_stop_mark, 1, 6)

        sim_group.setLayout(sim_layout)
        layout.addWidget(sim_group)

        # Fabric Bus
        bus_group = QGroupBox("🚌 Fabric Bus + Dashboard")
        bus_layout = QHBoxLayout()
        bus_layout.setSpacing(10)

        bus_layout.addWidget(QLabel("Port:"))
        self.bus_port = QLineEdit("8008")
        self.bus_port.setMaximumWidth(60)
        bus_layout.addWidget(self.bus_port)

        btn_start_bus = QPushButton("Start Bus")
        btn_start_bus.clicked.connect(self.start_bus)
        bus_layout.addWidget(btn_start_bus)

        btn_stop_bus = QPushButton("Stop Bus")
        btn_stop_bus.clicked.connect(self.stop_bus)
        bus_layout.addWidget(btn_stop_bus)

        btn_dashboard = QPushButton("Open Dashboard")
        btn_dashboard.clicked.connect(self.open_dashboard)
        bus_layout.addWidget(btn_dashboard)

        bus_layout.addStretch()
        bus_group.setLayout(bus_layout)
        layout.addWidget(bus_group)

        # Host Demos
        host_group = QGroupBox("📡 Host Demo Publishers")
        host_layout = QHBoxLayout()
        host_layout.setSpacing(8)

        btn_start_rsvp = QPushButton("Start RSVP")
        btn_start_rsvp.clicked.connect(self.start_rsvp)
        host_layout.addWidget(btn_start_rsvp)

        btn_stop_rsvp = QPushButton("Stop RSVP")
        btn_stop_rsvp.clicked.connect(self.stop_rsvp)
        host_layout.addWidget(btn_stop_rsvp)

        btn_start_ssvep = QPushButton("Start SSVEP")
        btn_start_ssvep.clicked.connect(self.start_ssvep)
        host_layout.addWidget(btn_start_ssvep)

        btn_stop_ssvep = QPushButton("Stop SSVEP")
        btn_stop_ssvep.clicked.connect(self.stop_ssvep)
        host_layout.addWidget(btn_stop_ssvep)

        btn_start_errp = QPushButton("Start ErrP")
        btn_start_errp.clicked.connect(self.start_errp)
        host_layout.addWidget(btn_start_errp)

        btn_stop_errp = QPushButton("Stop ErrP")
        btn_stop_errp.clicked.connect(self.stop_errp)
        host_layout.addWidget(btn_stop_errp)

        host_group.setLayout(host_layout)
        layout.addWidget(host_group)

        # Logs
        log_group = QGroupBox("📋 System Logs")
        log_layout = QVBoxLayout()

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(300)
        log_layout.addWidget(self.log_text)

        log_group.setLayout(log_layout)
        layout.addWidget(log_group, stretch=1)

        panel.setLayout(layout)
        return panel

    def create_data_panel(self):
        """Create right data panel with live updates"""
        panel = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(15)

        # Title
        title = QLabel("🧠 Live Neural Data")
        title.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
        title.setStyleSheet("color: rgba(255, 255, 255, 255); background: transparent;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Connection status
        self.status_label = QLabel("🔴 Disconnected")
        self.status_label.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        self.status_label.setStyleSheet("""
            background: rgba(50, 50, 65, 200);
            border: 1px solid rgba(100, 100, 120, 100);
            border-radius: 8px;
            padding: 12px;
        """)
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)

        # Tabs for cards and graph
        tabs = QTabWidget()
        tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid rgba(100, 100, 120, 100);
                border-radius: 8px;
                background: rgba(30, 30, 40, 150);
            }
            QTabBar::tab {
                background: rgba(50, 50, 65, 200);
                color: white;
                padding: 10px 20px;
                border: 1px solid rgba(100, 100, 120, 100);
                border-bottom: none;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(99, 102, 241, 200),
                    stop:1 rgba(79, 82, 221, 200));
            }
        """)

        # Cards tab
        cards_widget = QWidget()
        cards_layout = QVBoxLayout()
        cards_layout.setSpacing(15)

        self.p300_card = DataCard("P300 RSVP", "🧠", "#06B6D4")
        cards_layout.addWidget(self.p300_card)

        self.ssvep_card = DataCard("SSVEP", "🌀", "#8B5CF6")
        cards_layout.addWidget(self.ssvep_card)

        self.errp_card = DataCard("Error Potential", "⚠️", "#F97316")
        cards_layout.addWidget(self.errp_card)

        cards_layout.addStretch()
        cards_widget.setLayout(cards_layout)
        tabs.addTab(cards_widget, "📊 Metrics")

        # Graph tab
        graph_widget = self.create_graph_panel()
        tabs.addTab(graph_widget, "📈 Live Plot")

        layout.addWidget(tabs)
        panel.setLayout(layout)
        return panel

    def create_graph_panel(self):
        """Create rolling window graph for all 3 signals"""
        widget = QWidget()
        layout = QVBoxLayout()

        # Setup pyqtgraph with dark theme
        pg.setConfigOptions(antialias=True)

        # Create plot widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground((20, 20, 28))
        self.plot_widget.setLabel('left', 'Signal Value')
        self.plot_widget.setLabel('bottom', 'Time (seconds)')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.addLegend()

        # Create plot curves
        self.p300_curve = self.plot_widget.plot(
            pen=pg.mkPen(color='#06B6D4', width=2),
            name='P300'
        )
        self.ssvep_curve = self.plot_widget.plot(
            pen=pg.mkPen(color='#8B5CF6', width=2),
            name='SSVEP'
        )
        self.errp_curve = self.plot_widget.plot(
            pen=pg.mkPen(color='#F97316', width=2),
            name='ErrP'
        )

        layout.addWidget(self.plot_widget)
        widget.setLayout(layout)
        return widget

    def _log(self, msg: str):
        self.log_signal.emit(msg)

    def append_log(self, msg: str):
        self.log_text.append(msg.rstrip())
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )

    def _start_proc(self, name, cmd):
        if name in self.procs and self.procs[name].p and self.procs[name].p.poll() is None:
            self._log(f"[{name}] already running\n")
            return
        proc = Proc(name, cmd)
        self.procs[name] = proc
        proc.start(self._log)

    def _stop_proc(self, name):
        if name in self.procs:
            self.procs[name].stop(self._log)

    def start_eeg(self):
        fs = self.eeg_fs.text().strip() or "512"
        ch = self.eeg_ch.text().strip() or "8"
        cmd = [self.python, os.path.join(ROOT, "scripts", "simulate_lsl_eeg.py"), "--fs", fs, "--channels", ch]
        self._start_proc("EEG", cmd)

    def stop_eeg(self):
        self._stop_proc("EEG")

    def start_markers(self):
        task = self.mark_task.currentText()
        rate = self.mark_rate.text().strip() or "8.0"
        cmd = [self.python, os.path.join(ROOT, "scripts", "simulate_markers.py"), "--task", task, "--rate_hz", rate]
        self._start_proc("Markers", cmd)

    def stop_markers(self):
        self._stop_proc("Markers")

    def start_bus(self):
        port = self.bus_port.text().strip() or "8008"
        cmd = [self.python, "-m", "uvicorn", "core.api.fabric_bus:app", "--host", "0.0.0.0", "--port", port, "--reload"]
        self._start_proc("Bus", cmd)

    def stop_bus(self):
        self._stop_proc("Bus")

    def open_dashboard(self):
        port = self.bus_port.text().strip() or "8008"
        webbrowser.open(f"http://localhost:{port}/")

    def start_rsvp(self):
        self._start_proc("RSVP", [self.python, os.path.join(ROOT, "scripts", "host_rsvp_demo.py")])

    def stop_rsvp(self):
        self._stop_proc("RSVP")

    def start_ssvep(self):
        self._start_proc("SSVEP", [self.python, os.path.join(ROOT, "scripts", "host_ssvep_demo.py")])

    def stop_ssvep(self):
        self._stop_proc("SSVEP")

    def start_errp(self):
        self._start_proc("ErrP", [self.python, os.path.join(ROOT, "scripts", "host_errp_demo.py")])

    def stop_errp(self):
        self._stop_proc("ErrP")

    def update_data(self):
        """Fetch latest data from API and update cards + graphs"""
        try:
            port = self.bus_port.text().strip() or "8008"
            response = requests.get(f"http://localhost:{port}/latest", timeout=0.5)

            if response.status_code == 200:
                data = response.json()
                self.status_label.setText("🟢 Connected")
                self.status_label.setStyleSheet("""
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                        stop:0 rgba(16, 185, 129, 100),
                        stop:1 rgba(5, 150, 105, 100));
                    border: 1px solid rgba(16, 185, 129, 200);
                    border-radius: 8px;
                    padding: 12px;
                    color: rgb(16, 185, 129);
                """)

                # Get current time
                current_time = (datetime.now() - self.start_time).total_seconds()
                self.time_data.append(current_time)

                # Update P300 (with QC)
                if "p300" in data:
                    p300 = data["p300"]
                    quality = p300.get('quality', 0)
                    self.p300_card.update_data(
                        f"{p300['value']:.3f}",
                        f"Confidence: {p300['value']*100:.0f}%",
                        quality=quality
                    )
                    self.p300_data.append(p300['value'])
                else:
                    self.p300_data.append(None)

                # Update SSVEP (with margin QC)
                if "ssvep" in data:
                    ssvep = data["ssvep"]
                    target_idx = ssvep.get("meta", {}).get("target_idx", "--")
                    target_freq = ssvep.get("meta", {}).get("target_freq", "--")
                    margin = ssvep.get("meta", {}).get("margin", 0)
                    freq_text = f"{target_freq} Hz" if target_freq != "--" else "--"

                    # Use margin as quality indicator (higher margin = better separation)
                    quality = ssvep.get('quality', 0)
                    margin_info = f"Margin: {margin:.3f}" if isinstance(margin, (int, float)) else ""

                    self.ssvep_card.update_data(
                        str(target_idx),
                        f"{freq_text} | {margin_info}",
                        quality=quality
                    )
                    self.ssvep_data.append(ssvep['value'])
                else:
                    self.ssvep_data.append(None)

                # Update ErrP (with QC)
                if "errp" in data:
                    errp = data["errp"]
                    is_error = errp.get("meta", {}).get("is_error_predicted", False)
                    pred = "Error" if is_error else "Correct"
                    quality = errp.get('quality', 0)

                    self.errp_card.update_data(
                        f"{errp['value']*100:.0f}%",
                        f"Prediction: {pred}",
                        quality=quality
                    )
                    self.errp_data.append(errp['value'])
                else:
                    self.errp_data.append(None)

                # Update graph
                self.update_graph()

            else:
                self.status_label.setText("🔴 Disconnected")
                self.status_label.setStyleSheet("""
                    background: rgba(50, 50, 65, 200);
                    border: 1px solid rgba(239, 68, 68, 200);
                    border-radius: 8px;
                    padding: 12px;
                    color: rgb(239, 68, 68);
                """)

        except requests.exceptions.RequestException:
            self.status_label.setText("🔴 Disconnected")
            self.status_label.setStyleSheet("""
                background: rgba(50, 50, 65, 200);
                border: 1px solid rgba(239, 68, 68, 200);
                border-radius: 8px;
                padding: 12px;
                color: rgb(239, 68, 68);
            """)

    def update_graph(self):
        """Update the rolling window graph"""
        if len(self.time_data) > 0:
            time_array = list(self.time_data)

            # Filter out None values for each signal
            p300_array = [v if v is not None else float('nan') for v in self.p300_data]
            ssvep_array = [v if v is not None else float('nan') for v in self.ssvep_data]
            errp_array = [v if v is not None else float('nan') for v in self.errp_data]

            # Update curves
            self.p300_curve.setData(time_array, p300_array)
            self.ssvep_curve.setData(time_array, ssvep_array)
            self.errp_curve.setData(time_array, errp_array)

    def closeEvent(self, event):
        """Clean up on close"""
        self.timer.stop()
        for proc in self.procs.values():
            if proc:
                try:
                    proc.stop(self._log)
                except Exception:
                    pass
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # Modern style

    window = LauncherWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
