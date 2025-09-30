"""
EEG-BIDS Export Module

Writes EEG data, events, and metadata to Brain Imaging Data Structure (BIDS) format.
Supports continuous recording of LSL streams + event markers + score vectors.

BIDS structure:
    sub-{subject}/
        ses-{session}/
            eeg/
                sub-{subject}_ses-{session}_task-{task}_eeg.vhdr/.vmrk/.eeg (BrainVision)
                sub-{subject}_ses-{session}_task-{task}_eeg.json (sidecar)
                sub-{subject}_ses-{session}_task-{task}_events.tsv
                sub-{subject}_ses-{session}_task-{task}_scores.json

References:
    - BIDS spec: https://bids-specification.readthedocs.io/
    - MNE-BIDS: https://mne.tools/mne-bids/
"""
from __future__ import annotations
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
from dataclasses import asdict


class BIDSWriter:
    """Write EEG data + events to BIDS format"""

    def __init__(
        self,
        output_dir: str = "data/bids",
        subject: str = "001",
        session: Optional[str] = None,
        task: str = "bci"
    ):
        """
        Initialize BIDS writer.

        Args:
            output_dir: Root BIDS directory
            subject: Subject ID (e.g., "001")
            session: Session ID (default: timestamp)
            task: Task name (e.g., "bci", "rsvp", "ssvep")
        """
        self.output_dir = Path(output_dir)
        self.subject = subject
        self.session = session or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.task = task

        # Create BIDS directory structure
        self.subject_dir = self.output_dir / f"sub-{self.subject}"
        self.session_dir = self.subject_dir / f"ses-{self.session}"
        self.eeg_dir = self.session_dir / "eeg"
        self.eeg_dir.mkdir(parents=True, exist_ok=True)

        # File naming
        self.base_name = f"sub-{self.subject}_ses-{self.session}_task-{self.task}"

        # Internal buffers
        self.eeg_data: List[np.ndarray] = []
        self.events: List[Dict[str, Any]] = []
        self.scores: List[Dict[str, Any]] = []

        print(f"[BIDSWriter] Initialized")
        print(f"  Output: {self.eeg_dir}")
        print(f"  Subject: {self.subject}, Session: {self.session}, Task: {self.task}")

    def add_eeg_chunk(self, data: np.ndarray, timestamps: Optional[np.ndarray] = None):
        """
        Add EEG data chunk.

        Args:
            data: (n_channels, n_samples) array
            timestamps: Optional timestamps for each sample
        """
        self.eeg_data.append(data)

    def add_event(self, onset: float, duration: float, event_type: str, value: Optional[Any] = None):
        """
        Add an event marker.

        Args:
            onset: Event onset time (seconds)
            duration: Event duration (seconds)
            event_type: Event type/description
            value: Optional event value
        """
        event = {
            "onset": float(onset),
            "duration": float(duration),
            "event_type": event_type,
            "value": value
        }
        self.events.append(event)

    def add_score(self, score_vector):
        """
        Add a ScoreVector to the session.

        Args:
            score_vector: ScoreVector dataclass instance
        """
        score_dict = asdict(score_vector) if hasattr(score_vector, '__dataclass_fields__') else score_vector
        self.scores.append(score_dict)

    def write(self, eeg_data: Optional[np.ndarray] = None, fs: float = 512.0, ch_names: Optional[List[str]] = None):
        """
        Write all data to BIDS format.

        Args:
            eeg_data: (n_channels, n_samples) array (optional, uses buffered data if None)
            fs: Sampling frequency (Hz)
            ch_names: Channel names (e.g., ["Fz", "Cz", "Pz", ...])
        """
        # Use provided data or concatenate buffered chunks
        if eeg_data is None:
            if not self.eeg_data:
                print("[BIDSWriter] Warning: No EEG data to write")
                return
            eeg_data = np.concatenate(self.eeg_data, axis=1)

        n_channels, n_samples = eeg_data.shape
        if ch_names is None:
            ch_names = [f"Ch{i+1}" for i in range(n_channels)]

        # 1. Write EEG data (simple numpy format for now, BIDS-compatible via MNE)
        eeg_file = self.eeg_dir / f"{self.base_name}_eeg.npy"
        np.save(eeg_file, eeg_data)
        print(f"[BIDSWriter] Wrote EEG data: {eeg_file} ({n_channels} ch, {n_samples} samples)")

        # 2. Write EEG sidecar JSON
        sidecar = {
            "TaskName": self.task,
            "SamplingFrequency": float(fs),
            "PowerLineFrequency": 60,  # Hz (US standard)
            "EEGChannelCount": n_channels,
            "EEGReference": "average",
            "RecordingDuration": float(n_samples / fs),
            "RecordingType": "continuous",
            "Manufacturer": "LSL",
            "ManufacturersModelName": "Simulated",
            "EEGPlacementScheme": "Custom",
            "ChannelNames": ch_names
        }
        sidecar_file = self.eeg_dir / f"{self.base_name}_eeg.json"
        with open(sidecar_file, 'w') as f:
            json.dump(sidecar, f, indent=2)
        print(f"[BIDSWriter] Wrote sidecar: {sidecar_file}")

        # 3. Write events TSV
        if self.events:
            events_file = self.eeg_dir / f"{self.base_name}_events.tsv"
            with open(events_file, 'w') as f:
                # TSV header
                f.write("onset\tduration\tevent_type\tvalue\n")
                for evt in self.events:
                    f.write(f"{evt['onset']:.6f}\t{evt['duration']:.3f}\t{evt['event_type']}\t{evt.get('value', 'n/a')}\n")
            print(f"[BIDSWriter] Wrote events: {events_file} ({len(self.events)} events)")

        # 4. Write scores JSON
        if self.scores:
            scores_file = self.eeg_dir / f"{self.base_name}_scores.json"
            with open(scores_file, 'w') as f:
                json.dump(self.scores, f, indent=2)
            print(f"[BIDSWriter] Wrote scores: {scores_file} ({len(self.scores)} scores)")

        # 5. Write dataset_description.json (BIDS root)
        dataset_desc_file = self.output_dir / "dataset_description.json"
        if not dataset_desc_file.exists():
            dataset_desc = {
                "Name": "EEGCompute-Fabric BCI Dataset",
                "BIDSVersion": "1.6.0",
                "DatasetType": "raw",
                "Authors": ["EEGCompute-Fabric"],
                "License": "MIT"
            }
            with open(dataset_desc_file, 'w') as f:
                json.dump(dataset_desc, f, indent=2)
            print(f"[BIDSWriter] Created dataset description: {dataset_desc_file}")

        print(f"[BIDSWriter] ✓ BIDS export complete: {self.eeg_dir}")

    def get_session_path(self) -> Path:
        """Return path to session directory"""
        return self.eeg_dir

    def close(self):
        """Finalize and write all buffered data"""
        if self.eeg_data or self.events or self.scores:
            self.write()
        print(f"[BIDSWriter] Closed session: {self.session}")


# Convenience function
def export_session_to_bids(
    eeg_data: np.ndarray,
    events: List[Dict[str, Any]],
    scores: List[Dict[str, Any]],
    fs: float = 512.0,
    subject: str = "001",
    task: str = "bci",
    output_dir: str = "data/bids"
) -> Path:
    """
    One-shot BIDS export.

    Args:
        eeg_data: (n_channels, n_samples) array
        events: List of event dicts with 'onset', 'duration', 'event_type'
        scores: List of ScoreVector dicts
        fs: Sampling frequency
        subject: Subject ID
        task: Task name
        output_dir: Output directory

    Returns:
        Path to session directory
    """
    writer = BIDSWriter(output_dir=output_dir, subject=subject, task=task)

    # Add data
    for evt in events:
        writer.add_event(evt['onset'], evt['duration'], evt['event_type'], evt.get('value'))

    for score in scores:
        writer.add_score(score)

    # Write
    writer.write(eeg_data=eeg_data, fs=fs)

    return writer.get_session_path()
