from pathlib import Path
import json
from pybv import write_brainvision
def write_bids_session(root: str, subj: str, run: int, fs: int, data, ch_names):
    root = Path(root); ses = root / f"sub-{subj}" / "ses-01" / "eeg"
    ses.mkdir(parents=True, exist_ok=True)
    base = f"sub-{subj}_ses-01_task-task_run-{run:02d}_eeg"
    vhdr = ses / f"{base}.vhdr"
    write_brainvision(data, fs, ch_names, vhdr, events=None, resolution=1e-6, unit="µV")
    (ses / f"{base}.json").write_text(json.dumps({"TaskName":"task"}))
    return str(vhdr)
