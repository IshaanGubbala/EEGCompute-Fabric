import json, os, time, random
from pathlib import Path

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def write_jsonl(path, records):
    ensure_dir(os.path.dirname(path))
    with open(path, 'w') as f:
        for r in records:
            f.write(json.dumps(r)+"\n")

def read_jsonl(path):
    with open(path, 'r') as f:
        for line in f:
            line=line.strip()
            if not line: continue
            yield json.loads(line)

def load_cfg(path):
    import json as _json
    with open(path,'r') as f:
        return _json.load(f)

def now_ts():
    return time.time()

# ---- Image dataset helpers (PyTorch) ----
try:
    from PIL import Image
    from torch.utils.data import Dataset
except Exception:
    Image = None
    Dataset = object


class JsonlImageDataset(Dataset):
    """Top-level dataset class so it is picklable by DataLoader workers."""
    def __init__(self, path, transform=None, target_key='is_target', subset_filter=None):
        if Image is None or Dataset is object:
            raise RuntimeError("PyTorch/Pillow required. pip install torch torchvision Pillow")
        self.items = []
        for r in read_jsonl(path):
            if subset_filter and not subset_filter(r):
                continue
            if not os.path.exists(r['filepath']):
                continue
            self.items.append(r)
        self.transform = transform
        self.target_key = target_key

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        r = self.items[idx]
        img = Image.open(r['filepath']).convert('RGB')
        if self.transform:
            img = self.transform(img)
        y = float(r.get(self.target_key, 0))
        return img, y, r


def build_image_dataset(jsonl_path, transform=None, target_key='is_target', subset_filter=None):
    return JsonlImageDataset(jsonl_path, transform, target_key, subset_filter)
