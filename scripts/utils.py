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

# ---- EEG Simulation Utilities ----
def simulate_p300_boosts(cfg, rsvp_stream, subset_items, seed: int = 1337):
    """
    Produce per-item signed boosts based on RSVP events.
    Positive for targets, negative for non-targets, with stochasticity.
    Returns dict: item_id -> float boost in roughly [-1, 1].
    """
    import random, math
    def sigmoid(x):
        return 1/(1+math.exp(-x))
    sim = cfg.get('sim', {})
    a_t, b_t = sim.get('p300', {}).get('beta_target', [6,5])
    a_n, b_n = sim.get('p300', {}).get('beta_nontarget', [5,6])
    drop = sim.get('p300', {}).get('drop_rate', 0.2)
    rng = random.Random(seed)
    idx = {it['item_id']: it for it in subset_items}
    evidence = {}
    for ev in rsvp_stream['sequence']:
        iid = ev['item_id']
        it = idx.get(iid)
        if it is None:
            continue
        if rng.random() < drop:
            continue
        # draw probability then convert to logit evidence
        p = rng.betavariate(a_t, b_t) if it.get('is_target',0)==1 else rng.betavariate(a_n, b_n)
        p = max(min(p, 1-1e-6), 1e-6)
        logit = math.log(p/(1-p))
        evidence[iid] = evidence.get(iid, 0.0) + logit
    boosts = {}
    for iid, v in evidence.items():
        b = sigmoid(v) - 0.5
        if idx.get(iid, {}).get('is_target',0) == 1:
            boosts[iid] = abs(b)
        else:
            boosts[iid] = -abs(b)
    return boosts

def normalize_scores(scores, method: str = 'z_tanh'):
    """
    Normalize a dict[item_id->score] to roughly [-1,1].
    Supported: 'z_tanh' (z-score then tanh), 'minmax' (-1..1), 'standard' (z-score only).
    """
    import math
    if not scores:
        return {}
    vals = list(scores.values())
    n = len(vals)
    mean = sum(vals)/n
    var = sum((v-mean)**2 for v in vals)/max(1,n-1)
    std = math.sqrt(var) if var>0 else 1.0
    if method == 'z_tanh':
        return {k: math.tanh((v-mean)/(std+1e-6)) for k,v in scores.items()}
    elif method == 'minmax':
        vmin, vmax = min(vals), max(vals)
        if vmax==vmin:
            return {k: 0.0 for k in scores}
        return {k: (2*((v-vmin)/(vmax-vmin))-1.0) for k,v in scores.items()}
    elif method == 'standard':
        return {k: (v-mean)/(std+1e-6) for k,v in scores.items()}
    else:
        return scores
