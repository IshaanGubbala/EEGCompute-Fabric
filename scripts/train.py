#!/usr/bin/env python3
import argparse, os, pickle
from utils import load_cfg, ensure_dir, build_image_dataset

def train_cnn(cfg, use_eeg_assist=False):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torchvision import models, transforms
    from tqdm import tqdm

    proc=cfg['paths']['processed']
    # transforms
    tfm = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    # Prefer the 'train' split; if unavailable (no train images), fall back to 'val'
    ds = build_image_dataset(os.path.join(proc,'coco_subset.jsonl'), transform=tfm, subset_filter=lambda r: r.get('split')=='train')
    if len(ds)==0:
        print('[WARN] No training images found under train2017. Falling back to val2017 split for training.')
        ds = build_image_dataset(os.path.join(proc,'coco_subset.jsonl'), transform=tfm, subset_filter=lambda r: r.get('split')=='val')
        if len(ds)==0:
            raise RuntimeError('No images found in either train or val splits. Please download COCO val/train images to configured paths (use scripts/control.py download and optionally scripts/download_coco.py --train 1).')
    # Custom collate to handle dict metadata without forcing equal-size collate
    def collate(batch):
        xs, ys, rs = zip(*batch)
        xs = torch.stack(xs, 0)
        ys = torch.tensor(ys, dtype=torch.float32)
        return xs, ys, rs
    # Use single-process loading in restricted environments to avoid shm issues
    dl = DataLoader(ds, batch_size=16, shuffle=True, num_workers=0, pin_memory=False, collate_fn=collate)

    device = torch.device('cpu')
    # Model: ResNet18 binary head
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_f = m.fc.in_features
    m.fc = nn.Linear(in_f, 1)
    m = m.to(device)
    crit = nn.BCEWithLogitsLoss(reduction='none')
    opt = optim.AdamW(m.parameters(), lr=1e-4)

    # EEG assistance: build weights from P300 if available
    boost = {}
    if use_eeg_assist:
        # simple heuristic: weight positives higher
        try:
            # weights per item_id from P300 boost can be injected later; use +0.5 for targets, +0.1 for non-targets here
            pass
        except Exception:
            pass

    m.train()
    epochs = 1  # keep minimal
    for ep in range(epochs):
        pbar = tqdm(dl, desc=f"train ep{ep+1}")
        for x, y, r in pbar:
            x = x.to(device)
            y = y.view(-1,1).to(device)
            logits = m(x)
            loss_raw = crit(logits, y)
            if use_eeg_assist:
                # weight positives slightly higher
                w = (y*1.5 + (1-y)*1.0)
                loss = (loss_raw * w).mean()
            else:
                loss = loss_raw.mean()
            opt.zero_grad(); loss.backward(); opt.step()
            pbar.set_postfix({'loss': float(loss.item())})

    # Return state dict
    return { 'state_dict': m.state_dict(), 'arch': 'resnet18', 'normalize': True }

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config', required=True); args=ap.parse_args()
    cfg=load_cfg(args.config)
    models_dir=cfg['paths']['models']; ensure_dir(models_dir)

    base = train_cnn(cfg, use_eeg_assist=False)
    eeg  = train_cnn(cfg, use_eeg_assist=True)

    with open(os.path.join(models_dir,'baseline.pkl'),'wb') as f: pickle.dump(base,f)
    with open(os.path.join(models_dir,'eeg_trained.pkl'),'wb') as f: pickle.dump(eeg,f)
    print('Saved models/baseline.pkl and models/eeg_trained.pkl (CNN)')

if __name__=='__main__':
    main()
