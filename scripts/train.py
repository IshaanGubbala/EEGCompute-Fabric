#!/usr/bin/env python3
import argparse, os, pickle
from utils import load_cfg, ensure_dir, build_image_dataset
from utils import read_jsonl, simulate_p300_boosts, normalize_scores

def train_cnn(cfg, use_eeg_assist=False, log_writer=None, label='baseline'):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torchvision import models, transforms
    from tqdm import tqdm

    proc=cfg['paths']['processed']
    # transforms (separate train/val to allow augmentation)
    tr_cfg = cfg.get('training',{})
    # device selection
    device = torch.device('cpu')
    try:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
    except Exception:
        device = torch.device('cpu')
    try:
        torch.set_float32_matmul_precision('medium')
    except Exception:
        pass
    aug = bool(tr_cfg.get('augment', True))
    img_size = int(tr_cfg.get('image_size', 224))
    if aug:
        tfm_train = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])
    else:
        tfm_train = transforms.Compose([
            transforms.Resize((img_size,img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])
    tfm_val = transforms.Compose([
        transforms.Resize((img_size,img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    # Prefer the 'train' split; if unavailable (no train images), fall back to 'val'
    ds = build_image_dataset(os.path.join(proc,'coco_subset.jsonl'), transform=tfm_train, subset_filter=lambda r: r.get('split')=='train')
    if len(ds)==0:
        print('[WARN] No training images found under train2017. Falling back to val2017 split for training.')
        ds = build_image_dataset(os.path.join(proc,'coco_subset.jsonl'), transform=tfm_val, subset_filter=lambda r: r.get('split')=='val')
        if len(ds)==0:
            raise RuntimeError('No images found in either train or val splits. Please download COCO val/train images to configured paths (use scripts/control.py download and optionally scripts/download_coco.py --train 1).')
    # Custom collate to handle dict metadata without forcing equal-size collate
    def collate(batch):
        xs, ys, rs = zip(*batch)
        xs = torch.stack(xs, 0)
        ys = torch.tensor(ys, dtype=torch.float32)
        return xs, ys, rs
    # DataLoader tuning
    batch_size = int(tr_cfg.get('batch_size', 16))
    # Disable multiprocessing on MPS to avoid pickling issues
    if device.type == 'mps':
        num_workers = 0
    else:
        num_workers = int(tr_cfg.get('num_workers', max(1, (os.cpu_count() or 2) - 1)))
    pin_memory = (device.type == 'cuda')
    persistent = num_workers > 0
    try:
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate, persistent_workers=persistent, prefetch_factor=2)
    except Exception:
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False, collate_fn=collate)
    # Validation loader (val split if available)
    dsv = build_image_dataset(os.path.join(proc,'coco_subset.jsonl'), transform=tfm_val, subset_filter=lambda r: r.get('split')=='val')
    try:
        dl_val = DataLoader(dsv, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate, persistent_workers=persistent, prefetch_factor=2) if len(dsv)>0 else None
    except Exception:
        dl_val = DataLoader(dsv, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False, collate_fn=collate) if len(dsv)>0 else None

    # Model: selectable backbone with binary head
    backbone = str(tr_cfg.get('backbone','resnet18'))
    if backbone == 'mobilenet_v3_small':
        from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
        m = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        # MobileNetV3 classifier is a Sequential; get input features from first Linear layer
        in_f = m.classifier[0].in_features
        dropout_p = float(tr_cfg.get('dropout', 0.2))
        # Replace entire classifier with simpler version
        m.classifier = nn.Sequential(nn.Dropout(p=dropout_p), nn.Linear(in_f, 1))
        head_prefix = 'classifier'
    else:
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_f = m.fc.in_features
        dropout_p = float(tr_cfg.get('dropout', 0.2))
        m.fc = nn.Sequential(nn.Dropout(p=dropout_p), nn.Linear(in_f, 1))
        head_prefix = 'fc'
    m = m.to(device)
    # Only use channels_last for CUDA (not MPS)
    if device.type == 'cuda':
        m = m.to(memory_format=torch.channels_last)
        try:
            import torch.backends.cudnn as cudnn
            if hasattr(cudnn, 'benchmark'):
                cudnn.benchmark = True
        except Exception:
            pass
    # class imbalance: pos_weight auto if requested
    pos_weight_auto = bool(tr_cfg.get('pos_weight_auto', True))
    if pos_weight_auto:
        pos = sum(1 for it in ds.items if float(it.get('is_target',0))==1.0)
        neg = max(1, len(ds.items)-pos)
        pw = torch.tensor([neg/float(max(1,pos))], dtype=torch.float32, device=device)
    else:
        pw = None
    crit = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pw)
    lr = float(tr_cfg.get('lr', 1e-4))
    wd = float(tr_cfg.get('weight_decay', 5e-4))
    def get_params(train_backbone=True):
        if train_backbone:
            return m.parameters()
        else:
            return getattr(m, head_prefix).parameters()
    opt = optim.AdamW(get_params(train_backbone=True), lr=lr, weight_decay=wd)
    # LR scheduler options
    sched_type = tr_cfg.get('lr_schedule','plateau')
    if sched_type=='cosine':
        from torch.optim.lr_scheduler import CosineAnnealingLR
        sched = CosineAnnealingLR(opt, T_max=int(tr_cfg.get('epochs',5)))
    else:
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        sched = ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=1, threshold=1e-3)

    # EEG assistance: build weights from P300-like evidence
    boost = {}
    if use_eeg_assist:
        try:
            import json
            # Load RSVP
            with open(os.path.join(proc,'rsvp_streams.jsonl'),'r') as f:
                rsvp = json.loads(next(l for l in f if l.strip()))
            # Load real scores if available
            boosts = {}
            scores_path = os.path.join(proc,'scores_p300.jsonl')
            if os.path.exists(scores_path) and os.path.getsize(scores_path) > 0:
                for rec in read_jsonl(scores_path):
                    iid = rec.get('item_id'); s = float(rec.get('score',0.0))
                    boosts[iid] = s
                boosts = normalize_scores(boosts, method='z_tanh')
            # If missing, simulate using shared utility over the RSVP and current subset
            if not boosts:
                # use only items present in current dataset
                boosts = simulate_p300_boosts(cfg, rsvp, ds.items)
            boost = boosts
        except Exception:
            boost = {}

    m.train()
    epochs = int(tr_cfg.get('epochs', 5))
    import copy
    best_val = float('inf')
    best_state = None
    # freeze backbone for first N epochs if configured
    freeze_epochs = int(tr_cfg.get('freeze_backbone_epochs', 0))
    if freeze_epochs > 0:
        for name, p in m.named_parameters():
            if not name.startswith(head_prefix):
                p.requires_grad = False
        opt = optim.AdamW(get_params(train_backbone=False), lr=lr, weight_decay=wd)
    patience = int(tr_cfg.get('early_stop_patience', 2))
    bad_epochs = 0
    label_smooth = float(tr_cfg.get('label_smoothing', 0.05))
    # mixup/cutmix settings
    mixup_alpha = float(tr_cfg.get('mixup_alpha', 0.2))
    cutmix_alpha = float(tr_cfg.get('cutmix_alpha', 0.0))
    import random
    def _rand_bbox(W, H, lam):
        import math
        cut_rat = math.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        cx = random.randint(0, W)
        cy = random.randint(0, H)
        x1 = max(cx - cut_w // 2, 0)
        y1 = max(cy - cut_h // 2, 0)
        x2 = min(cx + cut_w // 2, W)
        y2 = min(cy + cut_h // 2, H)
        return x1, y1, x2, y2
    early_metric = tr_cfg.get('early_stop_metric','val_loss')  # or 'auroc'
    best_metric = -1e9 if early_metric=='auroc' else float('inf')
    use_amp = (device.type == 'cuda')
    scaler = None
    if use_amp:
        scaler = torch.cuda.amp.GradScaler()
    for ep in range(epochs):
        running = 0.0; n_ex=0
        pbar = tqdm(dl, desc=f"train ep{ep+1}")
        for x, y, r in pbar:
            x = x.to(device)
            y = y.view(-1,1).to(device)
            # label smoothing toward 0.5
            if label_smooth > 0:
                y = y*(1.0 - label_smooth) + 0.5*label_smooth
            # mixup / cutmix
            if mixup_alpha>0 or cutmix_alpha>0:
                perm = torch.randperm(x.size(0), device=x.device)
                y2 = y[perm]
                if cutmix_alpha>0 and random.random()<0.5:
                    lam = torch.distributions.Beta(cutmix_alpha, cutmix_alpha).sample().item()
                    x1,y1,x2,y2 = _rand_bbox(x.size(3), x.size(2), lam)
                    x[:,:,y1:y2,x1:x2] = x[perm,:,y1:y2,x1:x2]
                    # adjust lam by area
                    lam = 1 - ((x2-x1)*(y2-y1)/(x.size(2)*x.size(3)))
                    y = lam*y + (1-lam)*y2
                else:
                    lam = torch.distributions.Beta(mixup_alpha, mixup_alpha).sample().item()
                    x = lam*x + (1-lam)*x[perm]
                    y = lam*y + (1-lam)*y2
            # Only use channels_last for CUDA
            if device.type == 'cuda':
                x = x.to(memory_format=torch.channels_last)
            if use_amp:
                with torch.cuda.amp.autocast():
                    logits = m(x)
                    loss_raw = crit(logits, y)
            else:
                logits = m(x)
                loss_raw = crit(logits, y)
            if use_eeg_assist:
                # Map metadata item_ids to per-sample boosts
                # r is a list[dict]; build weights that nudge logits via loss weighting
                import torch
                item_boosts = []
                for rec in r:
                    b = float(boost.get(rec.get('item_id'), 0.0))
                    item_boosts.append(b)
                w = torch.tensor(item_boosts, dtype=loss_raw.dtype, device=loss_raw.device).view(-1,1)
                # Convert signed boost into weight multiplier around 1.0
                # Positive boosts emphasize positives slightly more; negative the opposite
                # Scale boosts: 0.7 chosen to be impactful but stable
                w = 1.0 + 0.7*w
                loss = (loss_raw * w).mean()
            else:
                loss = loss_raw.mean()
            opt.zero_grad()
            if use_amp:
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            else:
                loss.backward(); opt.step()
            running += float(loss.item())*x.size(0); n_ex += x.size(0)
            pbar.set_postfix({'loss': float(loss.item())})

        # epoch averages
        tr_loss = running/max(1,n_ex)
        if log_writer:
            log_writer('train', ep+1, label, tr_loss)

        # validation pass (and optional metric computation)
        if dl_val is not None:
            m.eval(); v_running=0.0; v_ex=0
            import torch
            val_logits=[]; val_targets=[]
            with torch.no_grad():
                for x,y,r in dl_val:
                    x=x.to(device); y=y.view(-1,1).to(device)
                    logits=m(x)
                    vloss=crit(logits,y).mean()
                    v_running += float(vloss.item())*x.size(0); v_ex += x.size(0)
                    val_logits += [float(t) for t in logits.view(-1)]
                    val_targets += [int(t) for t in y.view(-1)]
            val_loss = v_running/max(1,v_ex)
            if log_writer:
                log_writer('val', ep+1, label, val_loss)
            if val_loss < best_val - 1e-6:
                best_val = val_loss
                best_state = copy.deepcopy(m.state_dict())
                bad_epochs = 0
            else:
                bad_epochs += 1
            # step scheduler
            if sched_type=='cosine':
                sched.step()
            else:
                sched.step(val_loss)
            m.train()
            # early stopping
            # optional early stop on AUROC
            if early_metric=='auroc' and val_logits:
                # compute auroc quickly
                import math
                pairs = sorted([(1.0/(1.0+math.exp(-z)), t) for z,t in zip(val_logits,val_targets)], key=lambda x: x[0], reverse=True)
                P = sum(t for _,t in pairs); N = len(pairs)-P
                tp=0; fp=0; roc=[]
                for s,t in pairs:
                    if t==1: tp+=1
                    else: fp+=1
                    roc.append((fp/float(max(1,N)), tp/float(max(1,P))))
                roc = sorted(roc); auroc=0.0
                for i in range(1,len(roc)):
                    x0,y0=roc[i-1]; x1,y1=roc[i]; auroc += (x1-x0)*(y0+y1)/2.0
                # track best auroc
                if auroc > best_metric + 1e-6:
                    best_metric = auroc; bad_epochs = 0; best_state = copy.deepcopy(m.state_dict())
                else:
                    bad_epochs += 1
            if bad_epochs >= patience:
                break
        # unfreeze after freeze_epochs
        if freeze_epochs>0 and ep+1==freeze_epochs:
            for p in m.parameters(): p.requires_grad=True
            opt = optim.AdamW(get_params(train_backbone=True), lr=lr, weight_decay=wd)

    # Return best or final state dict
    sd = best_state if best_state is not None else m.state_dict()
    return { 'state_dict': sd, 'arch': 'resnet18', 'normalize': True }

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config', required=True); args=ap.parse_args()
    cfg=load_cfg(args.config)
    models_dir=cfg['paths']['models']; ensure_dir(models_dir)
    # logging
    logs_dir=os.path.join(cfg['paths'].get('reports','output/reports'),'..','logs')
    logs_dir=os.path.normpath(logs_dir)
    ensure_dir(logs_dir)
    log_path=os.path.join(logs_dir,'train_loss.csv')
    # initialize CSV header if not exists
    if not os.path.exists(log_path):
        with open(log_path,'w') as f:
            f.write('epoch,split,model,loss\n')

    def write_log(split, epoch, model, loss):
        with open(log_path,'a') as f:
            f.write(f"{epoch},{split},{model},{loss}\n")

    base = train_cnn(cfg, use_eeg_assist=False, log_writer=write_log, label='baseline')
    eeg  = train_cnn(cfg, use_eeg_assist=True, log_writer=write_log, label='eeg')

    with open(os.path.join(models_dir,'baseline.pkl'),'wb') as f: pickle.dump(base,f)
    with open(os.path.join(models_dir,'eeg_trained.pkl'),'wb') as f: pickle.dump(eeg,f)
    print('Saved models/baseline.pkl and models/eeg_trained.pkl (CNN)')

if __name__=='__main__':
    main()
