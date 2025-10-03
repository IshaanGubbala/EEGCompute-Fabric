#!/usr/bin/env python3
import argparse, json, os, time, math, random
from pathlib import Path
import threading

from utils import load_cfg, ensure_dir, read_jsonl, write_jsonl, build_image_dataset
from compute_p300 import brainflow_scores, synth_scores, brainflow_scores_ssvep, brainflow_scores_errp


def run_prepare(cfg_path: str, full: bool):
    import subprocess
    cmd = ["python","scripts/prepare.py","--config",cfg_path]
    if full:
        cmd += ["--full","1"]
    print('>',' '.join(cmd)); subprocess.check_call(cmd)


def run_calibrate():
    import subprocess
    print('> python scripts/calibrate.py'); subprocess.check_call(["python","scripts/calibrate.py"])


def compute_eeg_scores(cfg_path: str, mode: str = 'brainflow', key: str = 'dog'):
    cfg = load_cfg(cfg_path)
    proc = cfg['paths']['processed']
    # Load RSVP + subset
    import json as _json
    with open(os.path.join(proc,'rsvp_streams.jsonl'),'r') as f:
        rsvp = _json.loads(next(l for l in f if l.strip()))
    items = list(read_jsonl(os.path.join(proc,'coco_subset.jsonl')))
    val_items = [it for it in items if it.get('split')=='val']
    # EEG streaming duration from config (faster default)
    duration = float(cfg.get('eeg',{}).get('streaming_duration_s', 5))
    ns = argparse.Namespace(board_id='synthetic', serial=None, duration=duration, fs=None, channels='0,1,2,3', key=key, match='classes')
    # P300
    try:
        p300 = brainflow_scores(ns, cfg, rsvp, val_items)
    except Exception:
        p300 = synth_scores(cfg, rsvp, val_items, key=key, match_mode='classes')
    write_jsonl(os.path.join(proc,'scores_p300.jsonl'), [{"item_id":k,"score":v} for k,v in p300.items()])
    # SSVEP
    try:
        ssvep = brainflow_scores_ssvep(ns, cfg, rsvp, val_items)
    except Exception:
        rng=random.Random(2025); ssvep={ev['item_id']:0.1+0.1*rng.random() for ev in rsvp['sequence']}
    write_jsonl(os.path.join(proc,'scores_ssvep.jsonl'), [{"item_id":k,"score":v} for k,v in ssvep.items()])
    # ErrP
    try:
        errp = brainflow_scores_errp(ns, cfg, rsvp, val_items)
    except Exception:
        rng=random.Random(2025); errp={ev['item_id']:-0.1-0.05*rng.random() for ev in rsvp['sequence']}
    write_jsonl(os.path.join(proc,'scores_errp.jsonl'), [{"item_id":k,"score":v} for k,v in errp.items()])
    print('Computed EEG scores (BrainFlow or synth fallback): P300, SSVEP, ErrP')


def train_model(cfg, use_eeg_assist=False, label='baseline', log_writer=None):
    import time
    import importlib
    # Import train module from the current scripts directory
    train_mod = importlib.import_module('train')
    # Energy estimator using psutil if available
    try:
        import psutil
    except Exception:
        psutil=None
    samples=[]; stop_flag={'stop':False}
    def sampler():
        if not psutil: return
        while not stop_flag['stop']:
            try:
                samples.append(psutil.cpu_percent(interval=0.5))
            except Exception:
                break
    t0=time.time()
    th = threading.Thread(target=sampler, daemon=True); th.start()
    out = train_mod.train_cnn(cfg, use_eeg_assist=use_eeg_assist, log_writer=log_writer, label=label)
    stop_flag['stop']=True; th.join(timeout=1.0)
    dt=time.time()-t0
    # Compute energy estimate
    base_power = 15.0
    if samples:
        avg_util = sum(samples)/len(samples)
        power = base_power * (avg_util/100.0)
    else:
        power = base_power
    energy = power * dt
    out['energy_J_est'] = energy
    return out, dt


def _predict_collate(batch):
    import torch
    xs, ys, rs = zip(*batch)
    return torch.stack(xs,0), torch.tensor(ys, dtype=torch.float32), rs

def predict_logits(cfg, model_dict, split='val'):
    import torch
    from torchvision import transforms, models
    proc = cfg['paths']['processed']
    img_size = int(cfg.get('training', {}).get('image_size', 224))
    tfm = transforms.Compose([
        transforms.Resize((img_size,img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    ds = build_image_dataset(os.path.join(proc,'coco_subset.jsonl'), transform=tfm, subset_filter=lambda r: r.get('split')==split)
    from torch.utils.data import DataLoader
    # Use faster DataLoader for inference
    dl = DataLoader(ds, batch_size=256, shuffle=False, num_workers=8, collate_fn=_predict_collate, pin_memory=False, prefetch_factor=4)
    # build model - must match training structure (Sequential with Dropout + Linear)
    import torch.nn as nn
    # Check which backbone was used
    backbone = cfg.get('training', {}).get('backbone', 'resnet18')
    if backbone == 'mobilenet_v3_small':
        from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
        m = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        in_f = m.classifier[0].in_features
        m.classifier = nn.Sequential(nn.Dropout(p=0.0), nn.Linear(in_f, 1))  # No dropout at inference
    else:
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_f = m.fc.in_features
        m.fc = nn.Sequential(nn.Dropout(p=0.0), nn.Linear(in_f, 1))  # No dropout at inference
    m.load_state_dict(model_dict['state_dict'])
    m.eval()

    # Try to use MPS for inference if available
    device = torch.device('cpu')
    try:
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
    except Exception:
        pass
    m = m.to(device)

    logits=[]; gts=[]; meta=[]
    with torch.no_grad(), torch.inference_mode():
        for x,y,r in dl:
            x = x.to(device, non_blocking=True)
            z = m(x)
            logits += z.view(-1).tolist()
            gts += y.view(-1).int().tolist()
            meta += r
    return logits, gts, meta


def sigmoid(x):
    try:
        return 1.0/(1.0+math.exp(-x))
    except OverflowError:
        return 0.0 if x<0 else 1.0


def accuracy_from_logits(logits, gts, threshold=0.5):
    preds=[1 if sigmoid(z)>=threshold else 0 for z in logits]
    correct=sum(1 for p,t in zip(preds,gts) if p==t)
    return correct/max(1,len(gts))


def aucs_from_logits(logits, gts):
    # Compute AUROC and AUPRC without sklearn
    pairs = sorted([(float(sigmoid(z)), int(t)) for z,t in zip(logits,gts)], key=lambda x: x[0], reverse=True)
    P = sum(t for _,t in pairs); N = len(pairs)-P
    if P==0 or N==0:
        return 0.0, 0.0
    # ROC and PR points
    tp=0; fp=0; roc=[]; prec=[]; rec=[]
    for s,t in pairs:
        if t==1: tp+=1
        else: fp+=1
        tpr = tp/float(P); fpr = fp/float(N)
        roc.append((fpr,tpr))
        precision = tp/float(max(1,tp+fp)); recall = tpr
        prec.append(precision); rec.append(recall)
    roc = sorted(roc)
    auroc=0.0
    for i in range(1,len(roc)):
        x0,y0=roc[i-1]; x1,y1=roc[i]
        auroc += (x1-x0) * (y0+y1)/2.0
    pr = sorted(zip(rec,prec))
    auprc=0.0
    for i in range(1,len(pr)):
        r0,p0=pr[i-1]; r1,p1=pr[i]
        auprc += (r1-r0) * p1
    return auroc, auprc


def load_scores(proc_dir, fname):
    p = os.path.join(proc_dir, fname)
    d={}
    if os.path.exists(p) and os.path.getsize(p)>0:
        for rec in read_jsonl(p): d[rec['item_id']]=float(rec.get('score',0.0))
    return d


def fuse_logits_with_eeg(logits, meta, proc_dir, cfg):
    # combine CNN logits with EEG (P300, SSVEP, ErrP) boosts
    alpha = float(cfg.get('fusion',{}).get('alpha_p300_boost', 1.3))
    beta  = float(cfg.get('fusion',{}).get('beta_ssvep', 0.5))
    gamma = float(cfg.get('fusion',{}).get('gamma_errp', 0.3))
    p300 = load_scores(proc_dir, 'scores_p300.jsonl')
    ssvep= load_scores(proc_dir, 'scores_ssvep.jsonl')
    errp = load_scores(proc_dir, 'scores_errp.jsonl')
    fused=[]
    for z, r in zip(logits, meta):
        iid=r['item_id']
        boost = alpha*p300.get(iid,0.0) + beta*ssvep.get(iid,0.0) + gamma*errp.get(iid,0.0)
        fused.append(z + boost)
    return fused


def main():
    ap = argparse.ArgumentParser(description='Unified pipeline: calibrate → prepare → EEG → train 4 models → eval & report')
    ap.add_argument('--config', default='configs/v3.yaml')
    ap.add_argument('--full', type=int, default=0)
    ap.add_argument('--eeg-key', default='dog')
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    reports_dir = cfg['paths'].get('reports', 'output/reports')
    ensure_dir(reports_dir)

    run_calibrate()
    run_prepare(args.config, full=bool(args.full))
    compute_eeg_scores(args.config, mode='brainflow', key=args.eeg_key)

    # training logs CSV
    logs_dir=os.path.join(cfg['paths'].get('reports','output/reports'),'..','logs')
    logs_dir=os.path.normpath(logs_dir)
    ensure_dir(logs_dir)
    log_path=os.path.join(logs_dir,'train_loss.csv')
    if not os.path.exists(log_path):
        with open(log_path,'w') as f: f.write('epoch,split,model,loss\n')
    def write_log(split, epoch, model, loss):
        with open(log_path,'a') as f: f.write(f"{epoch},{split},{model},{loss}\n")

    base, t_base = train_model(cfg, use_eeg_assist=False, label='baseline', log_writer=write_log)
    eegt, t_eegt = train_model(cfg, use_eeg_assist=True, label='eeg_trained', log_writer=write_log)

    logits_base, gts, meta = predict_logits(cfg, base, split='val')
    logits_eegt, _, _ = predict_logits(cfg, eegt, split='val')

    proc = cfg['paths']['processed']
    logits_base_asst = fuse_logits_with_eeg(logits_base, meta, proc, cfg)
    logits_eegt_asst = fuse_logits_with_eeg(logits_eegt, meta, proc, cfg)

    # metrics
    acc_base = accuracy_from_logits(logits_base, gts)
    acc_eegt = accuracy_from_logits(logits_eegt, gts)
    acc_base_asst = accuracy_from_logits(logits_base_asst, gts)
    acc_eegt_asst = accuracy_from_logits(logits_eegt_asst, gts)
    auroc_base, auprc_base = aucs_from_logits(logits_base, gts)
    auroc_eegt, auprc_eegt = aucs_from_logits(logits_eegt, gts)
    auroc_base_asst, auprc_base_asst = aucs_from_logits(logits_base_asst, gts)
    auroc_eegt_asst, auprc_eegt_asst = aucs_from_logits(logits_eegt_asst, gts)

    energy_base = base.get('energy_J_est', 15.0*t_base)
    energy_eegt = eegt.get('energy_J_est', 15.0*t_eegt)

    run_id = time.strftime('%Y%m%d-%H%M%S', time.localtime()) + '-' + str(random.randint(1000,9999))
    results = {
        'run_id': run_id,
        'models': {
            'baseline': {'accuracy': acc_base, 'auroc': auroc_base, 'auprc': auprc_base, 'train_time_s': t_base, 'energy_J_est': energy_base, 'acc_per_sec': acc_base/max(1e-6,t_base), 'acc_per_kJ': acc_base/max(1e-6,energy_base/1000.0)},
            'eeg_trained': {'accuracy': acc_eegt, 'auroc': auroc_eegt, 'auprc': auprc_eegt, 'train_time_s': t_eegt, 'energy_J_est': energy_eegt, 'acc_per_sec': acc_eegt/max(1e-6,t_eegt), 'acc_per_kJ': acc_eegt/max(1e-6,energy_eegt/1000.0)},
            'assist_only': {'accuracy': acc_base_asst, 'auroc': auroc_base_asst, 'auprc': auprc_base_asst, 'train_time_s': t_base, 'energy_J_est': energy_base, 'acc_per_sec': acc_base_asst/max(1e-6,t_base), 'acc_per_kJ': acc_base_asst/max(1e-6,energy_base/1000.0)},
            'eeg_trained_assist': {'accuracy': acc_eegt_asst, 'auroc': auroc_eegt_asst, 'auprc': auprc_eegt_asst, 'train_time_s': t_eegt, 'energy_J_est': energy_eegt, 'acc_per_sec': acc_eegt_asst/max(1e-6,t_eegt), 'acc_per_kJ': acc_eegt_asst/max(1e-6,energy_eegt/1000.0)}
        },
        'notes': {'energy_note': 'Energy is an estimate from CPU utilization; consider attaching a real meter for accurate energy measurements.'}
    }
    write_jsonl(os.path.join(reports_dir,'benchmark.jsonl'), [results])
    md = os.path.join(reports_dir, f'benchmark_{run_id}.md')
    with open(md,'w') as f:
        f.write('# EEGCompute Benchmark\n\n')
        f.write(f'Run ID: {run_id}\n\n')
        f.write('## Accuracies (val split)\n')
        f.write(f"\n- baseline: {acc_base:.4f}\n- eeg_trained: {acc_eegt:.4f}\n- assist_only: {acc_base_asst:.4f}\n- eeg_trained_assist: {acc_eegt_asst:.4f}\n")
        f.write('\n## AUCs\n')
        f.write(f"\n- baseline: AUROC={auroc_base:.3f}, AUPRC={auprc_base:.3f}\n- eeg_trained: AUROC={auroc_eegt:.3f}, AUPRC={auprc_eegt:.3f}\n- assist_only: AUROC={auroc_base_asst:.3f}, AUPRC={auprc_base_asst:.3f}\n- eeg_trained_assist: AUROC={auroc_eegt_asst:.3f}, AUPRC={auprc_eegt_asst:.3f}\n")
        f.write('\n## Training Cost\n')
        f.write(f"\n- baseline: {t_base:.1f}s (~{energy_base:.0f} J)\n- eeg_trained: {t_eegt:.1f}s (~{energy_eegt:.0f} J)\n")
        f.write('\n## Notes\n')
        f.write(results['notes']['energy_note']+'\n')
    print(f'Wrote {md}')


if __name__=='__main__':
    main()
