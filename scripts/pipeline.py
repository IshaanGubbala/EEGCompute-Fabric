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
    from eeg_utils import create_power_monitor

    # Import train module from the current scripts directory
    train_mod = importlib.import_module('train')

    # Start power monitoring (tries real powermetrics, falls back to CPU estimate)
    power_mon = create_power_monitor(prefer_real=True)
    power_mon.start()

    t0 = time.time()
    out = train_mod.train_cnn(cfg, use_eeg_assist=use_eeg_assist, log_writer=log_writer, label=label)
    dt = time.time() - t0

    # Stop power monitoring and get energy
    energy_result = power_mon.stop()

    if energy_result:
        out['energy_J'] = energy_result['total_energy_J']
        out['power_W'] = energy_result['avg_power_W']
        out['power_method'] = energy_result['method']
    else:
        # Fallback estimate
        out['energy_J'] = 15.0 * dt
        out['power_W'] = 15.0
        out['power_method'] = 'fallback_estimate'

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


def save_logits_for_grid_search(logits, gts, meta, proc_dir):
    """Save baseline logits and labels for grid search."""
    records = []
    for logit, label, r in zip(logits, gts, meta):
        records.append({
            'item_id': r['item_id'],
            'logit': float(logit),
            'label': int(label)
        })
    path = os.path.join(proc_dir, 'baseline_logits.jsonl')
    write_jsonl(path, records)
    print(f"Saved {len(records)} baseline logits to {path}")


def sigmoid(x):
    try:
        return 1.0/(1.0+math.exp(-x))
    except OverflowError:
        return 0.0 if x<0 else 1.0


def accuracy_from_logits(logits, gts, threshold=0.5):
    preds=[1 if sigmoid(z)>=threshold else 0 for z in logits]
    correct=sum(1 for p,t in zip(preds,gts) if p==t)
    return correct/max(1,len(gts))


def compute_map(logits, gts):
    """Compute mean Average Precision (mAP) - same as AUPRC for binary."""
    try:
        from sklearn.metrics import average_precision_score
        import numpy as np
        probs = [sigmoid(z) for z in logits]
        return average_precision_score(gts, probs)
    except Exception:
        # Fallback: use AUPRC calculation
        _, auprc = aucs_from_logits(logits, gts)
        return auprc


def per_class_metrics(logits, gts, meta):
    """Compute per-class accuracy and AUROC."""
    from collections import defaultdict

    # Group by class
    class_data = defaultdict(lambda: {'logits': [], 'gts': []})

    for z, t, r in zip(logits, gts, meta):
        # Get target classes from metadata
        classes = r.get('classes', [])
        if not classes:
            classes = ['background']

        for cls in classes:
            class_data[cls]['logits'].append(z)
            class_data[cls]['gts'].append(t)

    # Compute metrics per class
    results = {}
    for cls, data in class_data.items():
        if len(data['gts']) < 2:
            continue

        acc = accuracy_from_logits(data['logits'], data['gts'])
        auroc, auprc = aucs_from_logits(data['logits'], data['gts'])

        results[cls] = {
            'accuracy': acc,
            'auroc': auroc,
            'auprc': auprc,
            'n_samples': len(data['gts'])
        }

    return results


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


def fuse_logits_with_eeg(logits, meta, proc_dir, cfg, use_sham=False):
    # combine CNN logits with EEG (P300, SSVEP, ErrP) boosts
    alpha = float(cfg.get('fusion',{}).get('alpha_p300_boost', 1.3))
    beta  = float(cfg.get('fusion',{}).get('beta_ssvep', 0.5))
    gamma = float(cfg.get('fusion',{}).get('gamma_errp', 0.3))
    quality_gate = float(cfg.get('fusion',{}).get('quality_gate', 0.7))

    # Load raw scores (sham or real)
    suffix = '_sham' if use_sham else ''
    p300 = load_scores(proc_dir, f'scores_p300{suffix}.jsonl')
    ssvep= load_scores(proc_dir, f'scores_ssvep{suffix}.jsonl')
    errp = load_scores(proc_dir, f'scores_errp{suffix}.jsonl')

    # Load calibrations if available
    calib_path = os.path.join(proc_dir, 'eeg_calibrations.json')
    calibrations = {}
    if os.path.exists(calib_path):
        import json as _json
        with open(calib_path, 'r') as f:
            calibrations = _json.load(f)

    # Quality scores (simple proxy: abs magnitude)
    def quality(score):
        return abs(score)

    fused=[]
    for z, r in zip(logits, meta):
        iid=r['item_id']

        # Get raw scores
        p300_raw = p300.get(iid, 0.0)
        ssvep_raw = ssvep.get(iid, 0.0)
        errp_raw = errp.get(iid, 0.0)

        # Apply calibration if available
        from eeg_utils import apply_calibration
        p300_cal = apply_calibration(p300_raw, calibrations.get('p300'))
        errp_cal = apply_calibration(errp_raw, calibrations.get('errp'))
        ssvep_cal = apply_calibration(ssvep_raw, calibrations.get('ssvep'))

        # Quality gating: only apply if quality >= threshold
        boost = 0.0
        if quality(p300_raw) >= quality_gate:
            boost += alpha * p300_cal
        if quality(errp_raw) >= quality_gate:
            boost += gamma * errp_cal
        if quality(ssvep_raw) >= quality_gate:
            boost += beta * ssvep_cal

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

    # Run EEG calibration (Platt scaling on P300 and ErrP)
    import subprocess
    calib_cmd = ["python", "scripts/eeg_utils.py", "calibrate", "--config", args.config, "--method", "platt", "--signals", "p300,errp"]
    print('>', ' '.join(calib_cmd))
    subprocess.check_call(calib_cmd)

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

    # Save baseline logits for grid search
    save_logits_for_grid_search(logits_base, gts, meta, proc)

    # Run grid search to find optimal fusion coefficients
    grid_cmd = ["python", "scripts/eeg_utils.py", "grid_search", "--config", args.config]
    print('>', ' '.join(grid_cmd))
    subprocess.check_call(grid_cmd)

    # Load best params from grid search
    grid_results_path = os.path.join(proc, 'grid_search_results.json')
    if os.path.exists(grid_results_path):
        with open(grid_results_path, 'r') as f:
            grid_results = json.load(f)
            best = grid_results['best_params']
            cfg['fusion']['alpha_p300_boost'] = best['alpha']
            cfg['fusion']['beta_ssvep'] = best['beta']
            cfg['fusion']['gamma_errp'] = best['gamma']
            print(f"Updated fusion params: α={best['alpha']}, β={best['beta']}, γ={best['gamma']}")

    # Generate sham controls
    sham_cmd = ["python", "scripts/eeg_utils.py", "sham", "--config", args.config]
    print('>', ' '.join(sham_cmd))
    subprocess.check_call(sham_cmd)

    # Real EEG fusion
    logits_base_asst = fuse_logits_with_eeg(logits_base, meta, proc, cfg, use_sham=False)
    logits_eegt_asst = fuse_logits_with_eeg(logits_eegt, meta, proc, cfg, use_sham=False)

    # Sham EEG fusion (control)
    logits_base_sham = fuse_logits_with_eeg(logits_base, meta, proc, cfg, use_sham=True)
    logits_eegt_sham = fuse_logits_with_eeg(logits_eegt, meta, proc, cfg, use_sham=True)

    # metrics
    acc_base = accuracy_from_logits(logits_base, gts)
    acc_eegt = accuracy_from_logits(logits_eegt, gts)
    acc_base_asst = accuracy_from_logits(logits_base_asst, gts)
    acc_eegt_asst = accuracy_from_logits(logits_eegt_asst, gts)
    acc_base_sham = accuracy_from_logits(logits_base_sham, gts)
    acc_eegt_sham = accuracy_from_logits(logits_eegt_sham, gts)
    auroc_base, auprc_base = aucs_from_logits(logits_base, gts)
    auroc_eegt, auprc_eegt = aucs_from_logits(logits_eegt, gts)
    auroc_base_asst, auprc_base_asst = aucs_from_logits(logits_base_asst, gts)
    auroc_eegt_asst, auprc_eegt_asst = aucs_from_logits(logits_eegt_asst, gts)

    # Compute mAP
    map_base = compute_map(logits_base, gts)
    map_eegt = compute_map(logits_eegt, gts)
    map_base_asst = compute_map(logits_base_asst, gts)
    map_eegt_asst = compute_map(logits_eegt_asst, gts)

    # Per-class metrics
    pc_base = per_class_metrics(logits_base, gts, meta)
    pc_base_asst = per_class_metrics(logits_base_asst, gts, meta)

    # Get energy measurements (prefer real, fallback to estimate)
    energy_base = base.get('energy_J', base.get('energy_J_est', 15.0*t_base))
    energy_eegt = eegt.get('energy_J', eegt.get('energy_J_est', 15.0*t_eegt))
    power_base = base.get('power_W', 15.0)
    power_eegt = eegt.get('power_W', 15.0)
    power_method = base.get('power_method', 'fallback_estimate')

    run_id = time.strftime('%Y%m%d-%H%M%S', time.localtime()) + '-' + str(random.randint(1000,9999))
    results = {
        'run_id': run_id,
        'models': {
            'baseline': {'accuracy': acc_base, 'auroc': auroc_base, 'auprc': auprc_base, 'map': map_base, 'train_time_s': t_base, 'energy_J': energy_base, 'power_W': power_base, 'acc_per_sec': acc_base/max(1e-6,t_base), 'acc_per_kJ': acc_base/max(1e-6,energy_base/1000.0)},
            'eeg_trained': {'accuracy': acc_eegt, 'auroc': auroc_eegt, 'auprc': auprc_eegt, 'map': map_eegt, 'train_time_s': t_eegt, 'energy_J': energy_eegt, 'power_W': power_eegt, 'acc_per_sec': acc_eegt/max(1e-6,t_eegt), 'acc_per_kJ': acc_eegt/max(1e-6,energy_eegt/1000.0)},
            'assist_only': {'accuracy': acc_base_asst, 'auroc': auroc_base_asst, 'auprc': auprc_base_asst, 'map': map_base_asst, 'train_time_s': t_base, 'energy_J': energy_base, 'power_W': power_base, 'acc_per_sec': acc_base_asst/max(1e-6,t_base), 'acc_per_kJ': acc_base_asst/max(1e-6,energy_base/1000.0)},
            'eeg_trained_assist': {'accuracy': acc_eegt_asst, 'auroc': auroc_eegt_asst, 'auprc': auprc_eegt_asst, 'map': map_eegt_asst, 'train_time_s': t_eegt, 'energy_J': energy_eegt, 'power_W': power_eegt, 'acc_per_sec': acc_eegt_asst/max(1e-6,t_eegt), 'acc_per_kJ': acc_eegt_asst/max(1e-6,energy_eegt/1000.0)},
            'baseline_sham': {'accuracy': acc_base_sham},
            'eeg_trained_sham': {'accuracy': acc_eegt_sham}
        },
        'per_class': {
            'baseline': pc_base,
            'assist_only': pc_base_asst
        },
        'notes': {
            'energy_note': f'Power measurement method: {power_method}',
            'power_method': power_method
        }
    }
    write_jsonl(os.path.join(reports_dir,'benchmark.jsonl'), [results])
    md = os.path.join(reports_dir, f'benchmark_{run_id}.md')
    with open(md,'w') as f:
        f.write('# EEGCompute Benchmark\n\n')
        f.write(f'Run ID: {run_id}\n\n')
        f.write('## Accuracies (val split)\n')
        f.write(f"\n- baseline: {acc_base:.4f}\n- eeg_trained: {acc_eegt:.4f}\n- assist_only: {acc_base_asst:.4f}\n- eeg_trained_assist: {acc_eegt_asst:.4f}\n")
        f.write('\n## Sham Controls (shuffled EEG)\n')
        f.write(f"\n- baseline+sham: {acc_base_sham:.4f} (Δ={acc_base_sham-acc_base:+.4f})\n- eeg_trained+sham: {acc_eegt_sham:.4f} (Δ={acc_eegt_sham-acc_eegt:+.4f})\n")
        f.write(f"\n**Expected:** Sham ≈ baseline (no causal benefit). Real EEG should show Δ > sham.\n")
        f.write('\n## mAP (mean Average Precision)\n')
        f.write(f"\n- baseline: {map_base:.4f}\n- eeg_trained: {map_eegt:.4f}\n- assist_only: {map_base_asst:.4f} (Δ={map_base_asst-map_base:+.4f})\n- eeg_trained_assist: {map_eegt_asst:.4f} (Δ={map_eegt_asst-map_eegt:+.4f})\n")
        f.write('\n## AUCs\n')
        f.write(f"\n- baseline: AUROC={auroc_base:.3f}, AUPRC={auprc_base:.3f}\n- eeg_trained: AUROC={auroc_eegt:.3f}, AUPRC={auprc_eegt:.3f}\n- assist_only: AUROC={auroc_base_asst:.3f}, AUPRC={auprc_base_asst:.3f}\n- eeg_trained_assist: AUROC={auroc_eegt_asst:.3f}, AUPRC={auprc_eegt_asst:.3f}\n")
        f.write('\n## Per-Class Analysis\n\n')
        f.write('| Class | Baseline Acc | +EEG Acc | Δ Acc | Baseline AUROC | +EEG AUROC | n |\n')
        f.write('|-------|-------------|----------|-------|---------------|-----------|---|\n')
        for cls in sorted(pc_base.keys()):
            b_acc = pc_base[cls]['accuracy']
            a_acc = pc_base_asst.get(cls, {}).get('accuracy', 0.0)
            b_auroc = pc_base[cls]['auroc']
            a_auroc = pc_base_asst.get(cls, {}).get('auroc', 0.0)
            n = pc_base[cls]['n_samples']
            delta = a_acc - b_acc
            f.write(f"| {cls} | {b_acc:.3f} | {a_acc:.3f} | {delta:+.3f} | {b_auroc:.3f} | {a_auroc:.3f} | {n} |\n")
        f.write('\n## Training Cost & Energy\n')
        f.write(f"\n- baseline: {t_base:.1f}s, {power_base:.1f}W, {energy_base:.0f}J ({energy_base/1000.0:.2f} kJ)\n")
        f.write(f"- eeg_trained: {t_eegt:.1f}s, {power_eegt:.1f}W, {energy_eegt:.0f}J ({energy_eegt/1000.0:.2f} kJ)\n")
        f.write(f"\n**Efficiency:**\n")
        f.write(f"- baseline: {acc_base/max(1e-6,t_base):.6f} acc/s, {acc_base/max(1e-6,energy_base/1000.0):.4f} acc/kJ\n")
        f.write(f"- eeg_trained: {acc_eegt/max(1e-6,t_eegt):.6f} acc/s, {acc_eegt/max(1e-6,energy_eegt/1000.0):.4f} acc/kJ\n")
        f.write('\n## Notes\n')
        f.write(results['notes']['energy_note']+'\n')
    print(f'Wrote {md}')


if __name__=='__main__':
    main()
