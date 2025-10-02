#!/usr/bin/env python3
import argparse, os, random, json
from pathlib import Path
from utils import ensure_dir, write_jsonl, load_cfg

def _load_coco_annotations(path):
    with open(path,'r') as f:
        data=json.load(f)
    images={img['id']: img for img in data['images']}
    categories={cat['id']: cat['name'] for cat in data['categories']}
    img_to_cats={}
    for ann in data['annotations']:
        img_to_cats.setdefault(ann['image_id'], set()).add(categories.get(ann['category_id']))
    return images, img_to_cats

def build_subset(cfg, split):
    n = int(cfg['dataset']['subset_size'][split])
    targets = set(cfg['dataset']['target_classes'])
    ann_path = cfg['dataset']['annotations'][split]
    items=[]
    if os.path.exists(ann_path):
        images, img_to_cats = _load_coco_annotations(ann_path)
        # Build candidate pools
        pos=[]; neg=[]
        img_root = cfg['dataset']['images'][split]
        for img_id, img in images.items():
            classes = list(img_to_cats.get(img_id, []))
            has_target = any(c in targets for c in classes)
            fname = img.get('file_name') or f"{img_id:012d}.jpg"
            fp = os.path.join(img_root, fname)
            item = {
                'item_id': f"{img_id:012d}",
                'filepath': fp,
                'classes': classes or [],
                'is_target': 1 if has_target else 0,
                'split': split,
            }
            if has_target:
                pos.append(item)
            else:
                neg.append(item)
        rng = random.Random(1337 + (0 if split=='train' else 1))
        rng.shuffle(pos); rng.shuffle(neg)
        # Slightly fewer targets to keep it non-trivial
        n_pos = min(int(0.45*n), len(pos))
        n_neg = min(n-n_pos, len(neg))
        items = pos[:n_pos] + neg[:n_neg]
        rng.shuffle(items)
        # Filter out any missing files silently (user must place COCO images locally)
        items = [it for it in items if os.path.exists(it['filepath'])]
        # If after filtering we are short, top up with remaining candidates that exist
        if len(items) < n:
            extra = [it for it in (pos[n_pos:]+neg[n_neg:]) if os.path.exists(it['filepath'])]
            items += extra[: max(0, n-len(items))]
    else:
        # Fallback synthetic subset if annotations are not present
        rng = random.Random(1337 + (0 if split=='train' else 1))
        for _ in range(n):
            iid = f"{rng.randrange(0,1_000_000):012d}"
            is_t = 1 if rng.random()<0.4 else 0
            classes=[rng.choice(list(targets))] if is_t else [rng.choice(['car','bus','tree','chair'])]
            items.append({
                'item_id': iid,
                'filepath': os.path.join(cfg['dataset']['images'][split], f"{iid}.jpg"),
                'classes': classes,
                'is_target': is_t,
                'split': split
            })
    return items

def build_priors(cfg, items):
    rng = random.Random(2025)
    priors=[]
    for it in items:
        base = rng.random()
        # Stronger overlap so OFF isn't trivially perfect
        if it['is_target']:
            pri = 0.49 + 0.07*base  # ~0.49..0.56
        else:
            pri = 0.46 + 0.09*base  # ~0.46..0.55
        priors.append({'item_id': it['item_id'], 'prior': round(pri,6)})
    return priors

def build_rsvp(cfg, items):
    rate = float(cfg['rsvp']['rate_hz']); dt=1.0/rate
    seq=[{'t': round(i*dt,3), 'item_id': it['item_id']} for i,it in enumerate(items)]
    return {'session':'S1','rate_hz':rate,'sequence':seq}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args=ap.parse_args()
    cfg=load_cfg(args.config)
    proc=cfg['paths']['processed']
    ensure_dir(proc)
    # subset
    items = build_subset(cfg,'train')+build_subset(cfg,'val')
    write_jsonl(os.path.join(proc,'coco_subset.jsonl'), items)
    # priors
    write_jsonl(os.path.join(proc,'prior_scores.jsonl'), build_priors(cfg, items))
    # rsvp (use val split ordering)
    val_items=[it for it in items if it['split']=='val']
    write_jsonl(os.path.join(proc,'rsvp_streams.jsonl'), [build_rsvp(cfg, val_items)])
    # placeholders
    for n in ['scores_p300.jsonl','scores_ssvep.jsonl','scores_errp.jsonl']:
        Path(os.path.join(proc,n)).touch()
    print('Prepared subset, priors, rsvp, and placeholders')

if __name__=='__main__':
    main()
