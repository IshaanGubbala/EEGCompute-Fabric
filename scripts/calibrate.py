#!/usr/bin/env python3
import argparse, json, os
from utils import ensure_dir

BARELY={
  "sim":{
    "p300":{"beta_target":[6,5],"beta_nontarget":[5,6],"drop_rate":0.20,"latency_jitter_ms":80},
    "ssvep":{"base_noise":0.22,"attended_boost":0.18},
    "errp":{"base_rate":0.35,"quality_mean":0.70,"quality_std":0.20}
  }
}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out', default='data/processed/calibration.json'); args=ap.parse_args()
    ensure_dir(os.path.dirname(args.out))
    with open(args.out,'w') as f: json.dump(BARELY,f,indent=2)
    print(f'Wrote {args.out} (barely-pass calibration)')

if __name__=='__main__':
    main()

