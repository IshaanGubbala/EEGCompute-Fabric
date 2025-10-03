#!/usr/bin/env python3
import argparse, json, os
from utils import ensure_dir

BARELY={
  "sim":{
    "p300":{"beta_target":[6,5],"beta_nontarget":[5,6],"drop_rate":0.20,"latency_jitter_ms":80},
    "ssvep":{"base_noise":0.22,"attended_boost":0.18},
    "errp":{"base_rate":0.35,"quality_mean":0.70,"quality_std":0.20}
  },
  "eeg":{
    "p300":{
      "bandpass_hz":[0.1,15.0],
      "baseline_window_s":[-0.2,0.0],
      "post_window_s":[0.3,0.6],
      "channels_hint":["Pz","Cz","P3","P4"],
      "rsvp_rate_hz": 8.0,
      "offset_s": 0.0
    },
    "ssvep":{
      "freq_hz": 12.0,
      "band_hz": [10.0,14.0],
      "baseline_window_s": [-0.2,0.0],
      "post_window_s": [0.3,0.6]
    },
    "errp":{
      "baseline_window_s": [-0.1,0.0],
      "post_window_s": [0.2,0.4]
    }
  }
}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out', default='data/processed/calibration.json'); args=ap.parse_args()
    ensure_dir(os.path.dirname(args.out))
    with open(args.out,'w') as f: json.dump(BARELY,f,indent=2)
    print(f'Wrote {args.out} (barely-pass calibration)')

if __name__=='__main__':
    main()
