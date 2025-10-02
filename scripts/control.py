#!/usr/bin/env python3
import argparse, subprocess, sys

def run(cmd):
    print('>',' '.join(cmd)); rc=subprocess.call(cmd); sys.exit(rc)

def main():
    ap=argparse.ArgumentParser(description='Minimal V3 control')
    ap.add_argument('cmd', choices=['prepare','train','calibrate','eval','download'])
    ap.add_argument('--config', default='configs/v3.yaml')
    ap.add_argument('--k', type=int, default=10)
    # download options
    ap.add_argument('--train', type=int, default=0, help='Download train2017 images (≈18GB)')
    ap.add_argument('--val', type=int, default=1, help='Download val2017 images (≈1GB)')
    ap.add_argument('--ann', type=int, default=1, help='Download annotations (≈250MB)')
    args=ap.parse_args()
    if args.cmd=='prepare': run(['python','scripts/prepare.py','--config',args.config])
    if args.cmd=='train': run(['python','scripts/train.py','--config',args.config])
    if args.cmd=='calibrate': run(['python','scripts/calibrate.py'])
    if args.cmd=='eval': run(['python','scripts/eval.py','--config',args.config,'--k',str(args.k),'--strict_locked','1'])
    if args.cmd=='download':
        dl = ['python','scripts/download_coco.py']
        dl += ['--val', str(args.val), '--ann', str(args.ann), '--train', str(args.train)]
        run(dl)

if __name__=='__main__':
    main()
