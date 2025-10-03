#!/usr/bin/env python3
import argparse, subprocess, sys

def run(cmd):
    print('>',' '.join(cmd)); rc=subprocess.call(cmd); sys.exit(rc)

def main():
    ap=argparse.ArgumentParser(description='Minimal V3 control')
    ap.add_argument('cmd', choices=['prepare','train','calibrate','download','compute-p300','benchmark'])
    ap.add_argument('--config', default='configs/v3.yaml')
    ap.add_argument('--k', type=int, default=10)
    ap.add_argument('--full', type=int, default=0)
    # download options
    ap.add_argument('--train', type=int, default=0, help='Download train2017 images (≈18GB)')
    ap.add_argument('--val', type=int, default=1, help='Download val2017 images (≈1GB)')
    ap.add_argument('--ann', type=int, default=1, help='Download annotations (≈250MB)')
    # compute-p300 extras
    ap.add_argument('--mode', default='synth')
    ap.add_argument('--key', default=None)
    ap.add_argument('--match', default='classes')
    ap.add_argument('--board-id', default='synthetic')
    ap.add_argument('--serial', default=None)
    ap.add_argument('--duration', default='30')
    ap.add_argument('--fs', default=None)
    ap.add_argument('--channels', default=None)
    args=ap.parse_args()
    if args.cmd=='prepare':
        cmd=['python','scripts/prepare.py','--config',args.config]
        if args.full:
            cmd += ['--full', str(args.full)]
        run(cmd)
    if args.cmd=='train': run(['python','scripts/train.py','--config',args.config])
    if args.cmd=='calibrate': run(['python','scripts/calibrate.py'])
    if args.cmd=='download':
        dl = ['python','scripts/download_coco.py']
        dl += ['--val', str(args.val), '--ann', str(args.ann), '--train', str(args.train)]
        run(dl)
    if args.cmd=='compute-p300':
        key = args.key or os.environ.get('P300_KEY','dog')
        cmd = ['python','scripts/compute_p300.py','--mode',args.mode,'--key',key,'--match',args.match,'--out','data/processed/scores_p300.jsonl','--config',args.config]
        if args.mode == 'brainflow':
            cmd += ['--board-id', args.board_id, '--duration', str(args.duration)]
            if args.serial: cmd += ['--serial', args.serial]
            if args.fs: cmd += ['--fs', str(args.fs)]
            if args.channels: cmd += ['--channels', args.channels]
        run(cmd)
    if args.cmd=='benchmark':
        cmd=['python','scripts/pipeline.py','--config',args.config]
        if args.full:
            cmd += ['--full', str(args.full)]
        run(cmd)

if __name__=='__main__':
    main()
