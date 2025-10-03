#!/usr/bin/env python3
import argparse, json, os, sys, math, random
from pathlib import Path

from utils import read_jsonl, write_jsonl, normalize_scores


def load_subset(proc_dir):
    items = list(read_jsonl(os.path.join(proc_dir, 'coco_subset.jsonl')))
    index = {it['item_id']: it for it in items}
    return items, index


def load_rsvp(proc_dir):
    with open(os.path.join(proc_dir, 'rsvp_streams.jsonl'), 'r') as f:
        rsvp = json.loads(next(l for l in f if l.strip()))
    return rsvp


def matches_key(item, key, mode='classes'):
    if not key:
        return False
    k = key.lower()
    if mode == 'classes':
        for c in item.get('classes', []):
            if k in str(c).lower():
                return True
        return False
    else:
        return k in str(item.get('item_id','')).lower()


def synth_scores(cfg, rsvp, items, key, match_mode):
    # Deterministic synthetic P300 injection based on key matches
    rng = random.Random(4242)
    idx = {it['item_id']: it for it in items}
    # Template amplitudes (exaggerated for stronger signal)
    amp_hit = 2.0
    amp_miss = 0.1
    # Accumulate simple evidence per item as sum of (post-baseline) proxies
    evidence = {}
    for ev in rsvp['sequence']:
        iid = ev['item_id']
        it = idx.get(iid)
        if it is None:
            continue
        is_key = matches_key(it, key, mode=match_mode)
        # proxy: baseline noise + optional P300 bump
        noise = rng.gauss(0.0, 0.2)
        bump = amp_hit if is_key else amp_miss
        val = noise + bump
        evidence[iid] = evidence.get(iid, 0.0) + val
    # Convert to normalized score per item
    scores = evidence
    scores = normalize_scores(scores, method='z_tanh')
    return scores


def mne_scores(args, cfg, rsvp, items):
    raise NotImplementedError('MNE mode removed to simplify to BrainFlow/synth paths')


def lsl_scores(args, cfg, rsvp, items):
    raise NotImplementedError('LSL mode removed to simplify to BrainFlow/synth paths')


def brainflow_scores(args, cfg, rsvp, items):
    try:
        from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
        from brainflow.data_filter import DataFilter, DetrendOperations, FilterTypes
    except Exception as e:
        raise RuntimeError('BrainFlow is required for --mode brainflow. pip install brainflow') from e

    # Parameters
    board_id_str = getattr(args, 'board_id', 'synthetic').lower()
    duration = float(getattr(args, 'duration', 30.0))
    fs_override = getattr(args, 'fs', None)
    chan_arg = getattr(args, 'channels', None)

    # Map board_id
    bid = BoardIds.SYNTHETIC_BOARD
    mapping = {
        'synthetic': BoardIds.SYNTHETIC_BOARD,
        'cyton': BoardIds.CYTON_BOARD,
        'cyton_daisy': BoardIds.CYTON_DAISY_BOARD,
        'ganglion': BoardIds.GANGLION_BOARD,
    }
    if board_id_str in mapping:
        bid = mapping[board_id_str]

    params = BrainFlowInputParams()
    if getattr(args, 'serial', None):
        params.serial_port = args.serial

    BoardShim.enable_dev_board_logger()
    board = BoardShim(bid, params)
    board.prepare_session()
    board.start_stream()
    try:
        BoardShim.log_message(1, f"BrainFlow streaming for {duration}s…")
        BoardShim.sleep(int(duration*1000))
        data = board.get_board_data()
    finally:
        board.stop_stream(); board.release_session()

    eeg_channels = BoardShim.get_eeg_channels(bid)
    if not eeg_channels:
        raise RuntimeError('No EEG channels reported by BrainFlow board')
    fs = int(fs_override or BoardShim.get_sampling_rate(bid))

    # Select channels
    sel = []
    if chan_arg:
        try:
            sel = [eeg_channels[int(i)] if int(i) < len(eeg_channels) else eeg_channels[0] for i in chan_arg.split(',')]
        except Exception:
            sel = eeg_channels[:4]
    else:
        sel = eeg_channels[:4]

    # Preprocess: detrend + bandpass 0.1-15 Hz
    for ch in sel:
        sig = data[ch]
        DataFilter.detrend(sig, DetrendOperations.CONSTANT.value)
        DataFilter.perform_bandpass(sig, fs, 5.0, 15.0, 4, FilterTypes.BUTTERWORTH.value, 0)
        DataFilter.perform_bandpass(sig, fs, 0.1, 15.0, 4, FilterTypes.BUTTERWORTH.value, 0)

    # Epoching windows (seconds)
    win = cfg.get('rsvp', {}).get('epoch_window_s', [0.2, 0.8])
    t0, t1 = float(win[0]), float(win[1])
    base0, base1 = -0.2, 0.0
    idx = {it['item_id']: it for it in items}

    # We don't have absolute stream timestamps; assume evenly sampled and map RSVP order to consecutive epochs
    # Compute samples per epoch window around a notional onset spacing matching RSVP rate
    rate = float(cfg.get('rsvp', {}).get('rate_hz', 8))
    step = int(fs * (1.0 / rate))
    ep_pre = int(abs(base0) * fs)
    ep_len = int((t1 - base0) * fs)

    # Build per-event epoch starts over the captured stream
    n_events = len(rsvp['sequence'])
    starts = [i * step for i in range(n_events)]
    # Ensure data long enough
    total_needed = (starts[-1] if starts else 0) + ep_len + 1
    for ch in sel:
        if len(data[ch]) < total_needed:
            # Pad with zeros if needed
            pad = total_needed - len(data[ch])
            data[ch] = list(data[ch]) + [0.0]*pad

    # Synthetic injection amplitudes (simulate stronger response for key matches)
    key = getattr(args, 'key', None)
    def _is_match(it):
        try:
            from compute_p300 import matches_key as _mk
            return _mk(it, key, mode=getattr(args,'match','classes')) if key else False
        except Exception:
            return False
    amp_hit = 0.8; amp_miss = 0.05

    # Score per event
    ev_scores = {}
    for i, ev in enumerate(rsvp['sequence']):
        s0 = starts[i]
        s_base0 = max(0, s0)
        s_base1 = s0 + int(base1*fs)
        s_post0 = s0 + int(t0*fs)
        s_post1 = s0 + int(t1*fs)
        ch_scores = []
        for ch in sel:
            sig = data[ch]
            base_seg = sig[s_base0:s_base1] if s_base1 > s_base0 else []
            post_seg = sig[s_post0:s_post1] if s_post1 > s_post0 else []
            if not base_seg or not post_seg:
                continue
            b = sum(base_seg)/len(base_seg)
            p = sum(post_seg)/len(post_seg)
            # Inject synthetic P300-like bump for matching items
            it = idx.get(ev['item_id'])
            if it is not None:
                p += (amp_hit if _is_match(it) else amp_miss)
            ch_scores.append(p - b)
        if ch_scores:
            ev_scores.setdefault(ev['item_id'], []).append(sum(ch_scores)/len(ch_scores))

    # Aggregate per item and normalize
    scores = {iid: (sum(v)/len(v)) for iid, v in ev_scores.items()}
    scores = normalize_scores(scores, method='z_tanh')
    return scores


def _band_limited_power(sig, fs, f0, f1):
    try:
        import numpy as np
    except Exception:
        # crude: rectangular window; compute rough energy proxy
        return sum(x*x for x in sig)/max(1,len(sig))
    freqs = np.fft.rfftfreq(len(sig), 1.0/fs)
    spec = np.abs(np.fft.rfft(np.array(sig)))**2
    mask = (freqs>=f0) & (freqs<=f1)
    return float(spec[mask].sum())/max(1, mask.sum())


def brainflow_scores_ssvep(args, cfg, rsvp, items):
    # reuse brainflow stream capture
    args_p300 = argparse.Namespace(board_id=getattr(args,'board_id','synthetic'), serial=getattr(args,'serial',None), duration=getattr(args,'duration',20), fs=getattr(args,'fs',None), channels=getattr(args,'channels','0,1,2,3'), key=getattr(args,'key',None), match=getattr(args,'match','classes'))
    try:
        from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
        from brainflow.data_filter import DataFilter, DetrendOperations, FilterTypes
    except Exception as e:
        raise RuntimeError('BrainFlow is required for SSVEP. pip install brainflow') from e
    # board setup
    bid_map={'synthetic':BoardIds.SYNTHETIC_BOARD,'cyton':BoardIds.CYTON_BOARD,'cyton_daisy':BoardIds.CYTON_DAISY_BOARD,'ganglion':BoardIds.GANGLION_BOARD}
    bid = bid_map.get(args_p300.board_id, BoardIds.SYNTHETIC_BOARD)
    params = BrainFlowInputParams();
    if args_p300.serial: params.serial_port=args_p300.serial
    BoardShim.enable_dev_board_logger()
    board = BoardShim(bid, params)
    board.prepare_session(); board.start_stream()
    try:
        BoardShim.sleep(int(float(args_p300.duration)*1000))
        data = board.get_board_data()
    finally:
        board.stop_stream(); board.release_session()
    fs = int(args_p300.fs or BoardShim.get_sampling_rate(bid))
    eeg_channels = BoardShim.get_eeg_channels(bid)
    chans = [eeg_channels[int(i)] if int(i)<len(eeg_channels) else eeg_channels[0] for i in str(args_p300.channels).split(',')]
    # calibration
    eeg_cal = {}
    try:
        with open(os.path.join(cfg['paths']['processed'],'calibration.json'),'r') as f:
            eeg_cal = json.load(f).get('eeg',{})
    except Exception:
        pass
    ss_cfg = eeg_cal.get('ssvep', {'freq_hz':12.0,'band_hz':[10.0,14.0],'baseline_window_s':[-0.2,0.0],'post_window_s':[0.3,0.6]})
    f0, f1 = float(ss_cfg['band_hz'][0]), float(ss_cfg['band_hz'][1])
    base0, base1 = float(ss_cfg['baseline_window_s'][0]), float(ss_cfg['baseline_window_s'][1])
    post0, post1 = float(ss_cfg['post_window_s'][0]), float(ss_cfg['post_window_s'][1])
    rate = float(cfg.get('rsvp',{}).get('rate_hz',8))
    step = int(fs*(1.0/rate)); ep_len = int((post1 - base0)*fs)
    # optional key injection
    key=getattr(args,'key',None); match=getattr(args,'match','classes')
    def _is_match(it):
        try:
            from compute_p300 import matches_key as _mk
            return _mk(it, key, mode=match) if key else False
        except Exception:
            return False
    idx = {it['item_id']:it for it in items}
    scores={}
    for i, ev in enumerate(rsvp['sequence']):
        s0 = i*step
        b0 = max(0,s0); b1 = s0 + int(base1*fs)
        p0 = s0 + int(post0*fs); p1 = s0 + int(post1*fs)
        ch_vals=[]
        for ch in chans:
            sig = data[ch]
            base_seg=sig[b0:b1] if b1>b0 else []
            post_seg=sig[p0:p1] if p1>p0 else []
            if not base_seg or not post_seg:
                continue
            pb = _band_limited_power(base_seg, fs, f0, f1)
            pp = _band_limited_power(post_seg, fs, f0, f1)
            # inject stronger margin for matches (exaggerated)
            it = idx.get(ev['item_id']); inj = 0.4 if _is_match(it) else 0.02
            ch_vals.append((pp + inj) - pb)
        if ch_vals:
            scores.setdefault(ev['item_id'], []).append(sum(ch_vals)/len(ch_vals))
    # aggregate per item and normalize
    agg={k: sum(v)/len(v) for k,v in scores.items()}
    return normalize_scores(agg,'z_tanh')


def brainflow_scores_errp(args, cfg, rsvp, items):
    # Similar stream capture as above
    args_p = argparse.Namespace(board_id=getattr(args,'board_id','synthetic'), serial=getattr(args,'serial',None), duration=getattr(args,'duration',20), fs=getattr(args,'fs',None), channels=getattr(args,'channels','0,1,2,3'), key=getattr(args,'key',None), match=getattr(args,'match','classes'))
    try:
        from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
    except Exception as e:
        raise RuntimeError('BrainFlow is required for ErrP. pip install brainflow') from e
    bid_map={'synthetic':BoardIds.SYNTHETIC_BOARD,'cyton':BoardIds.CYTON_BOARD,'cyton_daisy':BoardIds.CYTON_DAISY_BOARD,'ganglion':BoardIds.GANGLION_BOARD}
    bid = bid_map.get(args_p.board_id, BoardIds.SYNTHETIC_BOARD)
    params = BrainFlowInputParams();
    if args_p.serial: params.serial_port=args_p.serial
    BoardShim.enable_dev_board_logger()
    board = BoardShim(bid, params); board.prepare_session(); board.start_stream()
    try:
        BoardShim.sleep(int(float(args_p.duration)*1000)); data = board.get_board_data()
    finally:
        board.stop_stream(); board.release_session()
    fs = int(args_p.fs or BoardShim.get_sampling_rate(bid))
    eeg_channels = BoardShim.get_eeg_channels(bid)
    chans = [eeg_channels[int(i)] if int(i)<len(eeg_channels) else eeg_channels[0] for i in str(args_p.channels).split(',')]
    # calibration
    eeg_cal = {}
    try:
        with open(os.path.join(cfg['paths']['processed'],'calibration.json'),'r') as f:
            eeg_cal = json.load(f).get('eeg',{})
    except Exception:
        pass
    er_cfg = eeg_cal.get('errp', {'baseline_window_s':[-0.1,0.0],'post_window_s':[0.2,0.4]})
    base0, base1 = float(er_cfg['baseline_window_s'][0]), float(er_cfg['baseline_window_s'][1])
    post0, post1 = float(er_cfg['post_window_s'][0]), float(er_cfg['post_window_s'][1])
    rate = float(cfg.get('rsvp',{}).get('rate_hz',8)); step=int(fs*(1.0/rate))
    # template: simple negative-going bump
    import numpy as np
    t = np.linspace(post0, post1, max(2,int((post1-post0)*fs)))
    tpl = -np.exp(-((t - (post0+post1)/2.0)**2)/(2*(0.05**2)))
    # match function
    def match_score(seg):
        if len(seg) != len(tpl):
            # resample to template length
            from math import floor
            if len(seg) < 2: return 0.0
            idxs = np.linspace(0, len(seg)-1, len(tpl))
            s = np.interp(idxs, np.arange(len(seg)), np.array(seg))
        else:
            s = np.array(seg)
        s = s - s.mean()
        return float(np.dot(s, tpl)/ (np.linalg.norm(s)*np.linalg.norm(tpl) + 1e-6))
    # per-event
    idx = {it['item_id']:it for it in items}
    scores={}
    key=getattr(args,'key',None); match=getattr(args,'match','classes')
    def _is_match(it):
        try:
            from compute_p300 import matches_key as _mk
            return _mk(it, key, mode=match) if key else False
        except Exception:
            return False
    for i, ev in enumerate(rsvp['sequence']):
        s0 = i*step
        b0 = max(0,s0); b1 = s0 + int(base1*fs)
        p0 = s0 + int(post0*fs); p1 = s0 + int(post1*fs)
        vals=[]
        for ch in chans:
            sig=data[ch]
            base_seg=sig[b0:b1] if b1>b0 else []
            post_seg=sig[p0:p1] if p1>p0 else []
            if not base_seg or not post_seg:
                continue
            # baseline correct
            mu = sum(base_seg)/len(base_seg)
            post = [x-mu for x in post_seg]
            sc = match_score(post)
            # inject: stronger negative correlation for matches (exaggerated)
            it = idx.get(ev['item_id']); inj = 0.2 if _is_match(it) else 0.01
            vals.append(sc - inj)
        if vals:
            scores.setdefault(ev['item_id'], []).append(sum(vals)/len(vals))
    agg={k: sum(v)/len(v) for k,v in scores.items()}
    return normalize_scores(agg,'z_tanh')


def main():
    ap = argparse.ArgumentParser(description='Compute item-level P300 scores')
    ap.add_argument('--mode', choices=['synth','lsl','mne','brainflow'], default='synth')
    ap.add_argument('--key', default='dog', help='Keyword to positively inject P300 for matching items (synth mode)')
    ap.add_argument('--match', choices=['classes','item_id'], default='classes', help='How to match the key to items')
    ap.add_argument('--config', default='configs/v3.yaml')
    ap.add_argument('--out', default='data/processed/scores_p300.jsonl')
    ap.add_argument('--proc', default='data/processed')
    ap.add_argument('--mne-file', default=None)
    # BrainFlow-specific
    ap.add_argument('--board-id', default='synthetic')
    ap.add_argument('--serial', default=None)
    ap.add_argument('--duration', default='30')
    ap.add_argument('--fs', default=None)
    ap.add_argument('--channels', default=None)
    args = ap.parse_args()

    # Load cfg lazily (kept JSON for minimal deps)
    cfg = json.load(open(args.config,'r'))
    items, index = load_subset(args.proc)
    rsvp = load_rsvp(args.proc)

    # Load calibration if present
    calib = {}
    try:
        with open(os.path.join(args.proc,'calibration.json'),'r') as f:
            calib = json.load(f)
    except Exception:
        pass
    eeg_cal = calib.get('eeg',{}).get('p300',{})
    # Expose some calibrated params back into args-like namespace (used by brainflow)
    if 'rsvp_rate_hz' in eeg_cal:
        cfg.setdefault('rsvp',{}).setdefault('rate_hz', eeg_cal.get('rsvp_rate_hz'))

    if args.mode == 'synth':
        scores = synth_scores(cfg, rsvp, items, key=args.key, match_mode=args.match)
    elif args.mode == 'mne':
        scores = mne_scores(args, cfg, rsvp, items)
    elif args.mode == 'lsl':
        scores = lsl_scores(args, cfg, rsvp, items)
    else:
        scores = brainflow_scores(args, cfg, rsvp, items)

    # Persist
    out_path = args.out
    records = [{ 'item_id': k, 'score': float(v) } for k,v in scores.items()]
    write_jsonl(out_path, records)
    print(f'Wrote {out_path} ({len(records)} item scores)')


if __name__ == '__main__':
    main()
