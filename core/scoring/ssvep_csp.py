from __future__ import annotations
import numpy as np

def fbcca_score(epoch: np.ndarray, fs: int, targets=(31.0,32.0,33.0,34.5)):
    """
    Compute FBCCA scores for SSVEP targets.

    Returns:
        k: int - index of winning target
        max_score: float - correlation of winning target
        metadata: dict - full scores, margin, target_freq, confidence
    """
    t = np.arange(epoch.shape[-1]) / fs
    sig = epoch.mean(0) if epoch.ndim==2 else epoch
    scores = []
    for f in targets:
        ref = np.vstack([np.sin(2*np.pi*f*t), np.cos(2*np.pi*f*t)]).mean(0)
        corr = np.corrcoef(sig, ref)[0,1]
        scores.append(float(np.nan_to_num(corr)))

    k = int(np.argmax(scores))
    sorted_scores = sorted(scores, reverse=True)
    margin = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) >= 2 else 0.0

    metadata = {
        "scores": scores,
        "margin": float(margin),
        "target_freq": float(targets[k]),
        "confidence": float(sorted_scores[0]),
        "all_freqs": [float(f) for f in targets]
    }

    return k, scores[k], metadata
