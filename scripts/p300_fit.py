"""
P300 Calibration Script

Guided calibration routine for P300 detection:
1. Runs RSVP stimulus sequence (e.g., 3-5 minutes)
2. Collects EEG epochs around target/non-target events
3. Trains P300Riemann classifier
4. Evaluates AUROC, AUPRC, threshold
5. Saves model card to models/

Usage:
    python scripts/p300_fit.py --duration 180 --subject 001

Model card format:
    models/p300_sub-{subject}_{timestamp}.json
        {
            "model_type": "p300_riemann",
            "subject": "001",
            "task": "rsvp",
            "n_epochs": 200,
            "auroc": 0.85,
            "auprc": 0.82,
            "threshold_0.5": 0.45,
            "threshold_0.8": 0.62,
            "params": {"n_xdawn": 4, "C": 1.0},
            "trained_at": "2025-09-30T12:34:56"
        }
"""
from __future__ import annotations
import os, sys, time, argparse, json
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
import pickle

# Ensure project root on sys.path
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.scoring.p300_riemann import P300Riemann


def simulate_calibration_data(n_targets: int = 40, n_nontargets: int = 160, fs: int = 512, n_channels: int = 8):
    """
    Simulate P300 calibration data for demo.

    In production, this would be replaced by real LSL acquisition.

    Args:
        n_targets: Number of target epochs
        n_nontargets: Number of non-target epochs
        fs: Sampling frequency
        n_channels: Number of EEG channels

    Returns:
        X: (n_epochs, n_channels, n_samples) array
        y: (n_epochs,) binary labels (1 = target, 0 = non-target)
    """
    epoch_duration = 1.0  # seconds (0.2s pre + 0.8s post)
    n_samples = int(epoch_duration * fs)

    n_total = n_targets + n_nontargets
    X = np.random.randn(n_total, n_channels, n_samples)
    y = np.concatenate([np.ones(n_targets), np.zeros(n_nontargets)])

    # Add fake P300 signal to targets (300-500ms post-stimulus)
    p300_start = int(0.5 * fs)  # 300ms after start (0.2s pre-stim)
    p300_end = int(0.7 * fs)
    for i in range(n_targets):
        # Inject positive deflection on central channels
        X[i, 2:5, p300_start:p300_end] += np.random.uniform(1.5, 3.0)

    # Shuffle
    idx = np.random.permutation(n_total)
    X, y = X[idx], y[idx]

    return X, y


def train_and_evaluate(X, y, n_xdawn: int = 4, C: float = 1.0):
    """
    Train P300 classifier and evaluate performance.

    Args:
        X: (n_epochs, n_channels, n_samples) array
        y: (n_epochs,) binary labels
        n_xdawn: Number of Xdawn spatial filters
        C: Logistic regression regularization

    Returns:
        model: Trained P300Riemann classifier
        metrics: Dict with AUROC, AUPRC, thresholds
    """
    print(f"\n[P300 Calibration] Training classifier...")
    print(f"  n_epochs: {len(y)} (targets: {y.sum()}, non-targets: {(1-y).sum()})")
    print(f"  n_xdawn: {n_xdawn}, C: {C}")

    # Train model
    clf = P300Riemann(n_xdawn=n_xdawn, C=C)
    clf.fit(X, y)

    # Evaluate with cross-validation
    print(f"\n[P300 Calibration] Cross-validation (5-fold)...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf.clf, X, y, cv=cv, scoring='roc_auc')
    print(f"  CV AUROC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # Get predictions on full dataset (for threshold tuning)
    y_pred_proba = np.array([clf.clf.predict_proba([x])[0, 1] for x in X])

    # Compute metrics
    auroc = roc_auc_score(y, y_pred_proba)
    auprc = average_precision_score(y, y_pred_proba)

    # Find thresholds for different operating points
    fpr, tpr, thresholds = roc_curve(y, y_pred_proba)

    # Threshold at 50% sensitivity
    idx_50 = np.argmin(np.abs(tpr - 0.5))
    thresh_50 = thresholds[idx_50]

    # Threshold at 80% sensitivity
    idx_80 = np.argmin(np.abs(tpr - 0.8))
    thresh_80 = thresholds[idx_80]

    print(f"\n[P300 Calibration] Performance:")
    print(f"  AUROC: {auroc:.3f}")
    print(f"  AUPRC: {auprc:.3f}")
    print(f"  Threshold (50% sens): {thresh_50:.3f}")
    print(f"  Threshold (80% sens): {thresh_80:.3f}")

    metrics = {
        "auroc": float(auroc),
        "auprc": float(auprc),
        "cv_auroc_mean": float(cv_scores.mean()),
        "cv_auroc_std": float(cv_scores.std()),
        "threshold_0.5": float(thresh_50),
        "threshold_0.8": float(thresh_80),
    }

    return clf, metrics


def save_model_card(clf, metrics, subject: str, n_xdawn: int, C: float, output_dir: str = "models"):
    """
    Save trained model and metadata to disk.

    Args:
        clf: Trained P300Riemann classifier
        metrics: Performance metrics dict
        subject: Subject ID
        n_xdawn: Model hyperparameter
        C: Model hyperparameter
        output_dir: Output directory for models
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"p300_sub-{subject}_{timestamp}"

    # Save model card (JSON metadata)
    card = {
        "model_type": "p300_riemann",
        "subject": subject,
        "task": "rsvp",
        "n_epochs": len(metrics),
        "params": {
            "n_xdawn": n_xdawn,
            "C": C
        },
        "trained_at": datetime.now().isoformat(),
        **metrics
    }

    card_file = output_path / f"{base_name}.json"
    with open(card_file, 'w') as f:
        json.dump(card, f, indent=2)
    print(f"\n[P300 Calibration] Saved model card: {card_file}")

    # Save model pickle
    model_file = output_path / f"{base_name}.pkl"
    with open(model_file, 'wb') as f:
        pickle.dump(clf, f)
    print(f"[P300 Calibration] Saved model: {model_file}")

    return card_file, model_file


def main():
    parser = argparse.ArgumentParser(description="P300 Calibration Script")
    parser.add_argument("--duration", type=int, default=180,
                        help="Calibration duration (seconds)")
    parser.add_argument("--subject", type=str, default="001",
                        help="Subject ID")
    parser.add_argument("--n-targets", type=int, default=40,
                        help="Number of target epochs (simulated)")
    parser.add_argument("--n-nontargets", type=int, default=160,
                        help="Number of non-target epochs (simulated)")
    parser.add_argument("--n-xdawn", type=int, default=4,
                        help="Number of Xdawn spatial filters")
    parser.add_argument("--C", type=float, default=1.0,
                        help="Logistic regression regularization")
    parser.add_argument("--output-dir", type=str, default="models",
                        help="Output directory for models")

    args = parser.parse_args()

    print("=" * 60)
    print("P300 Calibration Script")
    print("=" * 60)
    print(f"  Subject: {args.subject}")
    print(f"  Duration: {args.duration}s")
    print(f"  Target epochs: {args.n_targets}")
    print(f"  Non-target epochs: {args.n_nontargets}")
    print()

    # 1. Simulate calibration data
    print("[P300 Calibration] Generating calibration data...")
    X, y = simulate_calibration_data(
        n_targets=args.n_targets,
        n_nontargets=args.n_nontargets
    )

    # 2. Train and evaluate
    clf, metrics = train_and_evaluate(X, y, n_xdawn=args.n_xdawn, C=args.C)

    # 3. Save model card
    card_file, model_file = save_model_card(
        clf, metrics, args.subject, args.n_xdawn, args.C, args.output_dir
    )

    # 4. Check minimum AUROC threshold
    if metrics['auroc'] >= 0.80:
        print(f"\n✓ Calibration PASSED (AUROC ≥ 0.80)")
    else:
        print(f"\n⚠ Calibration WARNING (AUROC < 0.80)")
        print(f"  Consider collecting more data or adjusting parameters")

    print("\n" + "=" * 60)
    print("P300 Calibration Complete")
    print("=" * 60)
    print(f"  Model card: {card_file}")
    print(f"  Model file: {model_file}")
    print()


if __name__ == "__main__":
    main()
