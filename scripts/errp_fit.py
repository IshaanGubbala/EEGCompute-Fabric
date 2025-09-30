"""
ErrP Calibration Script

Guided calibration routine for Error-related Potential (ErrP) detection:
1. Presents correct/incorrect feedback to user
2. Collects EEG epochs after feedback
3. Trains ErrPDecoder classifier
4. Evaluates AUROC, AUPRC, threshold at ~10% FPR
5. Saves model card to models/

Usage:
    python scripts/errp_fit.py --duration 180 --subject 001

Model card format:
    models/errp_sub-{subject}_{timestamp}.json
        {
            "model_type": "errp_lda",
            "subject": "001",
            "task": "feedback",
            "n_epochs": 200,
            "auroc": 0.75,
            "auprc": 0.70,
            "threshold_fpr_0.1": 0.65,
            "params": {"solver": "lsqr"},
            "trained_at": "2025-09-30T12:34:56"
        }
"""
from __future__ import annotations
import os, sys, argparse, json
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

from core.scoring.errp_decoder import ErrPDecoder


def simulate_calibration_data(n_errors: int = 50, n_correct: int = 150, fs: int = 512, n_channels: int = 8):
    """
    Simulate ErrP calibration data for demo.

    In production, this would be replaced by real LSL acquisition.

    Args:
        n_errors: Number of error epochs (feedback indicating mistake)
        n_correct: Number of correct epochs (feedback indicating success)
        fs: Sampling frequency
        n_channels: Number of EEG channels

    Returns:
        X: (n_epochs, n_channels, n_samples) array
        y: (n_epochs,) binary labels (1 = error, 0 = correct)
    """
    epoch_duration = 0.8  # seconds (200ms pre + 600ms post feedback)
    n_samples = int(epoch_duration * fs)

    n_total = n_errors + n_correct
    X = np.random.randn(n_total, n_channels, n_samples)
    y = np.concatenate([np.ones(n_errors), np.zeros(n_correct)])

    # Add fake ErrP signal to error trials (250-450ms post-feedback)
    errp_start = int(0.45 * fs)  # 250ms after start (200ms pre-feedback)
    errp_end = int(0.65 * fs)
    for i in range(n_errors):
        # Inject negative deflection on frontocentral channels
        X[i, 1:4, errp_start:errp_end] -= np.random.uniform(1.0, 2.5)

    # Shuffle
    idx = np.random.permutation(n_total)
    X, y = X[idx], y[idx]

    return X, y


def train_and_evaluate(X, y):
    """
    Train ErrP classifier and evaluate performance.

    Args:
        X: (n_epochs, n_channels, n_samples) array
        y: (n_epochs,) binary labels

    Returns:
        model: Trained ErrPDecoder classifier
        metrics: Dict with AUROC, AUPRC, thresholds
    """
    print(f"\n[ErrP Calibration] Training classifier...")
    print(f"  n_epochs: {len(y)} (errors: {y.sum()}, correct: {(1-y).sum()})")

    # Flatten epochs for LDA (expects 2D: n_samples x n_features)
    X_flat = X.reshape(len(X), -1)

    # Train model
    clf = ErrPDecoder()
    clf.fit(X_flat, y)

    # Evaluate with cross-validation
    print(f"\n[ErrP Calibration] Cross-validation (5-fold)...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf.clf, X_flat, y, cv=cv, scoring='roc_auc')
    print(f"  CV AUROC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # Get predictions on full dataset (for threshold tuning)
    y_pred_proba = clf.clf.predict_proba(X_flat)[:, 1]

    # Compute metrics
    auroc = roc_auc_score(y, y_pred_proba)
    auprc = average_precision_score(y, y_pred_proba)

    # Find threshold at ~10% FPR (high specificity)
    fpr, tpr, thresholds = roc_curve(y, y_pred_proba)
    idx_fpr10 = np.argmin(np.abs(fpr - 0.1))
    thresh_fpr10 = thresholds[idx_fpr10]
    tpr_at_fpr10 = tpr[idx_fpr10]

    print(f"\n[ErrP Calibration] Performance:")
    print(f"  AUROC: {auroc:.3f}")
    print(f"  AUPRC: {auprc:.3f}")
    print(f"  Threshold (FPR=10%): {thresh_fpr10:.3f} (TPR={tpr_at_fpr10:.2%})")

    metrics = {
        "auroc": float(auroc),
        "auprc": float(auprc),
        "cv_auroc_mean": float(cv_scores.mean()),
        "cv_auroc_std": float(cv_scores.std()),
        "threshold_fpr_0.1": float(thresh_fpr10),
        "tpr_at_fpr_0.1": float(tpr_at_fpr10),
    }

    return clf, metrics


def save_model_card(clf, metrics, subject: str, output_dir: str = "models"):
    """
    Save trained model and metadata to disk.

    Args:
        clf: Trained ErrPDecoder classifier
        metrics: Performance metrics dict
        subject: Subject ID
        output_dir: Output directory for models
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"errp_sub-{subject}_{timestamp}"

    # Save model card (JSON metadata)
    card = {
        "model_type": "errp_lda",
        "subject": subject,
        "task": "feedback",
        "n_epochs": len(metrics),
        "params": {
            "solver": "svd"
        },
        "trained_at": datetime.now().isoformat(),
        **metrics
    }

    card_file = output_path / f"{base_name}.json"
    with open(card_file, 'w') as f:
        json.dump(card, f, indent=2)
    print(f"\n[ErrP Calibration] Saved model card: {card_file}")

    # Save model pickle
    model_file = output_path / f"{base_name}.pkl"
    with open(model_file, 'wb') as f:
        pickle.dump(clf, f)
    print(f"[ErrP Calibration] Saved model: {model_file}")

    return card_file, model_file


def main():
    parser = argparse.ArgumentParser(description="ErrP Calibration Script")
    parser.add_argument("--duration", type=int, default=180,
                        help="Calibration duration (seconds)")
    parser.add_argument("--subject", type=str, default="001",
                        help="Subject ID")
    parser.add_argument("--n-errors", type=int, default=50,
                        help="Number of error epochs (simulated)")
    parser.add_argument("--n-correct", type=int, default=150,
                        help="Number of correct epochs (simulated)")
    parser.add_argument("--output-dir", type=str, default="models",
                        help="Output directory for models")

    args = parser.parse_args()

    print("=" * 60)
    print("ErrP Calibration Script")
    print("=" * 60)
    print(f"  Subject: {args.subject}")
    print(f"  Duration: {args.duration}s")
    print(f"  Error epochs: {args.n_errors}")
    print(f"  Correct epochs: {args.n_correct}")
    print()

    # 1. Simulate calibration data
    print("[ErrP Calibration] Generating calibration data...")
    X, y = simulate_calibration_data(
        n_errors=args.n_errors,
        n_correct=args.n_correct
    )

    # 2. Train and evaluate
    clf, metrics = train_and_evaluate(X, y)

    # 3. Save model card
    card_file, model_file = save_model_card(
        clf, metrics, args.subject, args.output_dir
    )

    # 4. Check minimum AUROC threshold
    if metrics['auroc'] >= 0.70:
        print(f"\n✓ Calibration PASSED (AUROC ≥ 0.70)")
    else:
        print(f"\n⚠ Calibration WARNING (AUROC < 0.70)")
        print(f"  Consider collecting more data or improving signal quality")

    print("\n" + "=" * 60)
    print("ErrP Calibration Complete")
    print("=" * 60)
    print(f"  Model card: {card_file}")
    print(f"  Model file: {model_file}")
    print()


if __name__ == "__main__":
    main()
