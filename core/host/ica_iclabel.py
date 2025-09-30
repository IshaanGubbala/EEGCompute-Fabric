from __future__ import annotations
import mne
from mne.preprocessing import ICA
from mne_icalabel import label_components

def run_ica_and_mask(raw: mne.io.BaseRaw, n_components=20, threshold=0.9, random_state=97):
    ica = ICA(n_components=n_components, random_state=random_state, max_iter="auto")
    ica.fit(raw.copy().load_data())
    labels = label_components(raw, ica, method="iclabel")
    comp_labels, proba = labels["labels"], labels["y_pred_proba"]
    bads = []
    for i, (lab, probs) in enumerate(zip(comp_labels, proba)):
        if lab in ("eye","muscle","heart") and max(probs) >= threshold:
            bads.append(i)
    raw_clean = ica.apply(raw.copy(), exclude=bads)
    return raw_clean, {"excluded": bads, "labels": comp_labels, "proba": proba.tolist()}
