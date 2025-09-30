from __future__ import annotations
import mne

def bandpass_notch(raw: mne.io.BaseRaw, l_freq=1.0, h_freq=40.0, notch=60.0):
    raw = raw.copy().filter(l_freq=l_freq, h_freq=h_freq, fir_design="firwin", phase="zero")
    raw = raw.notch_filter(freqs=[notch])
    return raw

def reref(raw: mne.io.BaseRaw, ref="average"):
    raw = raw.copy().set_eeg_reference(ref)
    return raw
