from __future__ import annotations
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

class ErrPDecoder:
    def __init__(self):
        self.clf = LDA(); self.ready = False
    def fit(self, X, y):
        self.clf.fit(X, y); self.ready = True
    def score(self, x1):
        if not self.ready: return 0.5
        return float(self.clf.predict_proba([x1])[0,1])
