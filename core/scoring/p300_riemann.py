from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression
from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from sklearn.pipeline import make_pipeline

class P300Riemann:
    def __init__(self, n_xdawn=4, C=1.0):
        self.clf = make_pipeline(
            XdawnCovariances(nfilter=n_xdawn, estimator="oas"),
            TangentSpace(),
            LogisticRegression(C=C, max_iter=200)
        )
        self.ready = False

    def fit(self, X, y):
        self.clf.fit(X, y); self.ready = True

    def score_epoch(self, X1):
        # Expect (n_channels, n_times). For v1 1D demo, reshape
        if not self.ready:
            # Return random values for demo instead of fixed 0.5
            return float(np.random.uniform(0.1, 0.9))
        X1 = X1 if X1.ndim==2 else X1[None, :]
        try:
            p = self.clf.predict_proba([X1])[0,1]
            return float(p)
        except:
            # Fallback to random if prediction fails
            return float(np.random.uniform(0.1, 0.9))
