import numpy as np
class ErrPBandit:
    def __init__(self, n_actions: int, lr=0.1, penalty_scale=1.0):
        self.Q = np.zeros(n_actions); self.lr = lr; self.penalty_scale = penalty_scale
    def step(self, a: int, reward: float, errp_prob: float):
        r = reward - self.penalty_scale*errp_prob
        self.Q[a] += self.lr*(r - self.Q[a]); return r
    def act(self, eps=0.1):
        return np.random.randint(len(self.Q)) if np.random.rand()<eps else int(np.argmax(self.Q))
