from typing import Dict
class ClosedLoop:
    def __init__(self, alpha=1.0, decay=0.95):
        self.alpha = alpha; self.decay = decay; self.mem: Dict[str, float] = {}
    def update(self, item_id: str, p: float):
        self.mem[item_id] = self.decay*self.mem.get(item_id, 0.0) + self.alpha*p
