import time
class Timer:
    def __init__(self): self.t0=None
    def start(self): self.t0=time.perf_counter()
    def stop(self): return (time.perf_counter()-self.t0) if self.t0 else None
