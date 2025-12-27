import time
from collections import deque


class Logger:
    def __init__(self, window_size=100):
        self.start = time.time()
        self.window_size = window_size
        self.losses = deque(maxlen=window_size)
        self.total_steps = 0
        self.total_loss = 0.0

    def log(self, step, loss, lr=None):
        # Update running statistics
        self.losses.append(loss)
        self.total_steps += 1
        self.total_loss += loss
        
        elapsed = time.time() - self.start
        avg_loss = sum(self.losses) / len(self.losses)
        total_avg_loss = self.total_loss / self.total_steps
        
        # Format learning rate if provided
        lr_str = f" lr={lr:.2e}" if lr is not None else ""
        
        # Enhanced logging with better formatting
        print(
            f"[{elapsed:7.1f}s] step={step:6d} "
            f"loss={loss:.4f} avg_loss={avg_loss:.4f} total_avg={total_avg_loss:.4f}{lr_str}"
        )