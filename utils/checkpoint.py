import torch
import os


def save_checkpoint(model, optimizer, step, path, scheduler=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
    }
    if scheduler is not None:
        checkpoint["scheduler"] = scheduler.state_dict()
    torch.save(checkpoint, path)
