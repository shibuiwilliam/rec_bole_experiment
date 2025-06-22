"""PyTorch compatibility utilities."""

import torch

# PyTorch 2.6 compatibility fix
original_torch_load = torch.load


def patched_torch_load(*args, **kwargs):
    """
    Patched version of torch.load that ensures compatibility with PyTorch 2.6+.

    Automatically adds weights_only=False if not specified to maintain
    backward compatibility with older code.
    """
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return original_torch_load(*args, **kwargs)


def apply_torch_compatibility_patch():
    """Apply PyTorch compatibility patches."""
    torch.load = patched_torch_load
