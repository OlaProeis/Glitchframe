"""Vendored RIFE IFNet + warplayer for frame interpolation.

Network architecture and warplayer are derived from Practical-RIFE
(`https://github.com/hzwer/Practical-RIFE`, MIT) and the matching
``train_log`` published alongside the RIFE v4.26 weights on Hugging Face.

Glitchframe only ships the inference-time modules; training utilities are omitted.
"""

from pipeline.rife_vendor.ifnet_hdv3 import IFNet  # noqa: F401

__all__ = ["IFNet"]
