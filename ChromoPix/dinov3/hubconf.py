# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""Minimal Torch Hub entry point for the ChromoPix DINOv3 ViT-B/16 backbone."""

from dinov3.hub.backbones import dinov3_vitb16

dependencies = ["torch", "numpy"]
__all__ = ["dinov3_vitb16"]
