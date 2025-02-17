# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .efficient_sam import build_efficient_sam
from .efficient_sam import build_efficient_sam_w_ad

def build_efficient_sam_vitt(checkpoint=None):
    return build_efficient_sam(
        encoder_patch_embed_dim=192,
        encoder_num_heads=3,
        checkpoint=checkpoint,
    ).eval()


def build_efficient_sam_vits(checkpoint=None):
    return build_efficient_sam(
        encoder_patch_embed_dim=384,
        encoder_num_heads=6,
        checkpoint=checkpoint,
    ).eval()

def build_efficient_sam_vitt_w_ad(checkpoint=None):
    return build_efficient_sam_w_ad(
        encoder_patch_embed_dim=192,
        encoder_num_heads=3,
        checkpoint=checkpoint,
    ).eval()

def build_efficient_sam_vits_w_ad(checkpoint=None):
    return build_efficient_sam_w_ad(
        encoder_patch_embed_dim=384,
        encoder_num_heads=6,
        checkpoint=checkpoint,
    ).eval()

eff_sam_model_registry = {
    "eff_vit_t": build_efficient_sam_vitt,
    "eff_vit_s": build_efficient_sam_vits,

    "eff_vit_t_w_ad": build_efficient_sam_vitt_w_ad,
    "eff_vit_s_w_ad": build_efficient_sam_vits_w_ad,
}