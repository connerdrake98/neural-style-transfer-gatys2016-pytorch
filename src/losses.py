"""
Loss utilities used by the style-transfer algorithm.

Contains:
- Gram matrix computation (used to capture texture statistics for a
  feature map)
- Content loss (MSE between generated and content feature maps)
- Style loss (MSE between Gram matrices of generated features and style features)
- Total variation (TV) loss (acts as s,oothness regularizer on the
  generated image pixels)

Mathematical notes (Gram matrix):
Given a feature tensor F of shape (N, C, H, W), we reshape at each sample to
shape (C, H * W). which we will define as F'. The Gram matrix G = F' @ F'^T is a C x C matrix where"
G[i,j] is the inner product between feature maps i and j across spacial locations, which captures relationships between feature channels that
represent texture. We normalize the Gram matrix by the
number of elements (C*H*W or C*(H*W)) so it behaves the same at different scales.
"""

import torch
import torch.nn.functional as F


def gram_matrix(feat: torch.Tensor) -> torch.Tensor:
    """Compute the Gram matrix for a feature map

    Args:
        feat: tensor with shape: (N, C, H, W)

    Returns:
        Gram matrix tensor with shape: (N, C, C) (normalized by C*H*W)
    """
    n, c, h, w = feat.size()
    # reshape to (N, C, H*W)
    f = feat.view(n, c, h * w)
    # batch matrix multiply to get (N, C, C)
    G = torch.bmm(f, f.transpose(1, 2))
    # normalize by total number of elements to keep scale consistent
    return G / (c * h * w)


def content_loss(gen_feat, content_feat):
    """Content loss as MSE between feature maps"""
    return F.mse_loss(gen_feat, content_feat)


def style_loss(gen_feats: dict, style_grams: dict, layer_weights: dict):
    """
    Compute weighted sum of style loss across layers.

    For each style layer, compute the Gram matrix of the generated features 
    and compare to the Gram matrix precomputed from the style image
    The final style loss is the weighted sum across layers.
    """
    loss = 0.0
    for lname, w in layer_weights.items():
        g = gram_matrix(gen_feats[lname])
        loss = loss + w * F.mse_loss(g, style_grams[lname])
    return loss


def tv_loss(img, tv_weight=1e-6):
    """
    Total variation loss makes the image more smooth / faithful to the content

    Implemented as average L1 difference between neighboring
    pixels (along height and width), scaled by `tv_weight`.
    """
    dh = torch.mean(torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]))
    dw = torch.mean(torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]))
    return tv_weight * (dh + dw)
