"""
VGG utilities

This module wraps a pretrained VGG16 from torchvision and contains a small
helper to extract features used for neural style transfer (Gatys et al.).

Notes / reasoning
- We use VGG16, commonly used in early neural style work. The network is used as a fixed feature extractor (i.e. weights are frozen).
- "Style" features are taken from early-to-mid convolutional layers because
  they capture multi-scale textures and patterns. It maps layer indices the same as they appear in torchvision's sequential ".features" to more human-readable names (conv1_1, conv2_1, ...) which are used in the losses.
- The content feature is usually taken from a deeper layer (conv4_2 in the
  original work) as layers this deep process larger-scale, content-related features.

Implementation details:
- The model expects normalized inputs (subtract ImageNet mean and divide by std per-channel)
- "VGGFeatures" goes along the sequential features from torchvision and records activations at the requested indices. It returns a dict containing style layer activations (with the human-readable layer names) and an additional key: "content" for the content activation
"""

import torch
import torch.nn as nn
from torchvision.models import VGG16, VGG16_Weights

# ImageNet mean/std used to normalize inputs for pretrained VGG
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)

# Layer indices (from torchvision's vgg.features) chosen for style and map to names used in the style loss
# The numbers correspond to Conv layers in the "features" module (0-based index of submodules including ReLU / Pooling layers).
# These are the same as the layer choices usually used in academic examples of Gatys-style transfer
STYLE_LAYERS = {"0": "conv1_1", "5": "conv2_1", "10": "conv3_1", "19": "conv4_1", "28": "conv5_1"}
# Content is usually taken from a deeper conv (commonly conv4_2)
CONTENT_LAYER = "21"  # conv4_2


class Normalization(nn.Module):
    """
    Normalize input by ImageNet mean/std using buffers (not parameters)

    Register mean/std as buffers so they can move to the same device as the model and can be saved/loaded with state_dict if needed.

    Forward applies (x - mean) / std for each channel
    """
    def __init__(self, mean, std, device):
        super().__init__()
        self.register_buffer("mean", mean.to(device))
        self.register_buffer("std", std.to(device))
    def forward(self, x):
        return (x - self.mean) / self.std


class VGGFeatures(nn.Module):
    """
    Extracts the desired activations from pretrained VGG16.

    Load VGG on target device, then call with an input tensor of shape (1,3,H,W). 
    The forward method returns a dict with keys matching STYLE_LAYERS values and a "content" key for the content activation

    Returned tensors have same batch dimension and spacial dimension as their corresponding convolution outputs.
    """
    def __init__(self, device):
        super().__init__()
        # Download / load pretrained VGG16 weights via torchvision.
        # We only need the convolutional feature extractor ".features"
        # Set it to eval mode since we don't want Dropout/BatchNorm updates.
        vgg = VGG16(weights=VGG16_Weights.DEFAULT).features.eval().to(device)
        for p in vgg.parameters():
            p.requires_grad_(False)
        self.vgg = vgg
        self.norm = Normalization(IMAGENET_MEAN, IMAGENET_STD, device)

    @torch.no_grad()
    def _indices(self):
        # Convert the STYLE_LAYERS keys (strings) into ints
        # collect the content layer index as well, then dedupe & sort
        # Keeps forward loop simple and fast (i.e. only compute until all features are collected).
        style_ids = sorted([int(i) for i in STYLE_LAYERS.keys()])
        content_id = int(CONTENT_LAYER)
        needed = sorted(set(style_ids + [content_id]))
        return needed, style_ids, content_id

    def forward(self, x):
        # Normalize first
        x = self.norm(x)
        feats = {}
        needed, _, content_id = self._indices()
        cur = x
        # Iterate through the sequential modules, capture activations at the desired indices.
        # Return the human-friendly names for style layers and a "content" key for the content activation
        for i, layer in enumerate(self.vgg):
            cur = layer(cur)
            if i in needed:
                name = STYLE_LAYERS.get(str(i))
                if name is not None:
                    feats[name] = cur
                if i == content_id:
                    feats["content"] = cur
            # Early exit: once we've collected all style layers + content, stop processing VGG layers (no need to continue)
            if len(feats) == len(STYLE_LAYERS) + 1:
                break
        return feats
