"""
Image input/output helper fns to read img from disk, resize/crop to desired size, convert to float tensor, move to target device, save helper converts tensor->PIL img and writes to disk

Resizing / Cropping details:
- We minimize visual changes to the content while allowing square inputs by scaling the img by "size / min(width, height)", then center cropping to "size" which keeps the aspect ratio the same and keeps the center of
  the image (often used as a default for style transfer)
"""

from PIL import Image
import torch
from torchvision import transforms
from pathlib import Path


def load_image(path, size, device):
    """Load an image from "path" and return tensor on "device"

    Rescales the img so the smaller side has size "size" and then crops to a square (maintaining the center) with dimensions "size" x "size." The output tensor in the range [0,1] and has shape (1,3,size,size).
    """
    path = Path(path)
    img = Image.open(path).convert("RGB")
    w, h = img.size
    # Scale so smaller side becomes "size", crop to center
    scale = size / min(w, h)
    nh, nw = int(h * scale), int(w * scale)
    tfm = transforms.Compose([
        transforms.Resize((nh, nw)),
        transforms.CenterCrop(size),
        transforms.ToTensor()
    ])
    tensor = tfm(img).unsqueeze(0).to(device)
    return tensor


def save_image(tensor, path):
    """Save a tensor (1,3,H,W) to "path" as img file

    Clamp tensor to [0,1], move to CPU, convert to a PIL
    image and save.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x = tensor.detach().clamp(0, 1).cpu().squeeze(0)
    img = transforms.ToPILImage()(x)
    img.save(path)
