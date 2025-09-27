"""
Entry point for running the neural style transfer.

Can use the Makefile / Docker entrypoint or run directly with a Python environment

Example:
    python -m src.main --content data/content.jpg --style data/style.jpg --out data/outputs/out.jpg --size 512 --steps 300
"""

import argparse
from pathlib import Path
import torch
from .image_io import load_image, save_image
from .engine import run_style_transfer
from .utils import get_device, set_seed


def parse_args():
    p = argparse.ArgumentParser("Neural Style Transfer (Gatys et al.)")
    p.add_argument("--content", required=True, help="path to content image")
    p.add_argument("--style", required=True, help="path to style image")
    p.add_argument("--out", required=True, help="output image path")
    p.add_argument("--size", type=int, default=512, help="output image size (square)")
    p.add_argument("--steps", type=int, default=300, help="number of optimization steps")
    p.add_argument("--content-weight", type=float, default=1e5, help="weight for content loss")
    p.add_argument("--style-weight", type=float, default=1.0, help="weight for style loss")
    p.add_argument("--tv-weight", type=float, default=1e-6, help="weight for total variation loss")
    p.add_argument("--optimizer", choices=["lbfgs", "adam"], default="lbfgs")
    p.add_argument("--lr", type=float, default=0.03, help="learning rate (for Adam)")
    p.add_argument("--init", choices=["content", "noise"], default="content", help="initialization for generated image")
    p.add_argument("--seed", type=int, default=1234, help="random seed")
    p.add_argument("--save-every", type=int, default=0, help="save intermediate result every N iterations (0 disables)")
    p.add_argument("--save-dir", default="data/outputs", help="directory to write intermediate outputs")
    p.add_argument("--cpu", action="store_true", help="force CPU even if GPU/MPS available")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    # Select device: use CPU if --cpu flag set, otherwise prefere these options (in order): CUDA, MPS, CPU
    device = torch.device("cpu") if args.cpu else get_device()
    content = load_image(args.content, args.size, device)
    style = load_image(args.style, args.size, device)

    # Prepare save location
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    def saver(img_tensor, step, loss):
        # Saver does not process reported loss for now, but it's included in the callback sig for future extensions
        save_image(img_tensor, save_dir / f"step_{step:05d}.jpg")

    out = run_style_transfer(
        content,
        style,
        steps=args.steps,
        content_weight=args.content_weight,
        style_weight=args.style_weight,
        tv_weight=args.tv_weight,
        init=args.init,
        optimizer=args.optimizer,
        lr=args.lr,
        device=device,
        save_every=args.save_every,
        save_callback=saver if args.save_every > 0 else None,
    )
    save_image(out, args.out)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
