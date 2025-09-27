"""
Core engine for neural style transfer. Implementation of optimization loop used to create an image that matches the content of one image and style (texture) of another. It supports two optimizers:

- LBFGS (closure-based): the original Gatys approach uses L-BFGS, often
  converges in fewer iterations (but iterations can be expensive)
- Adam: simpler gradient-based optimizer, can be easier to run and
  tune (more iterations, but simpler control)

Loss fn:
    total_loss = content_weight * content_loss + style_weight * style_loss + tv_loss

style_loss is computed as a weighted sum of MSE between Gram matrices
of generated and style features across multiple layers
"""

import torch
from torch.optim import LBFGS, Adam
from tqdm import trange
from .vgg import VGGFeatures, STYLE_LAYERS
from .losses import gram_matrix, content_loss, style_loss, tv_loss


def run_style_transfer(
    content_img, style_img, *,
    steps=300,
    content_weight=1e5,
    style_weight=1.0,
    tv_weight=1e-6,
    init="content",
    optimizer="lbfgs",
    lr=0.03,
    device=None,
    save_every=0,
    save_callback=None
):
    """Run the optimization to produce a styled image.

    Args:
        content_img: tensor (1,3,H,W) of the content image on the indicated device
        style_img: tensor (1,3,H,W) of the style image (on the indicated device)
        steps: n iterations (for Adam) or max_iter for LBFGS.
        content_weight/style_weight/tv_weight: scalar multipliers for the losses
        init: 'content' to start from content image; 'noise' to start from
            random noise.
        optimizer: 'lbfgs' or 'adam'.
        lr: learning rate for Adam.
        device: torch.device (default: content_img.device).
        save_every: how often (in iterations) to call save_callback.
        save_callback: function(img_tensor, step, loss) used to save intermediary results

    Returns:
        The final generated image (detached tensor), has shape (1,3,H,W).
    """
    device = content_img.device if device is None else device
    # Build the feature network once, on the correct device
    net = VGGFeatures(device)

    # Precompute content features and style Gram matrices using the fixed VGG extractor
    # We only need the per-layer style Gram matrices from the style image
    # We keep the content activation for the content img
    with torch.no_grad():
        c_feats = net(content_img)["content"]
        s_feats = net(style_img)
        # style_grams: dict[layer_name] -> Gram matrix tensor
        style_grams = {k: gram_matrix(v) for k, v in s_feats.items() if k in STYLE_LAYERS.values()}

    # Simple uniform weighting across selected style layers, can configure per-layer for more control if desired
    layer_weights = {k: 1.0 / len(STYLE_LAYERS) for k in STYLE_LAYERS.values()}

    # Create initial "generated img"
    if init == "content":
        img = content_img.clone()
    elif init == "noise":
        img = torch.randn_like(content_img)
    else:
        raise ValueError("init must be 'content' or 'noise'")
    img.requires_grad_(True)

    params = [img]

    # LBFGS needs closure for re-evaluation of loss and gradients. 
    # In PyTorch, optimizer calls it multiple times per step to do line searches internally
    # Closure lets computation of features and weighted loses, backpropagation
    # also lets us integrate the save callback as well 

    if optimizer.lower() == "lbfgs":
        opt = LBFGS(params, lr=1.0, max_iter=steps, history_size=50, line_search_fn="strong_wolfe")
        iters = 0

        def closure():
            nonlocal iters
            opt.zero_grad()
            feats = net(img)
            Lc = content_weight * content_loss(feats["content"], c_feats)
            Ls = style_weight * style_loss(feats, style_grams, layer_weights)
            Lt = tv_loss(img, tv_weight)
            loss = Lc + Ls + Lt
            # Backpropagate with respect to generated img
            loss.backward()
            iters += 1
            # Save intermediate imgs if indicated. 
            # For LBFGS, might happen multiple times per outer iteration due to line searches (track iterations based on n calls)
            if save_every and save_callback and iters % save_every == 0:
                save_callback(img.detach(), iters, float(loss.detach()))
            return loss

        opt.step(closure)
        # Return detached img tensor, no grad
        return img.detach()

    # Adam loop (update once per iteration)
    # easier to checkpoint but usually needs more iterations than LBFGS.
    opt = Adam(params, lr=lr)
    pbar = trange(steps, desc="optim")
    for t in pbar:
        opt.zero_grad()
        feats = net(img)
        Lc = content_weight * content_loss(feats["content"], c_feats)
        Ls = style_weight * style_loss(feats, style_grams, layer_weights)
        Lt = tv_loss(img, tv_weight)
        loss = Lc + Ls + Lt
        loss.backward()
        opt.step()
        if save_every and save_callback and ((t + 1) % save_every == 0):
            save_callback(img.detach(), t + 1, float(loss.detach()))
        pbar.set_postfix_str(f"loss={float(loss.detach()):.2e}")
    return img.detach()
