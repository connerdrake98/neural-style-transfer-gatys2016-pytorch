# Neural Style Transfer (Docker + VS Code)

Gatys et al. (2016) style transfer in PyTorch, packaged for Docker and VS Code.

## Quick Start

### CPU
```bash
# from repo root
make run-cpu
# output written to data/outputs/out_lbfgs.jpg
```

### GPU (NVIDIA)
**Requirements:** Docker with NVIDIA runtime and `nvidia-smi` available on host.
```bash
make run-gpu
```

## Build Images

### CPU
```bash
docker build -f docker/Dockerfile -t style-transfer \
  --build-arg PYTORCH_IMAGE=pytorch/pytorch:2.3.1-cpu .
```

### GPU (CUDA 12.1)
```bash
docker build -f docker/Dockerfile -t style-transfer \
  --build-arg PYTORCH_IMAGE=pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime .
```

## Run Containers Explicitly

### CPU
```bash
docker run --rm -v "$(pwd)":/app -w /app style-transfer \
  --content data/content.jpg --style data/style.jpg \
  --out data/outputs/out.jpg --size 512 --steps 300
```

### GPU
```bash
docker run --rm --gpus all -v "$(pwd)":/app -w /app style-transfer \
  --content data/content.jpg --style data/style.jpg \
  --out data/outputs/out.jpg --size 512 --steps 300
```

## VS Code (Recommended)

1. Install the **Dev Containers** extension.
2. `Ctrl/Cmd+Shift+P` → **Dev Containers: Reopen in Container**.
3. **For GPU devcontainer:** edit `.devcontainer/devcontainer.json` to use the CUDA `PYTORCH_IMAGE` and un-comment `"--gpus", "all"`.
4. Use the provided launch configurations to run/debug.

## Notes

- VGG16 weights download on first run and are cached at `/app/.cache/torch` inside the container.
- Tuning tips:
  - Start with `--content-weight 1e5 --style-weight 1.0 --tv-weight 1e-6`.
  - Increase `--style-weight` for more texture.
  - Lower `--tv-weight` for sharper but noisier results.
- For speed:
  - Reduce `--size` and `--steps`, **or**
  - Use Adam: `--optimizer adam --steps 1000 --lr 0.03`.

## License

Educational use only.
