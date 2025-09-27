# Neural Style Transfer (Docker + VS Code)

Gatys et al. (2016) style transfer in PyTorch, packaged for Docker and VS Code.

## Quick start (CPU)
```bash
# from repo root
make run-cpu
# output written to data/outputs/out_lbfgs.jpg
```

Quick start (GPU / NVIDIA)

Requirements: Docker with NVIDIA runtime; nvidia-smi visible on host.

```
make run-gpu
```

Build images explicitly
# CPU
```
docker build -f docker/Dockerfile -t style-transfer \
  --build-arg PYTORCH_IMAGE=pytorch/pytorch:2.3.1-cpu .
```

# GPU (CUDA 12.1)
```
docker build -f docker/Dockerfile -t style-transfer \
  --build-arg PYTORCH_IMAGE=pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime .
```

Run explicitly
# CPU
```
docker run --rm -v "$(pwd)":/app -w /app style-transfer \
  --content data/content.jpg --style data/style.jpg --out data/outputs/out.jpg --size 512 --steps 300
```

# GPU
```
docker run --rm --gpus all -v "$(pwd)":/app -w /app style-transfer \
  --content data/content.jpg --style data/style.jpg --out data/outputs/out.jpg --size 512 --steps 300
```

VS Code (recommended)

Install the "Dev Containers" extension.

Ctrl/Cmd+Shift+P → Dev Containers: Reopen in Container.

For GPU devcontainer: edit .devcontainer/devcontainer.json to use the CUDA PYTORCH_IMAGE and un-comment "--gpus", "all".

Use the provided launch configs to run/debug.

Notes

VGG16 weights download on first run (cached in /app/.cache/torch inside container).

Tune weights: --content-weight 1e5 --style-weight 1.0 --tv-weight 1e-6 are good starting points. Increase --style-weight for more texture; lower --tv-weight for sharper but noisier results.

For speed, reduce --size and --steps or use --optimizer adam --steps 1000 --lr 0.03.

License:
Educational use.
