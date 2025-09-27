IMAGE=style-transfer
# CPU default, set CUDA tag to use GPU
PYTORCH_IMAGE?=pytorch/pytorch:2.3.1-cpu
RUN_BASE=docker run --rm -v $(PWD):/app -w /app

build:
	docker build -f docker/Dockerfile --build-arg PYTORCH_IMAGE=$(PYTORCH_IMAGE) -t $(IMAGE) .

run-cpu: build
	$(RUN_BASE) $(IMAGE) --content data/content.jpg --style data/style.jpg --out data/outputs/out_lbfgs.jpg --size 512 --steps 300 --optimizer lbfgs --init content

run-gpu:
	$(MAKE) build PYTORCH_IMAGE=pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime
	$(RUN_BASE) --gpus all $(IMAGE) --content data/content.jpg --style data/style.jpg --out data/outputs/out_lbfgs.jpg --size 512 --steps 300 --optimizer lbfgs --init content
