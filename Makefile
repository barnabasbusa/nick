.PHONY: build build-gpu build-cuda-lib build-cuda clean test list-gpus help

BINARY_NAME=nick
GOBUILD=go build

# CUDA configuration
NVCC := $(shell which nvcc 2>/dev/null)
CUDA_PATH ?= /usr/local/cuda
# Override with: make build-cuda CUDA_ARCH=sm_XX
#   sm_70 Volta | sm_75 Turing | sm_80 Ampere | sm_89 Ada | sm_90 Hopper
CUDA_ARCH ?= sm_80

# Candidates per GPU thread (batch-inversion run length). MUST match
# miner.KernelIters in the Go code.
NICK_ITERS ?= 64

# CUDA tuning knobs (override on the make command line):
#   NICK_BLOCK - threads per block (e.g. 64/128/256)
#   MAXREG     - cap registers/thread to raise occupancy (e.g. 64/96/128); empty = off
NICK_BLOCK ?= 256
MAXREG ?=

all: build

## build: CPU-only build (no GPU libraries required)
build:
	@echo "Building CPU-only binary..."
	CGO_ENABLED=1 $(GOBUILD) -tags nocl -o $(BINARY_NAME) -v

## build-gpu: Build with OpenCL GPU support (Linux/macOS)
build-gpu:
	@echo "Building with OpenCL support..."
	CGO_ENABLED=1 $(GOBUILD) -o $(BINARY_NAME) -v

## build-cuda-lib: Compile the CUDA kernel library (required before build-cuda)
build-cuda-lib:
ifndef NVCC
	$(error "nvcc not found. Install the CUDA Toolkit and ensure nvcc is in PATH")
endif
	@echo "Compiling CUDA kernel library (arch=$(CUDA_ARCH))..."
	cd miner/kernel && $(NVCC) -c -o nick_cuda.o cuda_launcher.cu \
		-arch=$(CUDA_ARCH) \
		-DNICK_ITERS=$(NICK_ITERS) \
		-DNICK_BLOCK=$(NICK_BLOCK) \
		$(if $(MAXREG),-maxrregcount=$(MAXREG)) \
		-O3 \
		--use_fast_math \
		-Xcompiler -O3,-fPIC
	cd miner/kernel && ar rcs libnick_cuda.a nick_cuda.o
	@echo "CUDA library built: miner/kernel/libnick_cuda.a"

## build-cuda: Build with CUDA + OpenCL support (Linux, NVIDIA). Needs libOpenCL too.
build-cuda: build-cuda-lib
	@echo "Building with CUDA support..."
	CGO_ENABLED=1 $(GOBUILD) -tags cuda -o $(BINARY_NAME) -v

## test: Run the math/self-check tests (no GPU needed)
test:
	go test -tags nocl ./...

## list-gpus: Build with OpenCL and list devices
list-gpus: build-gpu
	./$(BINARY_NAME) search --list-gpus

## clean: Remove build artifacts
clean:
	go clean
	rm -f $(BINARY_NAME)
	rm -f miner/kernel/*.o miner/kernel/*.a

## help: Show this help
help:
	@echo "Usage: make [target]"
	@echo ""
	@grep -E '^## ' Makefile | sed 's/## /  /'
	@echo ""
	@echo "GPU prerequisites:"
	@echo "  OpenCL (Linux): sudo apt install opencl-headers ocl-icd-opencl-dev"
	@echo "                  NVIDIA: also nvidia-opencl-dev / nvidia drivers"
	@echo "  CUDA   (Linux): NVIDIA CUDA Toolkit 11.0+ with nvcc in PATH"
	@echo ""
	@echo "Examples:"
	@echo "  make build-gpu && ./nick search --gpu --initcode 0x6000... --suffix 0xaaaa"
	@echo "  make build-cuda && ./nick search --gpu --gpu-backend cuda --gpu-devices all --initcode 0x..."
