# `nick`

`nick` is a vanity address searcher for deployments using [Nick's method][nm].

## Quick Start

```
go install github.com/lightclient/nick@latest
nick search --initcode="0x60425000"
```

## Usage

```
NAME:
   nick search - Search for a vanity address to deploy a contract using nicks method.

USAGE:
   nick search [command [command options]]

OPTIONS:
   --threads value   number of threads to search on (default: 10)
   --score value     minimum score number to report (default: 5)
   --prefix value    desired prefix in vanity address (default: "0x0000")
   --suffix value    desired suffix in vanity address (default: "0xaaaa")
   --initcode value  desired initcode to deploy at vanity address (default: "0x")
   --gaslimit value  desired gas limit for deployment transaction (default: 250000)
   --gasprice value  desired gas price (gwei) for deployment transaction (default: 1000)
   --sig-r value     R value of the transaction signature (default: 0x0539)
   --help, -h        show help (default: false)
```

### Get transaction details for given signature

```
NAME:
   nick build - Build a json tx object and prints the deployment info.

USAGE:
   nick build [command [command options]] 

OPTIONS:
   --initcode value  desired initcode to deploy at vanity address (default: "0x")
   --gaslimit value  desired gas limit for deployment transaction (default: 250000)
   --gasprice value  desired gas price (gwei) for deployment transaction (default: 1000)
   --sig-r value     R value of the transaction signature (default: 0x0539)
   --sig-s value     S value of the transaction signature (default: 0x1337)
   --help, -h        show help (default: false)
```

## GPU acceleration

The search can be offloaded to a GPU (OpenCL or CUDA). The bottleneck in Nick's
method is the per-candidate `ecrecover`; the GPU kernel implements secp256k1
public-key recovery and both Keccak hashes on-device.

Because only the signature `s` varies during a search (`r`, `v`, and the sighash
are fixed), recovery collapses to `Q = Q_base + k·D` with a constant point
`D = r⁻¹·R`. The host precomputes `Q_base` and a comb table for `D`; candidate
`k` corresponds to signature `s = sigS + k`, and the kernel only adds `k·D`,
hashes, and matches the pattern. The winning candidate's full transaction is
reconstructed and re-verified on the CPU before printing.

### Build

```
make build        # CPU only (no GPU libraries required)
make build-gpu    # OpenCL  (needs: opencl-headers ocl-icd-opencl-dev; NVIDIA: nvidia-opencl-dev)
make build-cuda   # CUDA + OpenCL (needs the CUDA Toolkit with nvcc, plus libOpenCL)
```

### Run

```
nick list-gpus                          # OpenCL devices
nick list-gpus --gpu-backend cuda       # CUDA devices

# OpenCL on device 0
nick search --gpu --initcode 0x60425000 --suffix 0xaaaa

# CUDA across all GPUs
nick search --gpu --gpu-backend cuda --gpu-devices all --initcode 0x60425000 --suffix 0xaaaa
```

Extra `search` flags: `--gpu`, `--gpu-backend {opencl|cuda|auto}`,
`--gpu-device <i>`, `--gpu-devices <list|all>` (CUDA multi-GPU),
`--batch-size <n>`. A match requires the full `--prefix` and `--suffix` to match
(the per-nibble scoring of the CPU path is a search heuristic, not a GPU stop
condition).

### Performance & tuning

Each GPU thread processes a run of `NICK_ITERS` consecutive candidates in affine
coordinates. The pubkeys are `P_i = P0 + i·D`, and `i·D` is already in the comb
table, so each `P_i` is a single affine point addition; the per-point inversions
are all folded into one batched modular inversion (Montgomery's trick) for the
whole run. This is the profanity-style approach and keeps the points hash-ready
with no Jacobian→affine conversion per candidate. Field squaring uses a dedicated
Comba routine (~10 muls vs 16 for a generic multiply).

The main tuning knob is the run length:

- **`NICK_ITERS`** (default 64) — larger amortizes the inverse better but uses
  more per-thread local memory, lowering occupancy. It must match on both sides:
  `miner.KernelIters` (Go) and the kernel. OpenCL takes it automatically; for
  CUDA pass it to the build, e.g. `make build-cuda NICK_ITERS=32`. Benchmark
  16 / 32 / 64 / 128 on your card.
- **`--batch-size`** — threads per launch. Total candidates per launch is
  `batch-size × NICK_ITERS`; lower it if you see UI lag or kernel timeouts on a
  GPU that also drives a display.
- **CUDA build knobs** (override on the `make build-cuda` line):
  `NICK_BLOCK` (threads/block, e.g. 64/128/256), `MAXREG` (cap registers/thread
  to raise occupancy, e.g. 96/128), and `CUDA_ARCH` (use `sm_86` for an RTX 3090,
  `sm_89` for 40-series). The OpenCL local size (64) lives in `miner/gpu_opencl.go`.
  Example: `make build-cuda CUDA_ARCH=sm_86 NICK_ITERS=64 NICK_BLOCK=128 MAXREG=128`.

> Note: Nick's method is heavier per candidate than CREATE2 mining (a point
> accumulation + an amortized field inversion + two Keccak hashes), so absolute
> hashrate is lower than a pure-Keccak miner — but far faster than CPU
> `ecrecover`.

[nm]: https://yamenmerhi.medium.com/nicks-method-ethereum-keyless-execution-168a6659479c
