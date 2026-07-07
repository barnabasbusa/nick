# `nick` on macOS — Apple Silicon (Metal) GPU miner

This is a native **Metal** backend for `nick`, the Nick's-method vanity-address
searcher. It runs the secp256k1 public-key recovery and both Keccak hashes on
the Apple GPU, so any Mac with Apple Silicon (M1 … M5 and the Pro/Max/Ultra
variants) can mine — no CUDA, no NVIDIA, no OpenCL.

Measured: **~220 MH/s on an M5 Max (40-core GPU).** Scales roughly with GPU core
count, so expect proportionally less on an M1/M2/M3 base chip and more on an
Ultra.

## Requirements

- Apple Silicon Mac (Intel Macs have no Metal-capable Apple GPU for this).
- Go 1.21+ (`brew install go`).
- **Xcode is *not* required.** Only the Command Line Tools are needed:
  ```
  xcode-select --install
  ```
  The kernel is compiled at runtime by the Metal framework that ships with
  macOS, so there is no offline `metal`/`metallib` step and no `.metallib` to
  carry around.

## Build

```
make build-metal
```

That produces a self-contained `./nick` binary (the shader is embedded). Under
the hood it is just:

```
CGO_ENABLED=1 go build -tags "metal nocl" -o nick
```

(`nocl` disables the OpenCL path so the Metal build is clean and standalone.)

## List your GPU

```
./nick list-gpus --gpu-backend metal
```
```
Metal devices (1):
  [0] Apple M5 Max (Apple)
```

## Run

Use `--gpu --gpu-backend metal`. Everything else is identical to the CUDA/OpenCL
paths. The full example from the eip-8282 deployment:

```
go install github.com/fjl/geas/cmd/geas@latest

./nick search --gpu --gpu-backend metal \
  --initcode "0x$(geas ./eip8282/src/deposits/ctor.eas)" \
  --prefix "0x0000" --suffix "0x008282" --sig-r 0x8282
```

where `./eip8282` is `assets/eip-8282` from the EIP PR.

A quick smoke test that needs no external assets (any init code works — see
"How it works" for why the address is independent of the init code):

```
./nick search --gpu --gpu-backend metal --initcode 0x60425000 --suffix 0xaaaa
```

Output on a hit:

```
GPU mining on Apple M5 Max (batch size 67108864)
Target: prefix=0x0000 suffix=0x008282
Searched 2952790016 candidates, 220.26 MH/s
Found! (score 40, k=12345)
Sender:  0x…
Address: 0x0000…8282
Tx: { …signed legacy tx ready to broadcast… }
```

Every GPU hit is **re-derived on the CPU with go-ethereum's `ecrecover`** before
it is printed; a mismatch prints a loud `WARNING` instead of a result, so you can
trust anything reported as `Found!`.

## Tuning

| Knob | How | Effect |
|------|-----|--------|
| `--batch-size N` | flag | GPU threads per launch (default `1048576`). Total candidates per launch = `batch-size × 256`. Lower it if the UI lags while mining. |
| Threadgroup size | `NICK_TG=<32/64/128/256>` env | Occupancy tuning only; never changes results. Default `64`. On an M5 Max the difference is within noise — benchmark on yours. |
| Run length | `miner.MetalKernelIters` (build-time, default **256**) | Candidates each thread processes under one batched field inversion. Higher amortizes the inverse better (256 ≈ 35 % faster than 64 here) at the cost of per-thread memory. Capped at 256 by the comb table. Lower it (e.g. 128) if a base-model chip runs short on GPU memory. |

## How it works

Nick's method recovers a deployer EOA from a fixed `(r, v)` and a varying
signature `s`, then the contract lands at `keccak256(rlp([deployer, 0]))[12:]`.
Because only `s` changes, recovery collapses to `Q = Q_base + k·D` with a
constant `D = r⁻¹·R`: the host precomputes `Q_base` and a comb table for `D`
(this part is shared with the CUDA/OpenCL backends and validated against
go-ethereum), and the Metal kernel only adds `k·D`, hashes twice, and matches the
pattern. Each thread walks a run of 256 consecutive candidates in affine
coordinates, folding all the per-point inversions into one batched modular
inverse (Montgomery's trick).

Note the contract address depends only on the recovered **deployer + nonce 0**,
not on the init code — so hashrate and the search itself are identical whatever
`--initcode` you pass. You still need the *real* init code on the winning run so
the printed transaction actually deploys your contract (the init code is part of
the signed tx and fixes the sighash).

### Metal-specific implementation notes

- Apple GPUs have no 64-bit `mul_hi` intrinsic, so the field arithmetic uses a
  software `mulhi64` built from four 32-bit products.
- The field reduction (`fe_reduce`) and the other heavy field/point routines are
  marked `noinline`. This is **load-bearing**: Apple's Metal compiler miscompiles
  the point routines when they are fully inlined (under the 64-bit register
  pressure it spills a multiply operand and silently corrupts `Y3`). Keeping them
  out-of-line bounds the live register set and makes the GPU math match the CPU
  reference bit-for-bit. It is also *faster* here (better occupancy).

## Troubleshooting

- **`metal support not built`** — you built without the tag; use `make build-metal`.
- **`no Metal devices found`** — not an Apple Silicon GPU, or running under a VM
  without GPU passthrough.
- **A printed `WARNING` about a mismatch** — do not use that result; please open
  an issue with your chip model.
