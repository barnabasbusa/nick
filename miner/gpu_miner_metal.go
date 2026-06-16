//go:build metal

package miner

/*
#cgo CFLAGS: -I${SRCDIR}/kernel
#cgo CXXFLAGS: -std=c++17 -I${SRCDIR}/kernel -xobjective-c++ -fobjc-arc -O3
#cgo LDFLAGS: -framework Metal -framework Foundation
#include <stdlib.h>
#include "metal_launcher.h"
*/
import "C"

import (
	_ "embed"
	"fmt"
	"os"
	"strconv"
	"time"
	"unsafe"
)

// metalSrc is the Metal shader, compiled at runtime by the launcher. Embedding
// it means the binary is self-contained (no precompiled .metallib needed).
//
//go:embed kernel/nick.metal
var metalSrc string

// MetalMiner runs the Nick's-method search on an Apple Silicon GPU via Metal.
type MetalMiner struct {
	ptr       *C.MetalMiner
	batchSize int
	name      string
	uploaded  bool
}

// ListMetalGPUs enumerates the available Metal devices.
func ListMetalGPUs() ([]GPUInfo, error) {
	const cap = 16
	var names [cap][128]C.char
	n := int(C.metal_list_devices(&names[0], C.int(cap)))
	if n <= 0 {
		return nil, fmt.Errorf("no Metal devices found")
	}
	if n > cap {
		n = cap
	}
	out := make([]GPUInfo, n)
	for i := 0; i < n; i++ {
		out[i] = GPUInfo{Index: i, Name: C.GoString(&names[i][0]), Vendor: "Apple"}
	}
	return out, nil
}

// NewMetalMiner compiles the kernel on the given device and prepares device
// buffers. batchSize is the number of GPU threads per launch; each thread
// processes MetalKernelIters consecutive candidates.
func NewMetalMiner(device, batchSize int) (*MetalMiner, error) {
	if batchSize <= 0 {
		batchSize = 1 << 20
	}

	// Inject the run length so the shader's per-thread array is sized to match
	// the host's MetalKernelIters, then hand the source to the runtime compiler.
	// NICK_METAL_SRC overrides the embedded shader with a file on disk — for
	// kernel development/profiling without rebuilding the Go binary.
	shader := metalSrc
	if path := os.Getenv("NICK_METAL_SRC"); path != "" {
		if b, err := os.ReadFile(path); err == nil {
			shader = string(b)
		}
	}
	src := fmt.Sprintf("#define NICK_ITERS %d\n%s", MetalKernelIters, shader)
	cSrc := C.CString(src)
	defer C.free(unsafe.Pointer(cSrc))

	// Threadgroup size is occupancy tuning only (no effect on results), so it is
	// safe to expose via env for benchmarking. 0 lets the launcher pick.
	tg := 0
	if v := os.Getenv("NICK_TG"); v != "" {
		if parsed, err := strconv.Atoi(v); err == nil {
			tg = parsed
		}
	}

	var errBuf [256]C.char
	ptr := C.metal_new(C.int(device), cSrc, C.int(MetalKernelIters), C.int(tg),
		&errBuf[0], C.int(len(errBuf)))
	if ptr == nil {
		return nil, fmt.Errorf("metal init: %s", C.GoString(&errBuf[0]))
	}

	m := &MetalMiner{ptr: ptr, batchSize: batchSize}
	var nameBuf [128]C.char
	C.metal_device_name(ptr, &nameBuf[0], C.int(len(nameBuf)))
	m.name = C.GoString(&nameBuf[0])
	return m, nil
}

// Mine runs one batch over [startNonce, startNonce+BatchSize*MetalKernelIters).
func (m *MetalMiner) Mine(p *Precompute, prefix, suffix []byte, startNonce uint64) (*GPUResult, time.Duration, error) {
	if !m.uploaded {
		C.metal_upload(m.ptr,
			(*C.uint8_t)(unsafe.Pointer(&p.DTable[0])), C.int(len(p.DTable)),
			(*C.uint8_t)(unsafe.Pointer(&p.QBaseBytes[0])))
		m.uploaded = true
	}

	var addr [20]byte
	var nonce C.uint64_t
	var pfxPtr, sfxPtr *C.uint8_t
	if len(prefix) > 0 {
		pfxPtr = (*C.uint8_t)(unsafe.Pointer(&prefix[0]))
	}
	if len(suffix) > 0 {
		sfxPtr = (*C.uint8_t)(unsafe.Pointer(&suffix[0]))
	}

	start := time.Now()
	rc := C.metal_mine(m.ptr, C.int(m.batchSize),
		pfxPtr, C.int(len(prefix)),
		sfxPtr, C.int(len(suffix)),
		C.uint64_t(startNonce),
		(*C.uint8_t)(unsafe.Pointer(&addr[0])), &nonce)
	dur := time.Since(start)

	if rc < 0 {
		return nil, dur, fmt.Errorf("metal kernel execution failed")
	}
	if rc == 1 {
		return &GPUResult{Nonce: uint64(nonce), Address: addr}, dur, nil
	}
	return nil, dur, nil
}

// Close releases device resources.
func (m *MetalMiner) Close() {
	if m.ptr != nil {
		C.metal_free(m.ptr)
		m.ptr = nil
	}
}

// DeviceName returns the Metal device name.
func (m *MetalMiner) DeviceName() string { return m.name }

// BatchSize returns the number of GPU threads per launch.
func (m *MetalMiner) BatchSize() int { return m.batchSize }
