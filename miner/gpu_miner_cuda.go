//go:build cuda && linux
// +build cuda,linux

/*
 * CUDA miner (Linux + NVIDIA) for Nick's-method vanity addresses.
 *
 * Build: compile the CUDA library first (make build-cuda-lib), then
 *        go build -tags cuda.
 */

package miner

/*
#cgo LDFLAGS: -L${SRCDIR}/kernel -lnick_cuda -L/usr/local/cuda/lib64 -lcudart -lstdc++ -lm
#cgo CFLAGS: -I/usr/local/cuda/include

#include <stdlib.h>

typedef struct {
    int index;
    char name[256];
    int compute_units;
    int max_threads_per_block;
    unsigned long long total_memory;
} CUDADeviceInfo;

typedef struct NickCUDAContext NickCUDAContext;

extern int get_cuda_device_count();
extern int get_cuda_device_info(int index, CUDADeviceInfo* info);
extern NickCUDAContext* nick_cuda_init(int device_index, int batch_size);
extern void nick_cuda_close(NickCUDAContext* ctx);
extern int nick_cuda_set_table(NickCUDAContext* ctx, const unsigned char* table, int table_len);
extern int nick_cuda_mine(
    NickCUDAContext* ctx,
    const unsigned char* qbase,
    const unsigned char* prefix, int prefix_len,
    const unsigned char* suffix, int suffix_len,
    unsigned long long start_nonce,
    unsigned char* result_addr,
    unsigned long long* result_nonce);
*/
import "C"

import (
	"fmt"
	"time"
	"unsafe"
)

// CUDAGPUInfo describes a CUDA device.
type CUDAGPUInfo struct {
	Index        int
	Name         string
	ComputeUnits int
	MaxWorkSize  int
	TotalMemory  uint64
}

// CUDAMiner is a CUDA miner bound to a single device.
type CUDAMiner struct {
	ctx        *C.NickCUDAContext
	deviceName string
	batchSize  int
	loadedFor  unsafe.Pointer
}

// ListCUDAGPUs returns the available CUDA devices.
func ListCUDAGPUs() ([]CUDAGPUInfo, error) {
	count := int(C.get_cuda_device_count())
	if count <= 0 {
		if count < 0 {
			return nil, fmt.Errorf("CUDA error %d: check driver/toolkit compatibility", -count)
		}
		return nil, fmt.Errorf("no CUDA devices found")
	}
	gpus := make([]CUDAGPUInfo, count)
	for i := 0; i < count; i++ {
		var info C.CUDADeviceInfo
		if C.get_cuda_device_info(C.int(i), &info) != 0 {
			continue
		}
		gpus[i] = CUDAGPUInfo{
			Index:        int(info.index),
			Name:         C.GoString(&info.name[0]),
			ComputeUnits: int(info.compute_units),
			MaxWorkSize:  int(info.max_threads_per_block),
			TotalMemory:  uint64(info.total_memory),
		}
	}
	return gpus, nil
}

// NewCUDAMiner initializes a CUDA miner on the given device.
func NewCUDAMiner(deviceIndex int, batchSize int) (*CUDAMiner, error) {
	ctx := C.nick_cuda_init(C.int(deviceIndex), C.int(batchSize))
	if ctx == nil {
		return nil, fmt.Errorf("failed to initialize CUDA miner on device %d", deviceIndex)
	}
	name := fmt.Sprintf("CUDA device %d", deviceIndex)
	var info C.CUDADeviceInfo
	if C.get_cuda_device_info(C.int(deviceIndex), &info) == 0 {
		name = C.GoString(&info.name[0])
	}
	return &CUDAMiner{ctx: ctx, deviceName: name, batchSize: batchSize}, nil
}

// Close releases CUDA resources.
func (m *CUDAMiner) Close() {
	if m.ctx != nil {
		C.nick_cuda_close(m.ctx)
		m.ctx = nil
	}
}

// DeviceName returns the device name.
func (m *CUDAMiner) DeviceName() string { return m.deviceName }

// BatchSize returns the per-batch candidate count.
func (m *CUDAMiner) BatchSize() int { return m.batchSize }

// Mine runs one batch on the GPU.
func (m *CUDAMiner) Mine(p *Precompute, prefix, suffix []byte, startNonce uint64) (*GPUResult, time.Duration, error) {
	start := time.Now()

	if m.loadedFor != unsafe.Pointer(p) {
		if r := C.nick_cuda_set_table(m.ctx, (*C.uchar)(unsafe.Pointer(&p.DTable[0])), C.int(len(p.DTable))); r != 0 {
			return nil, 0, fmt.Errorf("cuda set_table failed")
		}
		m.loadedFor = unsafe.Pointer(p)
	}

	var prefixPtr, suffixPtr *C.uchar
	if len(prefix) > 0 {
		prefixPtr = (*C.uchar)(unsafe.Pointer(&prefix[0]))
	}
	if len(suffix) > 0 {
		suffixPtr = (*C.uchar)(unsafe.Pointer(&suffix[0]))
	}

	var addr [20]byte
	var nonce C.ulonglong
	ret := C.nick_cuda_mine(
		m.ctx,
		(*C.uchar)(unsafe.Pointer(&p.QBaseBytes[0])),
		prefixPtr, C.int(len(prefix)),
		suffixPtr, C.int(len(suffix)),
		C.ulonglong(startNonce),
		(*C.uchar)(unsafe.Pointer(&addr[0])),
		&nonce,
	)
	elapsed := time.Since(start)

	if ret < 0 {
		return nil, elapsed, fmt.Errorf("cuda mining error")
	}
	if ret == 1 {
		res := &GPUResult{Nonce: uint64(nonce), Address: addr}
		return res, elapsed, nil
	}
	return nil, elapsed, nil
}
