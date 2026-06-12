//go:build !cuda
// +build !cuda

/*
 * Stub CUDA + multi-GPU miners for builds without CUDA support.
 */

package miner

import (
	"fmt"
	"time"
)

// CUDAGPUInfo describes a CUDA device (stub).
type CUDAGPUInfo struct {
	Index        int
	Name         string
	ComputeUnits int
	MaxWorkSize  int
	TotalMemory  uint64
}

// CUDAMiner is a no-op stub.
type CUDAMiner struct{}

// ListCUDAGPUs reports that CUDA is not available.
func ListCUDAGPUs() ([]CUDAGPUInfo, error) {
	return nil, fmt.Errorf("CUDA support not enabled. Build with: make build-cuda")
}

// NewCUDAMiner reports that CUDA is not available.
func NewCUDAMiner(deviceIndex int, batchSize int) (*CUDAMiner, error) {
	return nil, fmt.Errorf("CUDA support not enabled. Build with: make build-cuda")
}

// Close is a no-op.
func (m *CUDAMiner) Close() {}

// DeviceName returns an empty string.
func (m *CUDAMiner) DeviceName() string { return "" }

// BatchSize returns 0.
func (m *CUDAMiner) BatchSize() int { return 0 }

// Mine reports that CUDA is not available.
func (m *CUDAMiner) Mine(p *Precompute, prefix, suffix []byte, startNonce uint64) (*GPUResult, time.Duration, error) {
	return nil, 0, fmt.Errorf("CUDA support not enabled. Build with: make build-cuda")
}

// MultiGPUMiner is a no-op stub.
type MultiGPUMiner struct{}

// NewMultiGPUMiner reports that CUDA is not available.
func NewMultiGPUMiner(deviceIDs []int, batchSize int) (*MultiGPUMiner, error) {
	return nil, fmt.Errorf("CUDA support not enabled. Build with: make build-cuda")
}

// Close is a no-op.
func (m *MultiGPUMiner) Close() {}

// DeviceNames returns nil.
func (m *MultiGPUMiner) DeviceNames() []string { return nil }

// TotalBatchSize returns 0.
func (m *MultiGPUMiner) TotalBatchSize() int { return 0 }

// Mine reports that CUDA is not available.
func (m *MultiGPUMiner) Mine(p *Precompute, prefix, suffix []byte, startNonce uint64) (*GPUResult, time.Duration, error) {
	return nil, 0, fmt.Errorf("CUDA support not enabled. Build with: make build-cuda")
}
