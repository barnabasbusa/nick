//go:build nocl || (!linux && !darwin)
// +build nocl !linux,!darwin

/*
 * Stub OpenCL miner used when OpenCL is unavailable: on unsupported platforms,
 * or when built with -tags nocl (e.g. CPU-only / no libOpenCL installed).
 */

package miner

import (
	"fmt"
	"time"
)

// GPUMiner is a no-op stub.
type GPUMiner struct{}

// ListGPUs reports that OpenCL is not available in this build.
func ListGPUs() ([]GPUInfo, error) {
	return nil, fmt.Errorf("OpenCL GPU mining not available in this build (needs Linux/macOS with OpenCL, built without -tags nocl)")
}

// NewGPUMiner reports that OpenCL is not available in this build.
func NewGPUMiner(deviceIndex int, batchSize int) (*GPUMiner, error) {
	return nil, fmt.Errorf("OpenCL GPU mining not available in this build")
}

// Close is a no-op.
func (m *GPUMiner) Close() {}

// DeviceName returns an empty string.
func (m *GPUMiner) DeviceName() string { return "" }

// BatchSize returns 0.
func (m *GPUMiner) BatchSize() int { return 0 }

// Mine reports that OpenCL is not available in this build.
func (m *GPUMiner) Mine(p *Precompute, prefix, suffix []byte, startNonce uint64) (*GPUResult, time.Duration, error) {
	return nil, 0, fmt.Errorf("OpenCL GPU mining not available in this build")
}
