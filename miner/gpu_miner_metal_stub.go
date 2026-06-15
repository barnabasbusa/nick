//go:build !metal

package miner

import (
	"fmt"
	"time"
)

// MetalMiner is a build-time placeholder when the `metal` tag is absent.
type MetalMiner struct{}

// ListMetalGPUs reports that Metal support was not compiled in.
func ListMetalGPUs() ([]GPUInfo, error) {
	return nil, fmt.Errorf("metal support not built (rebuild with `make build-metal`)")
}

// NewMetalMiner reports that Metal support was not compiled in.
func NewMetalMiner(device, batchSize int) (*MetalMiner, error) {
	return nil, fmt.Errorf("metal support not built (rebuild with `make build-metal`)")
}

func (m *MetalMiner) Mine(p *Precompute, prefix, suffix []byte, startNonce uint64) (*GPUResult, time.Duration, error) {
	return nil, 0, fmt.Errorf("metal support not built")
}
func (m *MetalMiner) Close()             {}
func (m *MetalMiner) DeviceName() string { return "" }
func (m *MetalMiner) BatchSize() int     { return 0 }
