//go:build cuda && linux
// +build cuda,linux

package miner

import (
	"fmt"
	"sync"
	"sync/atomic"
	"time"
)

// MultiGPUMiner runs the nick kernel across several CUDA devices concurrently.
type MultiGPUMiner struct {
	miners    []*CUDAMiner
	deviceIDs []int
}

// NewMultiGPUMiner initializes one CUDAMiner per device.
func NewMultiGPUMiner(deviceIDs []int, batchSize int) (*MultiGPUMiner, error) {
	if len(deviceIDs) == 0 {
		return nil, fmt.Errorf("no devices specified")
	}
	miners := make([]*CUDAMiner, 0, len(deviceIDs))
	for _, id := range deviceIDs {
		m, err := NewCUDAMiner(id, batchSize)
		if err != nil {
			for _, created := range miners {
				created.Close()
			}
			return nil, fmt.Errorf("failed to initialize miner on device %d: %v", id, err)
		}
		miners = append(miners, m)
	}
	return &MultiGPUMiner{miners: miners, deviceIDs: deviceIDs}, nil
}

// Close releases all device resources.
func (m *MultiGPUMiner) Close() {
	for _, miner := range m.miners {
		miner.Close()
	}
}

// DeviceNames returns the names of all devices.
func (m *MultiGPUMiner) DeviceNames() []string {
	names := make([]string, len(m.miners))
	for i, miner := range m.miners {
		names[i] = miner.DeviceName()
	}
	return names
}

// TotalBatchSize returns the combined per-iteration candidate count.
func (m *MultiGPUMiner) TotalBatchSize() int {
	total := 0
	for _, miner := range m.miners {
		total += miner.BatchSize()
	}
	return total
}

// Mine runs one batch per device over disjoint nonce sub-ranges.
func (m *MultiGPUMiner) Mine(p *Precompute, prefix, suffix []byte, startNonce uint64) (*GPUResult, time.Duration, error) {
	startTime := time.Now()

	resultChan := make(chan *GPUResult, 1)
	errorChan := make(chan error, 1)
	doneChan := make(chan struct{})

	var wg sync.WaitGroup
	var found atomic.Bool

	for i, miner := range m.miners {
		wg.Add(1)
		go func(idx int, miner *CUDAMiner) {
			defer wg.Done()

			// Each device covers BatchSize()*KernelIters candidates per call.
			gpuOffset := uint64(0)
			for j := 0; j < idx; j++ {
				gpuOffset += uint64(m.miners[j].BatchSize()) * uint64(KernelIters)
			}
			res, _, err := miner.Mine(p, prefix, suffix, startNonce+gpuOffset)
			if err != nil {
				select {
				case errorChan <- fmt.Errorf("device %d error: %v", m.deviceIDs[idx], err):
				default:
				}
				return
			}
			if res != nil && found.CompareAndSwap(false, true) {
				select {
				case resultChan <- res:
				default:
				}
			}
		}(i, miner)
	}

	go func() {
		wg.Wait()
		close(doneChan)
	}()

	select {
	case res := <-resultChan:
		return res, time.Since(startTime), nil
	case err := <-errorChan:
		return nil, time.Since(startTime), err
	case <-doneChan:
		return nil, time.Since(startTime), nil
	}
}
