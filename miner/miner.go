// Package miner provides GPU-accelerated searching for Nick's-method vanity
// contract deployment addresses.
//
// Nick's method recovers a deployer EOA from a fixed (r, v) and a varying
// signature s via ecrecover, then derives the contract address as
// keccak256(rlp([sender, 0]))[12:]. Because only s varies, the expensive
// elliptic-curve work collapses to a single fixed point D = r^-1 * R added k
// times to a fixed base point Q_base (see precompute.go). The GPU kernels
// exploit this: candidate k uses s = s_base + k and pubkey Q_base + k*D.
package miner

import "time"

// GPUInfo describes an available GPU device.
type GPUInfo struct {
	Index        int
	Name         string
	Vendor       string
	ComputeUnits int
	MaxWorkSize  int
}

// GPUResult is returned by a GPU miner when a matching address is found.
//
// Nonce is the value k such that the winning signature s = s_base + k. The host
// reconstructs the full transaction from it (see main.go). Address is the
// matched contract deployment address.
type GPUResult struct {
	Nonce   uint64
	Address [20]byte
}

// GPUMinerInterface is implemented by the OpenCL, CUDA and multi-GPU miners.
type GPUMinerInterface interface {
	// Mine runs one batch of the kernel over [startNonce, startNonce+BatchSize).
	// p carries the precomputed Q_base / D comb table; prefix and suffix are the
	// raw target bytes. Returns a result, the batch duration, or nil if no match
	// was found in this batch.
	Mine(p *Precompute, prefix, suffix []byte, startNonce uint64) (*GPUResult, time.Duration, error)

	// Close releases all device resources.
	Close()

	// DeviceName returns the human-readable device name.
	DeviceName() string

	// BatchSize returns the number of candidates processed per Mine call.
	BatchSize() int
}

// HexCharToNibble converts a single hex character to its numeric value.
func HexCharToNibble(c byte) byte {
	if c >= '0' && c <= '9' {
		return c - '0'
	}
	if c >= 'a' && c <= 'f' {
		return c - 'a' + 10
	}
	return c - 'A' + 10
}

// HexToByte converts two hex characters to a byte.
func HexToByte(c1, c2 byte) byte {
	return (HexCharToNibble(c1) << 4) | HexCharToNibble(c2)
}
