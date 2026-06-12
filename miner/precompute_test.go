package miner

import (
	"math/big"
	"testing"

	"github.com/ethereum/go-ethereum/crypto"
)

// ecrecoverSender is the ground-truth reference: exactly what nick-tx's
// recoverPlain does (crypto.Ecrecover with v=27 => recid 0).
func ecrecoverSender(t *testing.T, sighash []byte, r, s *big.Int) [20]byte {
	t.Helper()
	sig := make([]byte, 65)
	r.FillBytes(sig[0:32])
	s.FillBytes(sig[32:64])
	sig[64] = 0 // recid for v=27
	pub, err := crypto.Ecrecover(sighash, sig)
	if err != nil {
		t.Fatalf("Ecrecover failed for s=%v: %v", s, err)
	}
	h := crypto.Keccak256(pub[1:])
	var a [20]byte
	copy(a[:], h[12:])
	return a
}

// TestPrecomputeMatchesEcrecover locks the kernel's math target (Q_base + k*D)
// to go-ethereum's Ecrecover across many k, for both sender and contract
// address. If this passes, the GPU kernel only has to reproduce this arithmetic.
func TestPrecomputeMatchesEcrecover(t *testing.T) {
	cases := []struct {
		name  string
		r     *big.Int
		sBase *big.Int
		z     []byte
	}{
		{"defaults", big.NewInt(0x0539), big.NewInt(0x1337), crypto.Keccak256([]byte("nick deployment one"))},
		{"r=1", big.NewInt(1), big.NewInt(7), crypto.Keccak256([]byte("another sighash"))},
		// Not every integer is a valid curve x-coordinate, so pick a verified one.
		{"larger-r", firstValidR(0x1000000), big.NewInt(0x42), crypto.Keccak256([]byte{0x01, 0x02, 0x03})},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			p, err := NewPrecompute(tc.z, tc.r, tc.sBase)
			if err != nil {
				t.Fatalf("NewPrecompute: %v", err)
			}

			for k := uint64(0); k < 2000; k++ {
				// Model: pubkey from Q_base + k*D.
				x, y := p.PubForK(k)
				sender := SenderFromPub(x, y)
				gotContract := crypto.CreateAddress(sender, 0)

				// Ground truth: ecrecover with s = sBase + k.
				s := new(big.Int).Add(tc.sBase, new(big.Int).SetUint64(k))
				wantSender := ecrecoverSender(t, tc.z, tc.r, s)
				wantContract := crypto.CreateAddress(wantSender, 0)

				if [20]byte(sender) != wantSender {
					t.Fatalf("k=%d sender mismatch: got %x want %x", k, sender, wantSender)
				}
				if gotContract != wantContract {
					t.Fatalf("k=%d contract mismatch: got %x want %x", k, gotContract, wantContract)
				}
			}
		})
	}
}

// firstValidR returns the smallest r >= start that is a valid secp256k1
// x-coordinate (recoverable by recoverRPoint).
func firstValidR(start uint64) *big.Int {
	r := new(big.Int).SetUint64(start)
	one := big.NewInt(1)
	for {
		if _, _, err := recoverRPoint(r, 0); err == nil {
			return new(big.Int).Set(r)
		}
		r.Add(r, one)
	}
}

// TestInvalidRRejected confirms an r that is not a valid x-coordinate errors out
// rather than producing a wrong result.
func TestInvalidRRejected(t *testing.T) {
	// 0xdeadbeef is not a valid secp256k1 x-coordinate.
	if _, err := NewPrecompute(crypto.Keccak256([]byte("x")), big.NewInt(0xdeadbeef), big.NewInt(1)); err == nil {
		t.Fatal("expected error for invalid r, got nil")
	}
}

// TestCombTableMatchesScalarMul verifies the uploaded comb table reproduces
// k*D, i.e. the data the GPU kernel will consume is correct.
func TestCombTableMatchesScalarMul(t *testing.T) {
	p, err := NewPrecompute(crypto.Keccak256([]byte("table check")), big.NewInt(0x0539), big.NewInt(0x1337))
	if err != nil {
		t.Fatalf("NewPrecompute: %v", err)
	}
	if len(p.DTable) != DTableBytes {
		t.Fatalf("DTable size = %d, want %d", len(p.DTable), DTableBytes)
	}

	for _, k := range []uint64{0, 1, 2, 255, 256, 257, 0xabcd, 0x1234567, 0xfffffffff} {
		ax, ay := p.PubForK(k)
		bx, by := p.pubForKViaTable(k)
		if ax.Cmp(bx) != 0 || ay.Cmp(by) != 0 {
			t.Fatalf("k=%d table-derived pubkey mismatch:\n got  %x,%x\n want %x,%x", k, bx, by, ax, ay)
		}
	}
}

// TestFERoundTrip checks the little-endian field-element wire format.
func TestFERoundTrip(t *testing.T) {
	for _, v := range []*big.Int{big.NewInt(0), big.NewInt(1), big.NewInt(0x0102030405060708), secpP} {
		var buf [32]byte
		putFE(buf[:], new(big.Int).Mod(v, secpP))
		got := feFromBytes(buf[:])
		want := new(big.Int).Mod(v, secpP)
		if got.Cmp(want) != 0 {
			t.Fatalf("round trip: got %x want %x", got, want)
		}
	}
}
