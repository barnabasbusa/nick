package miner

import (
	"fmt"
	"math/big"

	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/crypto"
)

// secp256k1 curve parameters (via go-ethereum's decred-backed curve).
var (
	secpCurve = crypto.S256()
	secpN     = secpCurve.Params().N
	secpP     = secpCurve.Params().P
	secpB     = secpCurve.Params().B
)

// Comb table geometry for the 64-bit scalar k used on the GPU.
//
// k*D is evaluated as sum over 8 byte-windows w of  table[w][byte_w(k)], where
// table[w][b] = (b << (8*w)) * D. Digit 0 is the point at infinity and stored as
// all zeros (the kernel skips it).
const (
	combWindows    = 8   // one window per byte of the 64-bit scalar
	combEntries    = 256 // digits 0..255 per window
	feBytes        = 32  // bytes per field element (little-endian on the wire)
	combEntryBytes = 2 * feBytes
	DTableBytes    = combWindows * combEntries * combEntryBytes
	QBaseBytesLen  = 2 * feBytes
)

// Precompute holds everything derived once per search run from the fixed
// (sighash z, signature r, base s) so the GPU kernel only has to add k*D.
type Precompute struct {
	SBase *big.Int // base signature s for k=0; winning s = SBase + k
	R     *big.Int // signature r (fixed)

	QBaseX, QBaseY *big.Int // affine Q_base = recovered pubkey at s = SBase
	DX, DY         *big.Int // affine D = r^-1 * R

	// Wire formats consumed by the GPU hosts (little-endian field elements).
	QBaseBytes [QBaseBytesLen]byte // X(32) || Y(32)
	DTable     []byte              // combWindows*combEntries*combEntryBytes
}

// recoverRPoint reconstructs the curve point R from signature r and recovery id.
// For Ethereum v=27, recid=0, which selects the even-y root.
func recoverRPoint(r *big.Int, recid int) (x, y *big.Int, err error) {
	x = new(big.Int).Set(r)
	if recid >= 2 {
		x.Add(x, secpN)
	}
	if x.Sign() <= 0 || x.Cmp(secpP) >= 0 {
		return nil, nil, fmt.Errorf("r out of field range")
	}

	// y^2 = x^3 + b (b = 7 for secp256k1)
	y2 := new(big.Int).Exp(x, big.NewInt(3), secpP)
	y2.Add(y2, secpB)
	y2.Mod(y2, secpP)

	// secp256k1's p ≡ 3 (mod 4), so sqrt(a) = a^((p+1)/4) mod p.
	exp := new(big.Int).Add(secpP, big.NewInt(1))
	exp.Rsh(exp, 2)
	y = new(big.Int).Exp(y2, exp, secpP)

	// Verify it is an actual square root (r must be a valid x-coordinate).
	chk := new(big.Int).Mul(y, y)
	chk.Mod(chk, secpP)
	if chk.Cmp(y2) != 0 {
		return nil, nil, fmt.Errorf("r is not a valid curve x-coordinate")
	}

	// Select the root whose parity matches recid bit 0.
	if (y.Bit(0) == 1) != (recid&1 == 1) {
		y.Sub(secpP, y)
	}
	return x, y, nil
}

// NewPrecompute derives the run constants from the fixed sighash and signature
// parameters. recid is fixed at 0 (matching the tool's v=27).
//
// ECDSA recovery: Q = r^-1 (s*R - z*G) = u1*G + u2*R with
// u1 = -z*r^-1 mod n and u2 = s*r^-1 mod n. With s = SBase + k this is
// Q = Q_base + k*D where D = r^-1*R.
func NewPrecompute(sighash []byte, r, sBase *big.Int) (*Precompute, error) {
	if len(sighash) != 32 {
		return nil, fmt.Errorf("sighash must be 32 bytes, got %d", len(sighash))
	}
	if r.Sign() <= 0 || r.Cmp(secpN) >= 0 {
		return nil, fmt.Errorf("r out of group range")
	}

	Rx, Ry, err := recoverRPoint(r, 0)
	if err != nil {
		return nil, err
	}

	rInv := new(big.Int).ModInverse(r, secpN)
	if rInv == nil {
		return nil, fmt.Errorf("r not invertible mod n")
	}

	z := new(big.Int).SetBytes(sighash)

	// u1 = -z * rInv mod n
	u1 := new(big.Int).Mul(z, rInv)
	u1.Neg(u1)
	u1.Mod(u1, secpN)

	// u2 = sBase * rInv mod n
	u2 := new(big.Int).Mul(sBase, rInv)
	u2.Mod(u2, secpN)

	// Q_base = u1*G + u2*R
	p0x, p0y := secpCurve.ScalarBaseMult(u1.Bytes())
	rx, ry := secpCurve.ScalarMult(Rx, Ry, u2.Bytes())
	qbx, qby := secpCurve.Add(p0x, p0y, rx, ry)

	// D = rInv * R
	dx, dy := secpCurve.ScalarMult(Rx, Ry, rInv.Bytes())

	p := &Precompute{
		SBase:  new(big.Int).Set(sBase),
		R:      new(big.Int).Set(r),
		QBaseX: qbx, QBaseY: qby,
		DX: dx, DY: dy,
	}
	putFE(p.QBaseBytes[0:feBytes], qbx)
	putFE(p.QBaseBytes[feBytes:], qby)
	p.DTable = p.buildDTable()
	return p, nil
}

// buildDTable precomputes the 8-window comb table for D.
func (p *Precompute) buildDTable() []byte {
	table := make([]byte, DTableBytes)
	for w := 0; w < combWindows; w++ {
		// base = D * 2^(8*w)  (so digit b in this window contributes b*base)
		shift := uint(8 * w)
		baseScalar := new(big.Int).Lsh(big.NewInt(1), shift)
		bx, by := secpCurve.ScalarMult(p.DX, p.DY, baseScalar.Bytes())
		// Accumulate b*base for b = 1..255 by repeated addition.
		var accX, accY *big.Int // nil == point at infinity (digit 0)
		for b := 1; b < combEntries; b++ {
			if accX == nil {
				accX, accY = new(big.Int).Set(bx), new(big.Int).Set(by)
			} else {
				accX, accY = secpCurve.Add(accX, accY, bx, by)
			}
			off := (w*combEntries + b) * combEntryBytes
			putFE(table[off:off+feBytes], accX)
			putFE(table[off+feBytes:off+combEntryBytes], accY)
		}
		// digit 0 entry left as zeros (point at infinity)
	}
	return table
}

// PubForK returns the affine recovered public key for candidate k:
// Q_base + k*D. This is the host reference used to validate the GPU kernel.
func (p *Precompute) PubForK(k uint64) (x, y *big.Int) {
	if k == 0 {
		return new(big.Int).Set(p.QBaseX), new(big.Int).Set(p.QBaseY)
	}
	kx, ky := secpCurve.ScalarMult(p.DX, p.DY, new(big.Int).SetUint64(k).Bytes())
	return secpCurve.Add(p.QBaseX, p.QBaseY, kx, ky)
}

// pubForKViaTable reconstructs k*D from the comb table (host check that the
// uploaded table matches the kernel's expected arithmetic), then adds Q_base.
func (p *Precompute) pubForKViaTable(k uint64) (x, y *big.Int) {
	var accX, accY *big.Int
	for w := 0; w < combWindows; w++ {
		b := int((k >> (8 * uint(w))) & 0xff)
		if b == 0 {
			continue
		}
		off := (w*combEntries + b) * combEntryBytes
		ex := feFromBytes(p.DTable[off : off+feBytes])
		ey := feFromBytes(p.DTable[off+feBytes : off+combEntryBytes])
		if accX == nil {
			accX, accY = ex, ey
		} else {
			accX, accY = secpCurve.Add(accX, accY, ex, ey)
		}
	}
	if accX == nil {
		return new(big.Int).Set(p.QBaseX), new(big.Int).Set(p.QBaseY)
	}
	return secpCurve.Add(p.QBaseX, p.QBaseY, accX, accY)
}

// SenderFromPub derives the EOA address from an affine public key:
// keccak256(X || Y)[12:].
func SenderFromPub(x, y *big.Int) common.Address {
	var buf [64]byte
	x.FillBytes(buf[0:32])
	y.FillBytes(buf[32:64])
	h := crypto.Keccak256(buf[:])
	var a common.Address
	copy(a[:], h[12:])
	return a
}

// putFE writes v as a 32-byte little-endian field element (8 u32 limbs, LSW
// first) — the layout the GPU kernels load.
func putFE(dst []byte, v *big.Int) {
	var be [32]byte
	v.FillBytes(be[:])
	for i := 0; i < 32; i++ {
		dst[i] = be[31-i]
	}
}

// feFromBytes is the inverse of putFE.
func feFromBytes(src []byte) *big.Int {
	var be [32]byte
	for i := 0; i < 32; i++ {
		be[i] = src[31-i]
	}
	return new(big.Int).SetBytes(be[:])
}
