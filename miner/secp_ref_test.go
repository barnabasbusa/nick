package miner

// This file is a Go reference implementation of the EXACT 4x64-bit-limb
// secp256k1 field and point arithmetic that the GPU kernels (secp256k1.cl /
// secp256k1.cuh) implement. It exists only to validate that arithmetic against
// big.Int before/independently of running on a GPU. The C kernels are a
// near-verbatim translation of the functions below.
//
// Field elements are 4 little-endian uint64 limbs (limb[0] = least significant).
// Reduction exploits p = 2^256 - C with C = 0x1000003D1 (= 2^32 + 977).

import (
	"math/big"
	"math/bits"
	"testing"
)

type fe [4]uint64

const refC uint64 = 0x1000003D1

var (
	pLimbs  = fe{0xFFFFFFFEFFFFFC2F, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF}
	feOne   = fe{1, 0, 0, 0}
	pMinus2 = fe{0xFFFFFFFEFFFFFC2D, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF}
)

// --- helpers / conversions ---

func feFromBig(v *big.Int) fe {
	var be [32]byte
	new(big.Int).Mod(v, pBig()).FillBytes(be[:])
	var r fe
	for i := 0; i < 4; i++ {
		// limb i = bytes [32-8*(i+1) : 32-8*i] big-endian
		off := 32 - 8*(i+1)
		var w uint64
		for j := 0; j < 8; j++ {
			w = (w << 8) | uint64(be[off+j])
		}
		r[i] = w
	}
	return r
}

func (a fe) toBig() *big.Int {
	r := new(big.Int)
	for i := 3; i >= 0; i-- {
		r.Lsh(r, 64)
		r.Or(r, new(big.Int).SetUint64(a[i]))
	}
	return r
}

func pBig() *big.Int { return secpP }

// --- field arithmetic (mirrors the kernel) ---

func feGE(a, b *fe) bool {
	for i := 3; i >= 0; i-- {
		if a[i] != b[i] {
			return a[i] > b[i]
		}
	}
	return true
}

func feCondSubP(r *fe) {
	for feGE(r, &pLimbs) {
		var br uint64
		r[0], br = bits.Sub64(r[0], pLimbs[0], 0)
		r[1], br = bits.Sub64(r[1], pLimbs[1], br)
		r[2], br = bits.Sub64(r[2], pLimbs[2], br)
		r[3], br = bits.Sub64(r[3], pLimbs[3], br)
	}
}

// foldCarry folds a carry out of bit 256 back in (since 2^256 ≡ C mod p).
func foldCarry(r *fe, carry uint64) {
	for carry != 0 {
		add := carry * refC
		var c uint64
		r[0], c = bits.Add64(r[0], add, 0)
		r[1], c = bits.Add64(r[1], 0, c)
		r[2], c = bits.Add64(r[2], 0, c)
		r[3], c = bits.Add64(r[3], 0, c)
		carry = c
	}
}

func feAdd(r, a, b *fe) {
	var c uint64
	r[0], c = bits.Add64(a[0], b[0], 0)
	r[1], c = bits.Add64(a[1], b[1], c)
	r[2], c = bits.Add64(a[2], b[2], c)
	r[3], c = bits.Add64(a[3], b[3], c)
	foldCarry(r, c)
	feCondSubP(r)
}

func feSub(r, a, b *fe) {
	var br uint64
	r[0], br = bits.Sub64(a[0], b[0], 0)
	r[1], br = bits.Sub64(a[1], b[1], br)
	r[2], br = bits.Sub64(a[2], b[2], br)
	r[3], br = bits.Sub64(a[3], b[3], br)
	if br != 0 {
		// add p == subtract C (mod 2^256); cannot borrow again
		var b2 uint64
		r[0], b2 = bits.Sub64(r[0], refC, 0)
		r[1], b2 = bits.Sub64(r[1], 0, b2)
		r[2], b2 = bits.Sub64(r[2], 0, b2)
		r[3], b2 = bits.Sub64(r[3], 0, b2)
	}
}

// mulScalar5 = (a0..a3) * c into 5 limbs, with c < 2^34.
func mulScalar5(out *[5]uint64, a0, a1, a2, a3, c uint64) {
	var carry, cc uint64
	hi, lo := bits.Mul64(a0, c)
	out[0] = lo
	carry = hi
	hi, lo = bits.Mul64(a1, c)
	out[1], cc = bits.Add64(lo, carry, 0)
	carry = hi + cc
	hi, lo = bits.Mul64(a2, c)
	out[2], cc = bits.Add64(lo, carry, 0)
	carry = hi + cc
	hi, lo = bits.Mul64(a3, c)
	out[3], cc = bits.Add64(lo, carry, 0)
	carry = hi + cc
	out[4] = carry
}

// feReduce reduces a 512-bit value t[0..7] mod p into r.
func feReduce(r *fe, t *[8]uint64) {
	// acc = low(t[0..3]) + high(t[4..7]) * C   -> up to 5 limbs
	var hiC [5]uint64
	mulScalar5(&hiC, t[4], t[5], t[6], t[7], refC)

	var m [5]uint64
	var cc uint64
	m[0], cc = bits.Add64(t[0], hiC[0], 0)
	m[1], cc = bits.Add64(t[1], hiC[1], cc)
	m[2], cc = bits.Add64(t[2], hiC[2], cc)
	m[3], cc = bits.Add64(t[3], hiC[3], cc)
	m[4] = hiC[4] + cc

	// fold m[4] (the bits >= 2^256) back: value = m[0..3] + m[4]*C
	hi, lo := bits.Mul64(m[4], refC)
	r[0], cc = bits.Add64(m[0], lo, 0)
	r[1], cc = bits.Add64(m[1], hi, cc)
	r[2], cc = bits.Add64(m[2], 0, cc)
	r[3], cc = bits.Add64(m[3], 0, cc)
	foldCarry(r, cc)
	feCondSubP(r)
}

// feMul via Comba (column) multiplication with a 3-word accumulator.
func feMul(r, a, b *fe) {
	var t [8]uint64
	var c0, c1, c2 uint64
	for col := 0; col < 7; col++ {
		lo := col - 3
		if lo < 0 {
			lo = 0
		}
		hi := col
		if hi > 3 {
			hi = 3
		}
		for i := lo; i <= hi; i++ {
			j := col - i
			ph, pl := bits.Mul64(a[i], b[j])
			var carry uint64
			c0, carry = bits.Add64(c0, pl, 0)
			c1, carry = bits.Add64(c1, ph, carry)
			c2 += carry
		}
		t[col] = c0
		c0, c1, c2 = c1, c2, 0
	}
	t[7] = c0
	feReduce(r, &t)
}

func feSqr(r, a *fe) { feMul(r, a, a) }

func feInv(r, a *fe) {
	// a^(p-2) mod p, LSB-first square-and-multiply.
	acc := feOne
	base := *a
	for i := 0; i < 256; i++ {
		if (pMinus2[i/64]>>(uint(i)%64))&1 == 1 {
			feMul(&acc, &acc, &base)
		}
		feSqr(&base, &base)
	}
	*r = acc
}

func feIsZero(a *fe) bool { return a[0]|a[1]|a[2]|a[3] == 0 }
func feEq(a, b *fe) bool  { return a[0] == b[0] && a[1] == b[1] && a[2] == b[2] && a[3] == b[3] }

// --- point arithmetic (Jacobian, a=0) ---

type jac struct {
	X, Y, Z fe // Z == 0 means point at infinity
}

func jacIsInf(p *jac) bool { return feIsZero(&p.Z) }

// pointDouble: dbl-2009-l for a=0.
func pointDouble(r, p *jac) {
	if jacIsInf(p) || feIsZero(&p.Y) {
		*r = jac{} // infinity
		return
	}
	var A, B, C, D, E, F, t0, t1 fe
	feSqr(&A, &p.X)
	feSqr(&B, &p.Y)
	feSqr(&C, &B)
	feAdd(&t0, &p.X, &B)
	feSqr(&t0, &t0)
	feSub(&t0, &t0, &A)
	feSub(&t0, &t0, &C)
	feAdd(&D, &t0, &t0) // D = 2*((X+B)^2 - A - C)
	feAdd(&E, &A, &A)
	feAdd(&E, &E, &A) // E = 3*A
	feSqr(&F, &E)
	feAdd(&t0, &D, &D)
	var X3, Y3, Z3 fe
	feSub(&X3, &F, &t0) // X3 = F - 2D
	feSub(&t1, &D, &X3)
	feMul(&Y3, &E, &t1)
	// 8*C
	feAdd(&t0, &C, &C)
	feAdd(&t0, &t0, &t0)
	feAdd(&t0, &t0, &t0)
	feSub(&Y3, &Y3, &t0)
	feMul(&Z3, &p.Y, &p.Z)
	feAdd(&Z3, &Z3, &Z3) // Z3 = 2*Y*Z
	r.X, r.Y, r.Z = X3, Y3, Z3
}

// pointAddMixed: P (Jacobian) + Q (affine x2,y2). madd-2007-bl with edge cases.
func pointAddMixed(r, p *jac, x2, y2 *fe) {
	if jacIsInf(p) {
		r.X, r.Y = *x2, *y2
		r.Z = feOne
		return
	}
	var Z1Z1, U2, S2, H, HH, I, J, rr, V, t0, t1 fe
	feSqr(&Z1Z1, &p.Z)
	feMul(&U2, x2, &Z1Z1)
	feMul(&t0, y2, &p.Z)
	feMul(&S2, &t0, &Z1Z1) // S2 = Y2*Z1*Z1Z1
	feSub(&H, &U2, &p.X)
	feSub(&rr, &S2, &p.Y)
	if feIsZero(&H) {
		if feIsZero(&rr) {
			pointDouble(r, p)
			return
		}
		*r = jac{} // infinity
		return
	}
	feAdd(&rr, &rr, &rr) // r = 2*(S2 - Y1)
	feSqr(&HH, &H)
	feAdd(&I, &HH, &HH)
	feAdd(&I, &I, &I) // I = 4*HH
	feMul(&J, &H, &I)
	feMul(&V, &p.X, &I)
	var X3, Y3, Z3 fe
	feSqr(&X3, &rr)
	feSub(&X3, &X3, &J)
	feAdd(&t0, &V, &V)
	feSub(&X3, &X3, &t0) // X3 = r^2 - J - 2V
	feSub(&t1, &V, &X3)
	feMul(&Y3, &rr, &t1)
	feMul(&t0, &p.Y, &J)
	feAdd(&t0, &t0, &t0)
	feSub(&Y3, &Y3, &t0) // Y3 = r*(V-X3) - 2*Y1*J
	feAdd(&t0, &p.Z, &H)
	feSqr(&t0, &t0)
	feSub(&t0, &t0, &Z1Z1)
	feSub(&Z3, &t0, &HH) // Z3 = (Z1+H)^2 - Z1Z1 - HH
	r.X, r.Y, r.Z = X3, Y3, Z3
}

func jacToAffine(p *jac) (x, y fe) {
	var zinv, zinv2, zinv3 fe
	feInv(&zinv, &p.Z)
	feSqr(&zinv2, &zinv)
	feMul(&zinv3, &zinv2, &zinv)
	feMul(&x, &p.X, &zinv2)
	feMul(&y, &p.Y, &zinv3)
	return
}

// refPubForK reproduces the kernel: Q = Qbase + k*D using the comb table.
func refPubForK(p *Precompute, k uint64) (x, y fe) {
	var acc jac // infinity
	for w := 0; w < combWindows; w++ {
		b := (k >> (8 * uint(w))) & 0xff
		if b == 0 {
			continue
		}
		off := (w*combEntries + int(b)) * combEntryBytes
		tx := feFromBig(feFromBytes(p.DTable[off : off+feBytes]))
		ty := feFromBig(feFromBytes(p.DTable[off+feBytes : off+combEntryBytes]))
		pointAddMixed(&acc, &acc, &tx, &ty)
	}
	qbx := feFromBig(p.QBaseX)
	qby := feFromBig(p.QBaseY)
	pointAddMixed(&acc, &acc, &qbx, &qby)
	return jacToAffine(&acc)
}

// --- tests ---

func bigToFe(v *big.Int) fe { return feFromBig(v) }

func TestFieldArithmeticVsBig(t *testing.T) {
	vals := []*big.Int{
		big.NewInt(0), big.NewInt(1), big.NewInt(2),
		new(big.Int).Sub(secpP, big.NewInt(1)),
		new(big.Int).Sub(secpP, big.NewInt(2)),
		new(big.Int).Rsh(secpP, 1),
		secpCurve.Params().Gx, secpCurve.Params().Gy,
	}
	// add some pseudo-random-but-deterministic values
	seed := new(big.Int).SetUint64(0x9e3779b97f4a7c15)
	for i := 0; i < 64; i++ {
		seed.Mul(seed, big.NewInt(6364136223846793005))
		seed.Add(seed, big.NewInt(1442695040888963407))
		seed.Mod(seed, secpP)
		vals = append(vals, new(big.Int).Set(seed))
	}

	for _, a := range vals {
		for _, b := range vals {
			fa, fb := bigToFe(a), bigToFe(b)
			var r fe

			feAdd(&r, &fa, &fb)
			if want := new(big.Int).Mod(new(big.Int).Add(a, b), secpP); r.toBig().Cmp(want) != 0 {
				t.Fatalf("add(%x,%x)=%x want %x", a, b, r.toBig(), want)
			}
			feSub(&r, &fa, &fb)
			if want := new(big.Int).Mod(new(big.Int).Sub(a, b), secpP); r.toBig().Cmp(want) != 0 {
				t.Fatalf("sub(%x,%x)=%x want %x", a, b, r.toBig(), want)
			}
			feMul(&r, &fa, &fb)
			if want := new(big.Int).Mod(new(big.Int).Mul(a, b), secpP); r.toBig().Cmp(want) != 0 {
				t.Fatalf("mul(%x,%x)=%x want %x", a, b, r.toBig(), want)
			}
		}
		// square and inverse
		fa := bigToFe(a)
		var r fe
		feSqr(&r, &fa)
		if want := new(big.Int).Mod(new(big.Int).Mul(a, a), secpP); r.toBig().Cmp(want) != 0 {
			t.Fatalf("sqr(%x)=%x want %x", a, r.toBig(), want)
		}
		if a.Sign() != 0 {
			feInv(&r, &fa)
			if want := new(big.Int).ModInverse(a, secpP); r.toBig().Cmp(want) != 0 {
				t.Fatalf("inv(%x)=%x want %x", a, r.toBig(), want)
			}
		}
	}
}

func TestPointArithmeticVsCurve(t *testing.T) {
	// Build affine points k*G for several k, validate double & mixed-add.
	mkPoint := func(k int64) (x, y fe, bx, by *big.Int) {
		bx, by = secpCurve.ScalarBaseMult(big.NewInt(k).Bytes())
		return feFromBig(bx), feFromBig(by), bx, by
	}

	for _, k := range []int64{1, 2, 3, 5, 7, 12345, 9999991} {
		px, py, bx, by := mkPoint(k)
		// double via Jacobian
		p := jac{X: px, Y: py, Z: feOne}
		var d jac
		pointDouble(&d, &p)
		ax, ay := jacToAffine(&d)
		wx, wy := secpCurve.Double(bx, by)
		if ax.toBig().Cmp(wx) != 0 || ay.toBig().Cmp(wy) != 0 {
			t.Fatalf("double k=%d mismatch", k)
		}
		// mixed add p + G
		gx, gy := feFromBig(secpCurve.Params().Gx), feFromBig(secpCurve.Params().Gy)
		var s jac
		pointAddMixed(&s, &p, &gx, &gy)
		ax, ay = jacToAffine(&s)
		wx, wy = secpCurve.Add(bx, by, secpCurve.Params().Gx, secpCurve.Params().Gy)
		if ax.toBig().Cmp(wx) != 0 || ay.toBig().Cmp(wy) != 0 {
			t.Fatalf("mixed-add k=%d mismatch", k)
		}
	}
}

// TestKernelRefMatchesEcrecover is the full end-to-end check of the limb-level
// pipeline that the GPU kernel runs: comb(k*D) + Q_base -> affine -> sender,
// compared against go-ethereum Ecrecover. If this passes, the C kernel (a
// translation of the functions above) computes the right addresses.
func TestKernelRefMatchesEcrecover(t *testing.T) {
	var z [32]byte
	secpCurve.Params().Gx.FillBytes(z[:]) // arbitrary fixed 32-byte sighash
	p, err := NewPrecompute(z[:], big.NewInt(0x0539), big.NewInt(0x1337))
	if err != nil {
		t.Fatalf("NewPrecompute: %v", err)
	}
	for _, k := range []uint64{0, 1, 2, 3, 255, 256, 257, 65535, 65536, 1<<20 + 7, 0xdeadbeef} {
		x, y := refPubForK(p, k)
		sender := SenderFromPub(x.toBig(), y.toBig())
		want := ecrecoverSender(t, z[:], p.R, new(big.Int).Add(p.SBase, new(big.Int).SetUint64(k)))
		if [20]byte(sender) != want {
			t.Fatalf("k=%d kernel-ref sender %x != ecrecover %x", k, sender, want)
		}
	}
}
