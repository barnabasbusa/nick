/*
 * nick.metal - Apple Silicon (Metal) port of nick_lib.cl + nick.cl.
 *
 * secp256k1 field/point arithmetic + keccak256 for Nick's-method vanity mining.
 * This is a line-for-line translation of the OpenCL kernel; the only material
 * differences are Metal-specific:
 *
 *   - Apple GPUs have no 64-bit mul-high intrinsic, so MULHI is a software
 *     mulhi64 built from four 32-bit products.
 *   - Every pointer parameter carries an explicit address space (device/thread).
 *   - The winner is claimed with atomic_compare_exchange on a device atomic_int.
 *
 * Field elements are 4 little-endian 64-bit limbs. Reduction uses
 * p = 2^256 - C with C = 0x1000003D1. The host injects `#define NICK_ITERS N`
 * ahead of this source, so NICK_ITERS is a compile-time constant here.
 */
#include <metal_stdlib>
using namespace metal;

typedef ulong u64;
typedef uint  u32;
typedef uchar u8;

#define NI __attribute__((noinline))

#ifndef NICK_ITERS
#define NICK_ITERS 64
#endif

/* software 64x64 -> high 64 bits (Apple GPUs lack a u64 mul_hi) */
inline u64 mulhi64(u64 a, u64 b) {
    u32 a0 = (u32)a, a1 = (u32)(a >> 32);
    u32 b0 = (u32)b, b1 = (u32)(b >> 32);
    u64 ll = (u64)a0 * b0;
    u64 lh = (u64)a0 * b1;
    u64 hl = (u64)a1 * b0;
    u64 hh = (u64)a1 * b1;
    u64 mid = (ll >> 32) + (lh & 0xffffffff) + (hl & 0xffffffff);
    return hh + (lh >> 32) + (hl >> 32) + (mid >> 32);
}
#define MULHI(a, b) mulhi64((a), (b))

#define SECP_C  ((u64)0x1000003D1UL)
#define SECP_P0 ((u64)0xFFFFFFFEFFFFFC2FUL)
#define SECP_PF ((u64)0xFFFFFFFFFFFFFFFFUL)

typedef struct { u64 n[4]; } fe;
typedef struct { fe X, Y, Z; } jac; /* Jacobian; Z == 0 => infinity */

/* add with carry */
inline u64 addc(u64 a, u64 b, thread u64 *carry) {
    u64 s = a + b;
    u64 c1 = (u64)(s < a);
    u64 s2 = s + *carry;
    u64 c2 = (u64)(s2 < s);
    *carry = c1 + c2;
    return s2;
}

/* subtract with borrow */
inline u64 subb(u64 a, u64 b, thread u64 *borrow) {
    u64 d = a - b;
    u64 b1 = (u64)(a < b);
    u64 d2 = d - *borrow;
    u64 b2 = (u64)(d < *borrow);
    *borrow = b1 + b2;
    return d2;
}

inline int  fe_is_zero(thread const fe *a) { return (a->n[0] | a->n[1] | a->n[2] | a->n[3]) == 0; }
inline void fe_set_zero(thread fe *a) { a->n[0] = a->n[1] = a->n[2] = a->n[3] = 0; }
inline void fe_set_one(thread fe *a)  { a->n[0] = 1; a->n[1] = a->n[2] = a->n[3] = 0; }

inline int fe_ge_p(thread const fe *a) {
    if (a->n[3] != SECP_PF) return a->n[3] > SECP_PF;
    if (a->n[2] != SECP_PF) return a->n[2] > SECP_PF;
    if (a->n[1] != SECP_PF) return a->n[1] > SECP_PF;
    return a->n[0] >= SECP_P0;
}

inline void fe_cond_sub_p(thread fe *r) {
    while (fe_ge_p(r)) {
        u64 br = 0;
        r->n[0] = subb(r->n[0], SECP_P0, &br);
        r->n[1] = subb(r->n[1], SECP_PF, &br);
        r->n[2] = subb(r->n[2], SECP_PF, &br);
        r->n[3] = subb(r->n[3], SECP_PF, &br);
    }
}

/* fold a carry out of bit 256 back in: 2^256 == C (mod p) */
inline void fold_carry(thread fe *r, u64 carry) {
    while (carry != 0) {
        u64 add = carry * SECP_C;
        u64 c = 0;
        r->n[0] = addc(r->n[0], add, &c);
        r->n[1] = addc(r->n[1], 0, &c);
        r->n[2] = addc(r->n[2], 0, &c);
        r->n[3] = addc(r->n[3], 0, &c);
        carry = c;
    }
}

inline void fe_add(thread fe *r, thread const fe *a, thread const fe *b) {
    u64 c = 0;
    r->n[0] = addc(a->n[0], b->n[0], &c);
    r->n[1] = addc(a->n[1], b->n[1], &c);
    r->n[2] = addc(a->n[2], b->n[2], &c);
    r->n[3] = addc(a->n[3], b->n[3], &c);
    fold_carry(r, c);
    fe_cond_sub_p(r);
}

inline void fe_sub(thread fe *r, thread const fe *a, thread const fe *b) {
    u64 br = 0;
    r->n[0] = subb(a->n[0], b->n[0], &br);
    r->n[1] = subb(a->n[1], b->n[1], &br);
    r->n[2] = subb(a->n[2], b->n[2], &br);
    r->n[3] = subb(a->n[3], b->n[3], &br);
    if (br != 0) {
        u64 b2 = 0;
        r->n[0] = subb(r->n[0], SECP_C, &b2);
        r->n[1] = subb(r->n[1], 0, &b2);
        r->n[2] = subb(r->n[2], 0, &b2);
        r->n[3] = subb(r->n[3], 0, &b2);
    }
}

/* (a0..a3) * c -> 5 limbs out, c small (< 2^34) */
inline void mul_scalar5(thread u64 out[5], u64 a0, u64 a1, u64 a2, u64 a3, u64 c) {
    u64 carry, cc, hi, lo;
    lo = a0 * c; hi = MULHI(a0, c); out[0] = lo; carry = hi;
    lo = a1 * c; hi = MULHI(a1, c); cc = 0; out[1] = addc(lo, carry, &cc); carry = hi + cc;
    lo = a2 * c; hi = MULHI(a2, c); cc = 0; out[2] = addc(lo, carry, &cc); carry = hi + cc;
    lo = a3 * c; hi = MULHI(a3, c); cc = 0; out[3] = addc(lo, carry, &cc); carry = hi + cc;
    out[4] = carry;
}

/* reduce a 512-bit value t[0..7] mod p into r.
 *
 * NOTE: this MUST stay noinline. Apple's Metal compiler miscompiles the
 * secp256k1 point routines (e.g. point_add_mixed) when fe_reduce is inlined:
 * under the resulting 64-bit register pressure it spills an operand of a later
 * fe_mul incorrectly, corrupting Y3 while X3/Z3 stay correct. Keeping the
 * reduction out-of-line bounds the live 64-bit set at every call site and makes
 * the field math match the CPU reference bit-for-bit. (Verified on M5 Max.) */
NI void fe_reduce(thread fe *r, thread u64 t[8]) {
    u64 hiC[5];
    mul_scalar5(hiC, t[4], t[5], t[6], t[7], SECP_C);

    u64 m[5], cc = 0;
    m[0] = addc(t[0], hiC[0], &cc);
    m[1] = addc(t[1], hiC[1], &cc);
    m[2] = addc(t[2], hiC[2], &cc);
    m[3] = addc(t[3], hiC[3], &cc);
    m[4] = hiC[4] + cc;

    u64 hi = MULHI(m[4], SECP_C), lo = m[4] * SECP_C;
    cc = 0;
    r->n[0] = addc(m[0], lo, &cc);
    r->n[1] = addc(m[1], hi, &cc);
    r->n[2] = addc(m[2], 0, &cc);
    r->n[3] = addc(m[3], 0, &cc);
    fold_carry(r, cc);
    fe_cond_sub_p(r);
}

/* Comba (column) multiplication with a 3-word accumulator */
NI void fe_mul(thread fe *r, thread const fe *a, thread const fe *b) {
    u64 t[8];
    u64 c0 = 0, c1 = 0, c2 = 0;
    for (int col = 0; col < 7; col++) {
        int lo = col - 3; if (lo < 0) lo = 0;
        int hi = col; if (hi > 3) hi = 3;
        for (int i = lo; i <= hi; i++) {
            int j = col - i;
            u64 pl = a->n[i] * b->n[j];
            u64 ph = MULHI(a->n[i], b->n[j]);
            u64 carry = 0;
            c0 = addc(c0, pl, &carry);
            c1 = addc(c1, ph, &carry);
            c2 += carry;
        }
        t[col] = c0; c0 = c1; c1 = c2; c2 = 0;
    }
    t[7] = c0;
    fe_reduce(r, t);
}

/* Dedicated Comba squaring (10 muls instead of 16) */
NI void fe_sqr(thread fe *r, thread const fe *a) {
    u64 t[8];
    u64 c0 = 0, c1 = 0, c2 = 0, hi, lo, cr;
#define SQ_ACC(PH, PL) do { cr = 0; c0 = addc(c0, (PL), &cr); c1 = addc(c1, (PH), &cr); c2 += cr; } while (0)
    lo = a->n[0] * a->n[0]; hi = MULHI(a->n[0], a->n[0]); SQ_ACC(hi, lo);
    t[0] = c0; c0 = c1; c1 = c2; c2 = 0;
    lo = a->n[0] * a->n[1]; hi = MULHI(a->n[0], a->n[1]); SQ_ACC(hi, lo); SQ_ACC(hi, lo);
    t[1] = c0; c0 = c1; c1 = c2; c2 = 0;
    lo = a->n[0] * a->n[2]; hi = MULHI(a->n[0], a->n[2]); SQ_ACC(hi, lo); SQ_ACC(hi, lo);
    lo = a->n[1] * a->n[1]; hi = MULHI(a->n[1], a->n[1]); SQ_ACC(hi, lo);
    t[2] = c0; c0 = c1; c1 = c2; c2 = 0;
    lo = a->n[0] * a->n[3]; hi = MULHI(a->n[0], a->n[3]); SQ_ACC(hi, lo); SQ_ACC(hi, lo);
    lo = a->n[1] * a->n[2]; hi = MULHI(a->n[1], a->n[2]); SQ_ACC(hi, lo); SQ_ACC(hi, lo);
    t[3] = c0; c0 = c1; c1 = c2; c2 = 0;
    lo = a->n[1] * a->n[3]; hi = MULHI(a->n[1], a->n[3]); SQ_ACC(hi, lo); SQ_ACC(hi, lo);
    lo = a->n[2] * a->n[2]; hi = MULHI(a->n[2], a->n[2]); SQ_ACC(hi, lo);
    t[4] = c0; c0 = c1; c1 = c2; c2 = 0;
    lo = a->n[2] * a->n[3]; hi = MULHI(a->n[2], a->n[3]); SQ_ACC(hi, lo); SQ_ACC(hi, lo);
    t[5] = c0; c0 = c1; c1 = c2; c2 = 0;
    lo = a->n[3] * a->n[3]; hi = MULHI(a->n[3], a->n[3]); SQ_ACC(hi, lo);
    t[6] = c0; c0 = c1;
    t[7] = c0;
#undef SQ_ACC
    fe_reduce(r, t);
}

/* r = a^(p-2) mod p (modular inverse), LSB-first square-and-multiply */
NI void fe_inv(thread fe *r, thread const fe *a) {
    fe acc; fe_set_one(&acc);
    fe base = *a;
    u64 e[4];
    e[0] = (u64)0xFFFFFFFEFFFFFC2DUL;
    e[1] = SECP_PF; e[2] = SECP_PF; e[3] = SECP_PF;
    for (int i = 0; i < 256; i++) {
        if ((e[i >> 6] >> (i & 63)) & 1UL) {
            fe_mul(&acc, &acc, &base);
        }
        fe_sqr(&base, &base);
    }
    *r = acc;
}

inline int jac_is_inf(thread const jac *p) { return fe_is_zero(&p->Z); }

/* Jacobian doubling, a = 0 (dbl-2009-l) */
NI void point_double(thread jac *r, thread const jac *p) {
    if (jac_is_inf(p) || fe_is_zero(&p->Y)) {
        fe_set_zero(&r->X); fe_set_zero(&r->Y); fe_set_zero(&r->Z);
        return;
    }
    fe A, B, C, D, E, F, t0, t1, X3, Y3, Z3;
    fe_sqr(&A, &p->X);
    fe_sqr(&B, &p->Y);
    fe_sqr(&C, &B);
    fe_add(&t0, &p->X, &B);
    fe_sqr(&t0, &t0);
    fe_sub(&t0, &t0, &A);
    fe_sub(&t0, &t0, &C);
    fe_add(&D, &t0, &t0);
    fe_add(&E, &A, &A);
    fe_add(&E, &E, &A);
    fe_sqr(&F, &E);
    fe_add(&t0, &D, &D);
    fe_sub(&X3, &F, &t0);
    fe_sub(&t1, &D, &X3);
    fe_mul(&Y3, &E, &t1);
    fe_add(&t0, &C, &C);
    fe_add(&t0, &t0, &t0);
    fe_add(&t0, &t0, &t0);
    fe_sub(&Y3, &Y3, &t0);
    fe_mul(&Z3, &p->Y, &p->Z);
    fe_add(&Z3, &Z3, &Z3);
    r->X = X3; r->Y = Y3; r->Z = Z3;
}

/* Jacobian P + affine Q (madd-2007-bl) with edge cases */
NI void point_add_mixed(thread jac *r, thread const jac *p, thread const fe *x2, thread const fe *y2) {
    if (jac_is_inf(p)) {
        r->X = *x2; r->Y = *y2; fe_set_one(&r->Z);
        return;
    }
    fe Z1Z1, U2, S2, H, HH, I, J, rr, V, t0, t1, X3, Y3, Z3;
    fe_sqr(&Z1Z1, &p->Z);
    fe_mul(&U2, x2, &Z1Z1);
    fe_mul(&t0, y2, &p->Z);
    fe_mul(&S2, &t0, &Z1Z1);
    fe_sub(&H, &U2, &p->X);
    fe_sub(&rr, &S2, &p->Y);
    if (fe_is_zero(&H)) {
        if (fe_is_zero(&rr)) {
            point_double(r, p);
            return;
        }
        fe_set_zero(&r->X); fe_set_zero(&r->Y); fe_set_zero(&r->Z);
        return;
    }
    fe_add(&rr, &rr, &rr);
    fe_sqr(&HH, &H);
    fe_add(&I, &HH, &HH);
    fe_add(&I, &I, &I);
    fe_mul(&J, &H, &I);
    fe_mul(&V, &p->X, &I);
    fe_sqr(&X3, &rr);
    fe_sub(&X3, &X3, &J);
    fe_add(&t0, &V, &V);
    fe_sub(&X3, &X3, &t0);
    fe_sub(&t1, &V, &X3);
    fe_mul(&Y3, &rr, &t1);
    fe_mul(&t0, &p->Y, &J);
    fe_add(&t0, &t0, &t0);
    fe_sub(&Y3, &Y3, &t0);
    fe_add(&t0, &p->Z, &H);
    fe_sqr(&t0, &t0);
    fe_sub(&t0, &t0, &Z1Z1);
    fe_sub(&Z3, &t0, &HH);
    r->X = X3; r->Y = Y3; r->Z = Z3;
}

NI void jac_to_affine(thread const jac *p, thread fe *x, thread fe *y) {
    fe zinv, zinv2, zinv3;
    fe_inv(&zinv, &p->Z);
    fe_sqr(&zinv2, &zinv);
    fe_mul(&zinv3, &zinv2, &zinv);
    fe_mul(x, &p->X, &zinv2);
    fe_mul(y, &p->Y, &zinv3);
}

/* load a field element from 32 little-endian bytes (thread memory) */
inline void fe_from_le32(thread const u8 *in, thread fe *out) {
    for (int i = 0; i < 4; i++) {
        u64 w = 0;
        for (int j = 0; j < 8; j++) w |= ((u64)in[i * 8 + j]) << (8 * j);
        out->n[i] = w;
    }
}

/* serialize a field element to 32 big-endian bytes */
inline void fe_to_be32(thread const fe *a, thread u8 *out) {
    for (int limb = 0; limb < 4; limb++) {
        u64 w = a->n[3 - limb];
        for (int j = 0; j < 8; j++) out[limb * 8 + j] = (u8)(w >> (8 * (7 - j)));
    }
}

/* ---- Keccak-256 (legacy padding, single block) ----
 *
 * Bit-interleaved representation: each 64-bit lane is held as a uint2 (.x =
 * even bits, .y = odd bits), so a 64-bit rotation becomes a pair of native
 * 32-bit rotations (irol). Apple GPUs have no 64-bit rotate instruction, so
 * this is the standard win for Keccak on 32-bit-native hardware — the round
 * structure (theta/rho-pi/chi/iota) is otherwise identical to the plain
 * version. */

inline uint rotl32(uint x, uint n) { return n == 0 ? x : ((x << n) | (x >> (32 - n))); }

/* ROL64 of an interleaved lane by r:
 *   r even (2m): rotate each half by m
 *   r odd (2m+1): swap halves; even<-odd rot m+1, odd<-even rot m  */
inline uint2 irol(uint2 a, int r) {
    int m = r >> 1;
    return (r & 1) ? uint2(rotl32(a.y, m + 1), rotl32(a.x, m))
                   : uint2(rotl32(a.x, m), rotl32(a.y, m));
}

/* compact the even-position bits of x into the low 32 bits (Morton encode) */
inline uint even_bits(u64 x) {
    x &= 0x5555555555555555UL;
    x = (x | (x >> 1)) & 0x3333333333333333UL;
    x = (x | (x >> 2)) & 0x0F0F0F0F0F0F0F0FUL;
    x = (x | (x >> 4)) & 0x00FF00FF00FF00FFUL;
    x = (x | (x >> 8)) & 0x0000FFFF0000FFFFUL;
    x = (x | (x >> 16)) & 0x00000000FFFFFFFFUL;
    return (uint)x;
}
inline uint2 il64(u64 x) { return uint2(even_bits(x), even_bits(x >> 1)); }

/* spread 32 bits to even positions (Morton decode) */
inline u64 spread(uint v) {
    u64 x = v;
    x = (x | (x << 16)) & 0x0000FFFF0000FFFFUL;
    x = (x | (x << 8)) & 0x00FF00FF00FF00FFUL;
    x = (x | (x << 4)) & 0x0F0F0F0F0F0F0F0FUL;
    x = (x | (x << 2)) & 0x3333333333333333UL;
    x = (x | (x << 1)) & 0x5555555555555555UL;
    return x;
}
inline u64 dil(uint2 a) { return spread(a.x) | (spread(a.y) << 1); }

NI void keccak_f1600(thread uint2 st[25]) {
    /* round constants, bit-interleaved (.x even bits, .y odd bits) */
    const uint2 RC[24] = {
        uint2(0x00000001u, 0x00000000u), uint2(0x00000000u, 0x00000089u),
        uint2(0x00000000u, 0x8000008bu), uint2(0x00000000u, 0x80008080u),
        uint2(0x00000001u, 0x0000008bu), uint2(0x00000001u, 0x00008000u),
        uint2(0x00000001u, 0x80008088u), uint2(0x00000001u, 0x80000082u),
        uint2(0x00000000u, 0x0000000bu), uint2(0x00000000u, 0x0000000au),
        uint2(0x00000001u, 0x00008082u), uint2(0x00000000u, 0x00008003u),
        uint2(0x00000001u, 0x0000808bu), uint2(0x00000001u, 0x8000000bu),
        uint2(0x00000001u, 0x8000008au), uint2(0x00000001u, 0x80000081u),
        uint2(0x00000000u, 0x80000081u), uint2(0x00000000u, 0x80000008u),
        uint2(0x00000000u, 0x00000083u), uint2(0x00000000u, 0x80008003u),
        uint2(0x00000001u, 0x80008088u), uint2(0x00000000u, 0x80000088u),
        uint2(0x00000001u, 0x00008000u), uint2(0x00000000u, 0x80008082u)};
    uint2 bc[5], t;
    for (int round = 0; round < 24; ++round) {
        bc[0] = st[0] ^ st[5] ^ st[10] ^ st[15] ^ st[20];
        bc[1] = st[1] ^ st[6] ^ st[11] ^ st[16] ^ st[21];
        bc[2] = st[2] ^ st[7] ^ st[12] ^ st[17] ^ st[22];
        bc[3] = st[3] ^ st[8] ^ st[13] ^ st[18] ^ st[23];
        bc[4] = st[4] ^ st[9] ^ st[14] ^ st[19] ^ st[24];

        t = bc[4] ^ irol(bc[1], 1); st[0] ^= t; st[5] ^= t; st[10] ^= t; st[15] ^= t; st[20] ^= t;
        t = bc[0] ^ irol(bc[2], 1); st[1] ^= t; st[6] ^= t; st[11] ^= t; st[16] ^= t; st[21] ^= t;
        t = bc[1] ^ irol(bc[3], 1); st[2] ^= t; st[7] ^= t; st[12] ^= t; st[17] ^= t; st[22] ^= t;
        t = bc[2] ^ irol(bc[4], 1); st[3] ^= t; st[8] ^= t; st[13] ^= t; st[18] ^= t; st[23] ^= t;
        t = bc[3] ^ irol(bc[0], 1); st[4] ^= t; st[9] ^= t; st[14] ^= t; st[19] ^= t; st[24] ^= t;

        t = st[1];
        st[1] = irol(st[6], 44);
        st[6] = irol(st[9], 20);
        st[9] = irol(st[22], 61);
        st[22] = irol(st[14], 39);
        st[14] = irol(st[20], 18);
        st[20] = irol(st[2], 62);
        st[2] = irol(st[12], 43);
        st[12] = irol(st[13], 25);
        st[13] = irol(st[19], 8);
        st[19] = irol(st[23], 56);
        st[23] = irol(st[15], 41);
        st[15] = irol(st[4], 27);
        st[4] = irol(st[24], 14);
        st[24] = irol(st[21], 2);
        st[21] = irol(st[8], 55);
        st[8] = irol(st[16], 45);
        st[16] = irol(st[5], 36);
        st[5] = irol(st[3], 28);
        st[3] = irol(st[18], 21);
        st[18] = irol(st[17], 15);
        st[17] = irol(st[11], 10);
        st[11] = irol(st[7], 6);
        st[7] = irol(st[10], 3);
        st[10] = irol(t, 1);

        bc[0] = st[0]; bc[1] = st[1]; bc[2] = st[2]; bc[3] = st[3]; bc[4] = st[4];
        st[0] = bc[0] ^ (~bc[1] & bc[2]); st[1] = bc[1] ^ (~bc[2] & bc[3]);
        st[2] = bc[2] ^ (~bc[3] & bc[4]); st[3] = bc[3] ^ (~bc[4] & bc[0]);
        st[4] = bc[4] ^ (~bc[0] & bc[1]);
        bc[0] = st[5]; bc[1] = st[6]; bc[2] = st[7]; bc[3] = st[8]; bc[4] = st[9];
        st[5] = bc[0] ^ (~bc[1] & bc[2]); st[6] = bc[1] ^ (~bc[2] & bc[3]);
        st[7] = bc[2] ^ (~bc[3] & bc[4]); st[8] = bc[3] ^ (~bc[4] & bc[0]);
        st[9] = bc[4] ^ (~bc[0] & bc[1]);
        bc[0] = st[10]; bc[1] = st[11]; bc[2] = st[12]; bc[3] = st[13]; bc[4] = st[14];
        st[10] = bc[0] ^ (~bc[1] & bc[2]); st[11] = bc[1] ^ (~bc[2] & bc[3]);
        st[12] = bc[2] ^ (~bc[3] & bc[4]); st[13] = bc[3] ^ (~bc[4] & bc[0]);
        st[14] = bc[4] ^ (~bc[0] & bc[1]);
        bc[0] = st[15]; bc[1] = st[16]; bc[2] = st[17]; bc[3] = st[18]; bc[4] = st[19];
        st[15] = bc[0] ^ (~bc[1] & bc[2]); st[16] = bc[1] ^ (~bc[2] & bc[3]);
        st[17] = bc[2] ^ (~bc[3] & bc[4]); st[18] = bc[3] ^ (~bc[4] & bc[0]);
        st[19] = bc[4] ^ (~bc[0] & bc[1]);
        bc[0] = st[20]; bc[1] = st[21]; bc[2] = st[22]; bc[3] = st[23]; bc[4] = st[24];
        st[20] = bc[0] ^ (~bc[1] & bc[2]); st[21] = bc[1] ^ (~bc[2] & bc[3]);
        st[22] = bc[2] ^ (~bc[3] & bc[4]); st[23] = bc[3] ^ (~bc[4] & bc[0]);
        st[24] = bc[4] ^ (~bc[0] & bc[1]);

        st[0] ^= RC[round];
    }
}

/* keccak256 of a single-block message (len <= 135) into out[0..31]. */
NI void keccak256(thread const u8 *in, int len, thread u8 *out) {
    /* assemble rate words, pad, then interleave into the bit-split state */
    u64 lanes[17];
    for (int i = 0; i < 17; i++) lanes[i] = 0;
    for (int i = 0; i < len; i++)
        lanes[i >> 3] |= ((u64)in[i]) << (8 * (i & 7));
    lanes[len >> 3] |= ((u64)0x01) << (8 * (len & 7));
    lanes[16] |= ((u64)0x80) << 56;

    uint2 st[25];
    for (int i = 0; i < 17; i++) st[i] = il64(lanes[i]);
    for (int i = 17; i < 25; i++) st[i] = uint2(0u, 0u);

    keccak_f1600(st);

    for (int i = 0; i < 4; i++) {
        u64 w = dil(st[i]);
        for (int j = 0; j < 8; j++) out[i * 8 + j] = (u8)(w >> (8 * j));
    }
}

/* Jacobian point Q_base + k*D via the comb table (no inverse). */
NI void nick_acc_for_k(device const u8 *d_table, thread const u8 *q_base, u64 k, thread jac *out) {
    jac acc;
    fe_set_zero(&acc.X); fe_set_zero(&acc.Y); fe_set_zero(&acc.Z);

    u8 buf[64];
    for (int w = 0; w < 8; w++) {
        u32 b = (u32)((k >> (8 * w)) & 0xff);
        if (b == 0) continue;
        device const u8 *e = d_table + ((w * 256 + (int)b) * 64);
        for (int i = 0; i < 64; i++) buf[i] = e[i];
        fe tx, ty;
        fe_from_le32(buf, &tx);
        fe_from_le32(buf + 32, &ty);
        point_add_mixed(&acc, &acc, &tx, &ty);
    }
    fe qbx, qby;
    fe_from_le32(q_base, &qbx);
    fe_from_le32(q_base + 32, &qby);
    point_add_mixed(&acc, &acc, &qbx, &qby);
    *out = acc;
}

/* Contract deployment address from an affine public key. */
NI void affine_to_addr(thread const fe *ax, thread const fe *ay, thread u8 *out_addr) {
    u8 pub[64];
    fe_to_be32(ax, pub);
    fe_to_be32(ay, pub + 32);

    u8 h[32];
    keccak256(pub, 64, h);

    u8 rlp[23];
    rlp[0] = 0xd6;
    rlp[1] = 0x94;
    for (int i = 0; i < 20; i++) rlp[2 + i] = h[12 + i];
    rlp[22] = 0x80;

    u8 h2[32];
    keccak256(rlp, 23, h2);
    for (int i = 0; i < 20; i++) out_addr[i] = h2[12 + i];
}

inline int nick_match(thread const u8 *addr, thread const u8 *prefix, int prefix_len,
                      thread const u8 *suffix, int suffix_len) {
    for (int i = 0; i < prefix_len; i++)
        if (addr[i] != prefix[i]) return 0;
    for (int i = 0; i < suffix_len; i++)
        if (addr[20 - suffix_len + i] != suffix[i]) return 0;
    return 1;
}

#if NICK_ITERS > 256
#error "NICK_ITERS must be <= 256 (uses comb-table window 0 for i*D)"
#endif

/* claim the winner slot once, via atomic compare-exchange on `found`. */
inline void claim(device atomic_int *found,
                  device u8 *result_address, device u64 *result_nonce,
                  thread const u8 *addr, u64 nonce) {
    int expected = 0;
    if (atomic_compare_exchange_weak_explicit(found, &expected, 1,
            memory_order_relaxed, memory_order_relaxed)) {
        for (int b = 0; b < 20; b++) result_address[b] = addr[b];
        *result_nonce = nonce;
    }
}

/*
 * Process NICK_ITERS consecutive candidates [base, base+NICK_ITERS) in affine
 * coordinates with a single batched field inversion (Montgomery's trick).
 */
kernel void mine_nick(
    device const u8       *d_table        [[buffer(0)]],
    device const u8       *q_base         [[buffer(1)]],
    device const u8       *prefix         [[buffer(2)]],
    constant int          &prefix_len     [[buffer(3)]],
    device const u8       *suffix         [[buffer(4)]],
    constant int          &suffix_len     [[buffer(5)]],
    constant u64          &start_nonce    [[buffer(6)]],
    device u8             *result_address [[buffer(7)]],
    device u64            *result_nonce   [[buffer(8)]],
    device atomic_int     *found          [[buffer(9)]],
    uint                   gid            [[thread_position_in_grid]]) {

    if (atomic_load_explicit(found, memory_order_relaxed)) return;

    u64 base = start_nonce + (u64)gid * (u64)NICK_ITERS;

    /* private copies of the small constant inputs */
    u8 qb[64];
    for (int i = 0; i < 64; i++) qb[i] = q_base[i];
    u8 pfx[20], sfx[20];
    for (int i = 0; i < prefix_len; i++) pfx[i] = prefix[i];
    for (int i = 0; i < suffix_len; i++) sfx[i] = suffix[i];

    /* P0 = Q_base + base*D, converted to affine (the run's only point inverse). */
    jac P0j;
    nick_acc_for_k(d_table, qb, base, &P0j);
    fe px, py;
    jac_to_affine(&P0j, &px, &py);

    const int m = NICK_ITERS - 1;
    fe pref[NICK_ITERS];
    u8 buf[64];

    /* forward: pref[j] = prod of delta_0..delta_{j-1}; delta_j = table[0][j+1].x - px */
    fe acc;
    fe_set_one(&acc);
    for (int j = 0; j < m; j++) {
        device const u8 *e = d_table + ((j + 1) * 64);
        for (int b = 0; b < 32; b++) buf[b] = e[b];
        fe mx, delta;
        fe_from_le32(buf, &mx);
        fe_sub(&delta, &mx, &px);
        pref[j] = acc;
        fe_mul(&acc, &acc, &delta);
    }
    fe inv;
    fe_inv(&inv, &acc);

    /* candidate 0 = P0 */
    {
        u8 addr[20];
        affine_to_addr(&px, &py, addr);
        if (nick_match(addr, pfx, prefix_len, sfx, suffix_len))
            claim(found, result_address, result_nonce, addr, base);
    }

    /* candidates N-1..1 */
    for (int j = m - 1; j >= 0; j--) {
        device const u8 *e = d_table + ((j + 1) * 64);
        for (int b = 0; b < 64; b++) buf[b] = e[b];
        fe mx, my, delta, invj;
        fe_from_le32(buf, &mx);
        fe_from_le32(buf + 32, &my);
        fe_sub(&delta, &mx, &px);
        fe_mul(&invj, &inv, &pref[j]);
        fe_mul(&inv, &inv, &delta);

        fe lam, tt, xi, yi;
        fe_sub(&tt, &my, &py);
        fe_mul(&lam, &tt, &invj);
        fe_sqr(&tt, &lam);
        fe_sub(&xi, &tt, &px);
        fe_sub(&xi, &xi, &mx);
        fe_sub(&tt, &px, &xi);
        fe_mul(&yi, &lam, &tt);
        fe_sub(&yi, &yi, &py);

        u8 addr[20];
        affine_to_addr(&xi, &yi, addr);
        if (nick_match(addr, pfx, prefix_len, sfx, suffix_len))
            claim(found, result_address, result_nonce, addr, base + (u64)(j + 1));
    }
}
