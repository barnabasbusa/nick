/*
 * nick_lib.cl - shared secp256k1 + keccak256 device code for the Nick's-method
 * vanity miner. Compiled both as OpenCL (concatenated before nick.cl) and as
 * CUDA (#included by nick.cu).
 *
 * The field/point arithmetic is a verbatim translation of the Go reference in
 * miner/secp_ref_test.go, which is validated against go-ethereum's big.Int
 * curve and crypto.Ecrecover. Field elements are 4 little-endian 64-bit limbs.
 * Reduction uses p = 2^256 - C with C = 0x1000003D1.
 */

#if defined(__OPENCL_VERSION__) || defined(__OPENCL_C_VERSION__)
typedef ulong u64;
typedef uint u32;
typedef uchar u8;
#define DEV inline
#define GLOBAL __global
#define MULHI(a, b) mul_hi((a), (b))
#else
/* CUDA: u64/u32/u8 avoid clashing with <sys/types.h>'s ulong/uint */
typedef unsigned long long u64;
typedef unsigned char u8;
typedef unsigned int u32;
#define DEV __device__ __forceinline__
#define GLOBAL
#define MULHI(a, b) __umul64hi((a), (b))
#endif

#define SECP_C ((u64)0x1000003D1UL)
#define SECP_P0 ((u64)0xFFFFFFFEFFFFFC2FUL)
#define SECP_PF ((u64)0xFFFFFFFFFFFFFFFFUL)

typedef struct {
    u64 n[4];
} fe;

typedef struct {
    fe X, Y, Z; /* Jacobian; Z == 0 means point at infinity */
} jac;

/* add with carry: returns a+b+(*carry), updates *carry to the carry-out */
DEV u64 addc(u64 a, u64 b, u64 *carry) {
    u64 s = a + b;
    u64 c1 = (u64)(s < a);
    u64 s2 = s + *carry;
    u64 c2 = (u64)(s2 < s);
    *carry = c1 + c2;
    return s2;
}

/* subtract with borrow: returns a-b-(*borrow), updates *borrow */
DEV u64 subb(u64 a, u64 b, u64 *borrow) {
    u64 d = a - b;
    u64 b1 = (u64)(a < b);
    u64 d2 = d - *borrow;
    u64 b2 = (u64)(d < *borrow);
    *borrow = b1 + b2;
    return d2;
}

DEV int fe_is_zero(const fe *a) { return (a->n[0] | a->n[1] | a->n[2] | a->n[3]) == 0; }
DEV void fe_set_zero(fe *a) { a->n[0] = a->n[1] = a->n[2] = a->n[3] = 0; }
DEV void fe_set_one(fe *a) { a->n[0] = 1; a->n[1] = a->n[2] = a->n[3] = 0; }

DEV int fe_ge_p(const fe *a) {
    if (a->n[3] != SECP_PF) return a->n[3] > SECP_PF;
    if (a->n[2] != SECP_PF) return a->n[2] > SECP_PF;
    if (a->n[1] != SECP_PF) return a->n[1] > SECP_PF;
    return a->n[0] >= SECP_P0;
}

DEV void fe_cond_sub_p(fe *r) {
    while (fe_ge_p(r)) {
        u64 br = 0;
        r->n[0] = subb(r->n[0], SECP_P0, &br);
        r->n[1] = subb(r->n[1], SECP_PF, &br);
        r->n[2] = subb(r->n[2], SECP_PF, &br);
        r->n[3] = subb(r->n[3], SECP_PF, &br);
    }
}

/* fold a carry out of bit 256 back in: 2^256 ≡ C (mod p) */
DEV void fold_carry(fe *r, u64 carry) {
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

DEV void fe_add(fe *r, const fe *a, const fe *b) {
    u64 c = 0;
    r->n[0] = addc(a->n[0], b->n[0], &c);
    r->n[1] = addc(a->n[1], b->n[1], &c);
    r->n[2] = addc(a->n[2], b->n[2], &c);
    r->n[3] = addc(a->n[3], b->n[3], &c);
    fold_carry(r, c);
    fe_cond_sub_p(r);
}

DEV void fe_sub(fe *r, const fe *a, const fe *b) {
    u64 br = 0;
    r->n[0] = subb(a->n[0], b->n[0], &br);
    r->n[1] = subb(a->n[1], b->n[1], &br);
    r->n[2] = subb(a->n[2], b->n[2], &br);
    r->n[3] = subb(a->n[3], b->n[3], &br);
    if (br != 0) {
        /* add p == subtract C (mod 2^256); cannot borrow again */
        u64 b2 = 0;
        r->n[0] = subb(r->n[0], SECP_C, &b2);
        r->n[1] = subb(r->n[1], 0, &b2);
        r->n[2] = subb(r->n[2], 0, &b2);
        r->n[3] = subb(r->n[3], 0, &b2);
    }
}

/* (a0..a3) * c -> 5 limbs out, with c small (< 2^34) */
DEV void mul_scalar5(u64 out[5], u64 a0, u64 a1, u64 a2, u64 a3, u64 c) {
    u64 carry, cc, hi, lo;
    lo = a0 * c; hi = MULHI(a0, c); out[0] = lo; carry = hi;
    lo = a1 * c; hi = MULHI(a1, c); cc = 0; out[1] = addc(lo, carry, &cc); carry = hi + cc;
    lo = a2 * c; hi = MULHI(a2, c); cc = 0; out[2] = addc(lo, carry, &cc); carry = hi + cc;
    lo = a3 * c; hi = MULHI(a3, c); cc = 0; out[3] = addc(lo, carry, &cc); carry = hi + cc;
    out[4] = carry;
}

/* reduce a 512-bit value t[0..7] mod p into r */
DEV void fe_reduce(fe *r, u64 t[8]) {
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
DEV void fe_mul(fe *r, const fe *a, const fe *b) {
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

DEV void fe_sqr(fe *r, const fe *a) { fe_mul(r, a, a); }

/* r = a^(p-2) mod p (modular inverse), LSB-first square-and-multiply */
DEV void fe_inv(fe *r, const fe *a) {
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

DEV int jac_is_inf(const jac *p) { return fe_is_zero(&p->Z); }

/* Jacobian doubling, a = 0 (dbl-2009-l) */
DEV void point_double(jac *r, const jac *p) {
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
    fe_add(&D, &t0, &t0);          /* D = 2*((X+B)^2 - A - C) */
    fe_add(&E, &A, &A);
    fe_add(&E, &E, &A);            /* E = 3*A */
    fe_sqr(&F, &E);
    fe_add(&t0, &D, &D);
    fe_sub(&X3, &F, &t0);          /* X3 = F - 2D */
    fe_sub(&t1, &D, &X3);
    fe_mul(&Y3, &E, &t1);
    fe_add(&t0, &C, &C);
    fe_add(&t0, &t0, &t0);
    fe_add(&t0, &t0, &t0);         /* t0 = 8C */
    fe_sub(&Y3, &Y3, &t0);
    fe_mul(&Z3, &p->Y, &p->Z);
    fe_add(&Z3, &Z3, &Z3);         /* Z3 = 2YZ */
    r->X = X3; r->Y = Y3; r->Z = Z3;
}

/* Jacobian P + affine Q (madd-2007-bl) with edge cases */
DEV void point_add_mixed(jac *r, const jac *p, const fe *x2, const fe *y2) {
    if (jac_is_inf(p)) {
        r->X = *x2; r->Y = *y2; fe_set_one(&r->Z);
        return;
    }
    fe Z1Z1, U2, S2, H, HH, I, J, rr, V, t0, t1, X3, Y3, Z3;
    fe_sqr(&Z1Z1, &p->Z);
    fe_mul(&U2, x2, &Z1Z1);
    fe_mul(&t0, y2, &p->Z);
    fe_mul(&S2, &t0, &Z1Z1);       /* S2 = Y2*Z1*Z1Z1 */
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
    fe_add(&rr, &rr, &rr);         /* r = 2*(S2 - Y1) */
    fe_sqr(&HH, &H);
    fe_add(&I, &HH, &HH);
    fe_add(&I, &I, &I);            /* I = 4*HH */
    fe_mul(&J, &H, &I);
    fe_mul(&V, &p->X, &I);
    fe_sqr(&X3, &rr);
    fe_sub(&X3, &X3, &J);
    fe_add(&t0, &V, &V);
    fe_sub(&X3, &X3, &t0);         /* X3 = r^2 - J - 2V */
    fe_sub(&t1, &V, &X3);
    fe_mul(&Y3, &rr, &t1);
    fe_mul(&t0, &p->Y, &J);
    fe_add(&t0, &t0, &t0);
    fe_sub(&Y3, &Y3, &t0);         /* Y3 = r*(V-X3) - 2*Y1*J */
    fe_add(&t0, &p->Z, &H);
    fe_sqr(&t0, &t0);
    fe_sub(&t0, &t0, &Z1Z1);
    fe_sub(&Z3, &t0, &HH);         /* Z3 = (Z1+H)^2 - Z1Z1 - HH */
    r->X = X3; r->Y = Y3; r->Z = Z3;
}

DEV void jac_to_affine(const jac *p, fe *x, fe *y) {
    fe zinv, zinv2, zinv3;
    fe_inv(&zinv, &p->Z);
    fe_sqr(&zinv2, &zinv);
    fe_mul(&zinv3, &zinv2, &zinv);
    fe_mul(x, &p->X, &zinv2);
    fe_mul(y, &p->Y, &zinv3);
}

/* load a field element from 32 little-endian bytes (private memory) */
DEV void fe_from_le32(const u8 *in, fe *out) {
    for (int i = 0; i < 4; i++) {
        u64 w = 0;
        for (int j = 0; j < 8; j++) w |= ((u64)in[i * 8 + j]) << (8 * j);
        out->n[i] = w;
    }
}

/* serialize a field element to 32 big-endian bytes */
DEV void fe_to_be32(const fe *a, u8 *out) {
    for (int limb = 0; limb < 4; limb++) {
        u64 w = a->n[3 - limb];
        for (int j = 0; j < 8; j++) out[limb * 8 + j] = (u8)(w >> (8 * (7 - j)));
    }
}

/* ---- Keccak-256 (legacy padding, single block) ---- */

DEV u64 rotl64(u64 x, u32 n) { return (x << n) | (x >> (64 - n)); }

DEV void keccak_f1600(u64 st[25]) {
    const u64 RC[24] = {
        0x0000000000000001UL, 0x0000000000008082UL, 0x800000000000808aUL, 0x8000000080008000UL,
        0x000000000000808bUL, 0x0000000080000001UL, 0x8000000080008081UL, 0x8000000000008009UL,
        0x000000000000008aUL, 0x0000000000000088UL, 0x0000000080008009UL, 0x000000008000000aUL,
        0x000000008000808bUL, 0x800000000000008bUL, 0x8000000000008089UL, 0x8000000000008003UL,
        0x8000000000008002UL, 0x8000000000000080UL, 0x000000000000800aUL, 0x800000008000000aUL,
        0x8000000080008081UL, 0x8000000000008080UL, 0x0000000080000001UL, 0x8000000080008008UL};
    u64 bc[5], t;
    for (int round = 0; round < 24; ++round) {
        bc[0] = st[0] ^ st[5] ^ st[10] ^ st[15] ^ st[20];
        bc[1] = st[1] ^ st[6] ^ st[11] ^ st[16] ^ st[21];
        bc[2] = st[2] ^ st[7] ^ st[12] ^ st[17] ^ st[22];
        bc[3] = st[3] ^ st[8] ^ st[13] ^ st[18] ^ st[23];
        bc[4] = st[4] ^ st[9] ^ st[14] ^ st[19] ^ st[24];

        t = bc[4] ^ rotl64(bc[1], 1); st[0] ^= t; st[5] ^= t; st[10] ^= t; st[15] ^= t; st[20] ^= t;
        t = bc[0] ^ rotl64(bc[2], 1); st[1] ^= t; st[6] ^= t; st[11] ^= t; st[16] ^= t; st[21] ^= t;
        t = bc[1] ^ rotl64(bc[3], 1); st[2] ^= t; st[7] ^= t; st[12] ^= t; st[17] ^= t; st[22] ^= t;
        t = bc[2] ^ rotl64(bc[4], 1); st[3] ^= t; st[8] ^= t; st[13] ^= t; st[18] ^= t; st[23] ^= t;
        t = bc[3] ^ rotl64(bc[0], 1); st[4] ^= t; st[9] ^= t; st[14] ^= t; st[19] ^= t; st[24] ^= t;

        t = st[1];
        st[1] = rotl64(st[6], 44);
        st[6] = rotl64(st[9], 20);
        st[9] = rotl64(st[22], 61);
        st[22] = rotl64(st[14], 39);
        st[14] = rotl64(st[20], 18);
        st[20] = rotl64(st[2], 62);
        st[2] = rotl64(st[12], 43);
        st[12] = rotl64(st[13], 25);
        st[13] = rotl64(st[19], 8);
        st[19] = rotl64(st[23], 56);
        st[23] = rotl64(st[15], 41);
        st[15] = rotl64(st[4], 27);
        st[4] = rotl64(st[24], 14);
        st[24] = rotl64(st[21], 2);
        st[21] = rotl64(st[8], 55);
        st[8] = rotl64(st[16], 45);
        st[16] = rotl64(st[5], 36);
        st[5] = rotl64(st[3], 28);
        st[3] = rotl64(st[18], 21);
        st[18] = rotl64(st[17], 15);
        st[17] = rotl64(st[11], 10);
        st[11] = rotl64(st[7], 6);
        st[7] = rotl64(st[10], 3);
        st[10] = rotl64(t, 1);

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

/* keccak256 of a single block message (len <= 135) into out[0..31] */
DEV void keccak256(const u8 *in, int len, u8 *out) {
    u8 block[136];
    for (int i = 0; i < 136; i++) block[i] = 0;
    for (int i = 0; i < len; i++) block[i] = in[i];
    block[len] ^= 0x01;
    block[135] ^= 0x80;

    u64 st[25];
    for (int i = 0; i < 25; i++) st[i] = 0;
    for (int i = 0; i < 17; i++) {
        u64 w = 0;
        for (int j = 0; j < 8; j++) w |= ((u64)block[i * 8 + j]) << (8 * j);
        st[i] = w;
    }
    keccak_f1600(st);
    for (int i = 0; i < 4; i++) {
        u64 w = st[i];
        for (int j = 0; j < 8; j++) out[i * 8 + j] = (u8)(w >> (8 * j));
    }
}

/*
 * Core per-candidate computation shared by both backends.
 * All inputs are private-memory copies (caller copies from global first).
 *   d_table  : 128KB comb table  (combWindows*combEntries*64 bytes)
 *   q_base   : 64 bytes  (X||Y little-endian)
 * Produces the 20-byte contract deployment address for candidate k.
 */
DEV void nick_address_for_k(
    GLOBAL const u8 *d_table, const u8 *q_base, u64 k, u8 out_addr[20]) {
    jac acc;
    fe_set_zero(&acc.X); fe_set_zero(&acc.Y); fe_set_zero(&acc.Z); /* infinity */

    u8 buf[64];
    for (int w = 0; w < 8; w++) {
        u32 b = (u32)((k >> (8 * w)) & 0xff);
        if (b == 0) continue;
        /* table[w][b] at offset (w*256 + b) * 64 */
        GLOBAL const u8 *e = d_table + ((w * 256 + (int)b) * 64);
        for (int i = 0; i < 64; i++) buf[i] = e[i];
        fe tx, ty;
        fe_from_le32(buf, &tx);
        fe_from_le32(buf + 32, &ty);
        point_add_mixed(&acc, &acc, &tx, &ty);
    }

    /* + Q_base */
    fe qbx, qby;
    fe_from_le32(q_base, &qbx);
    fe_from_le32(q_base + 32, &qby);
    point_add_mixed(&acc, &acc, &qbx, &qby);

    fe ax, ay;
    jac_to_affine(&acc, &ax, &ay);

    u8 pub[64];
    fe_to_be32(&ax, pub);
    fe_to_be32(&ay, pub + 32);

    u8 h[32];
    keccak256(pub, 64, h);          /* sender = h[12:32] */

    u8 rlp[23];
    rlp[0] = 0xd6; rlp[1] = 0x94;
    for (int i = 0; i < 20; i++) rlp[2 + i] = h[12 + i];
    rlp[22] = 0x80;

    u8 h2[32];
    keccak256(rlp, 23, h2);         /* contract addr = h2[12:32] */
    for (int i = 0; i < 20; i++) out_addr[i] = h2[12 + i];
}

/* Returns 1 if addr matches prefix (first prefix_len bytes) AND suffix
 * (last suffix_len bytes). */
DEV int nick_match(const u8 *addr, const u8 *prefix, int prefix_len,
                   const u8 *suffix, int suffix_len) {
    for (int i = 0; i < prefix_len; i++)
        if (addr[i] != prefix[i]) return 0;
    for (int i = 0; i < suffix_len; i++)
        if (addr[20 - suffix_len + i] != suffix[i]) return 0;
    return 1;
}
