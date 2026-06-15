/*
 * OpenCL entry kernel for Nick's-method vanity mining.
 * nick_lib.cl is prepended by the host (passed as a separate program source
 * string), so all device helpers are already declared here. Each work item
 * processes NICK_ITERS consecutive candidates (see nick_mine_run).
 */

__kernel void mine_nick(
    __global const u8 *d_table,   /* comb table for D (128 KB) */
    __global const u8 *q_base,    /* 64 bytes: X||Y little-endian */
    __global const u8 *prefix,
    const int prefix_len,
    __global const u8 *suffix,
    const int suffix_len,
    const u64 start_nonce,
    __global u8 *result_address,  /* 20 bytes */
    __global u64 *result_nonce,   /* k of the winning candidate */
    __global volatile int *found) {
    if (*found) return;

    u64 base = start_nonce + (u64)get_global_id(0) * (u64)NICK_ITERS;

    u8 qb[64];
    for (int i = 0; i < 64; i++) qb[i] = q_base[i];
    u8 pfx[20], sfx[20];
    for (int i = 0; i < prefix_len; i++) pfx[i] = prefix[i];
    for (int i = 0; i < suffix_len; i++) sfx[i] = suffix[i];

    nick_mine_run(d_table, qb, base, pfx, prefix_len, sfx, suffix_len,
                  result_address, result_nonce, found);
}
