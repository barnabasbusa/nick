/*
 * CUDA entry kernel for Nick's-method vanity mining.
 * Includes the shared device library (same math as the OpenCL build). Each
 * thread processes NICK_ITERS consecutive candidates (see nick_mine_run).
 */

#include "nick_lib.cl"

extern "C" __global__ void mine_nick(
    const u8 *__restrict__ d_table,
    const u8 *__restrict__ q_base,
    const u8 *__restrict__ prefix,
    int prefix_len,
    const u8 *__restrict__ suffix,
    int suffix_len,
    unsigned long long start_nonce,
    u8 *__restrict__ result_address,
    unsigned long long *__restrict__ result_nonce,
    int *found) {
    if (*found) return;

    u64 base = start_nonce + (u64)(blockIdx.x * blockDim.x + threadIdx.x) * (u64)NICK_ITERS;

    u8 qb[64];
    for (int i = 0; i < 64; i++) qb[i] = q_base[i];
    u8 pfx[20], sfx[20];
    for (int i = 0; i < prefix_len; i++) pfx[i] = prefix[i];
    for (int i = 0; i < suffix_len; i++) sfx[i] = suffix[i];

    nick_mine_run(d_table, qb, base, pfx, prefix_len, sfx, suffix_len,
                  result_address, result_nonce, found);
}

/* Exposes the compiled run length so the host can keep its nonce stepping in
 * sync with NICK_ITERS (see gpu_miner_cuda.go). */
extern "C" int nick_cuda_iters() { return NICK_ITERS; }
