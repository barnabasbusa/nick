/*
 * CUDA entry kernel for Nick's-method vanity mining.
 * Includes the shared device library (same math as the OpenCL build).
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

    u64 k = start_nonce + (u64)(blockIdx.x * blockDim.x + threadIdx.x);

    u8 qb[64];
    for (int i = 0; i < 64; i++) qb[i] = q_base[i];
    u8 pfx[20], sfx[20];
    for (int i = 0; i < prefix_len; i++) pfx[i] = prefix[i];
    for (int i = 0; i < suffix_len; i++) sfx[i] = suffix[i];

    u8 addr[20];
    nick_address_for_k(d_table, qb, k, addr);

    if (nick_match(addr, pfx, prefix_len, sfx, suffix_len)) {
        if (atomicCAS(found, 0, 1) == 0) {
            for (int i = 0; i < 20; i++) result_address[i] = addr[i];
            *result_nonce = k;
        }
    }
}
