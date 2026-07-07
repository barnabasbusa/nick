/*
 * metal_launcher.h - C ABI over the Metal compute backend for the Nick's-method
 * vanity miner. Consumed by miner/gpu_miner_metal.go via cgo.
 */
#ifndef NICK_METAL_LAUNCHER_H
#define NICK_METAL_LAUNCHER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MetalMiner MetalMiner;

/* List Metal devices: writes up to `cap` names (each truncated to 127 bytes
 * into names[i]) and returns the total device count. names may be NULL to just
 * count. */
int metal_list_devices(char names[][128], int cap);

/* Create a miner on device `device_index`, compiling `msl_src` at runtime with
 * the given per-thread run length and threadgroup size. Returns NULL on failure
 * and writes a message into err (up to errlen bytes). */
MetalMiner *metal_new(int device_index, const char *msl_src,
                      int kernel_iters, int threadgroup_size,
                      char *err, int errlen);

/* Copy the device name into out (NUL-terminated, up to outlen). */
void metal_device_name(MetalMiner *m, char *out, int outlen);

/* Upload the run constants (comb table for D + base pubkey). Called once before
 * the first mine; d_table is d_table_len bytes, q_base is 64 bytes. */
void metal_upload(MetalMiner *m, const uint8_t *d_table, int d_table_len,
                  const uint8_t *q_base);

/* Run one batch of `threads` work items (each processes kernel_iters
 * candidates) over [start_nonce, ...). On a match returns 1 and fills
 * out_addr[20] and *out_nonce (the winning k); returns 0 for no match, -1 on
 * error. */
int metal_mine(MetalMiner *m, int threads,
               const uint8_t *prefix, int prefix_len,
               const uint8_t *suffix, int suffix_len,
               uint64_t start_nonce,
               uint8_t *out_addr, uint64_t *out_nonce);

void metal_free(MetalMiner *m);

#ifdef __cplusplus
}
#endif

#endif /* NICK_METAL_LAUNCHER_H */
