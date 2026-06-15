/*
 * CUDA host library for the Nick's-method vanity miner.
 * Provides the C-callable functions used by the Go cgo bindings.
 *
 * Compile with: nvcc -c -o nick_cuda.o cuda_launcher.cu -arch=sm_80
 *               ar rcs libnick_cuda.a nick_cuda.o
 */

#include <cuda_runtime.h>
#include <stdlib.h>
#include <string.h>

#include "nick.cu"

/* comb table size: 8 windows * 256 entries * 64 bytes */
#define NICK_TABLE_BYTES 131072

/* threads per block (tunable: build with -DNICK_BLOCK=128 etc.) */
#ifndef NICK_BLOCK
#define NICK_BLOCK 256
#endif

/* Device info structure (must match the Go side) */
typedef struct {
    int index;
    char name[256];
    int compute_units;
    int max_threads_per_block;
    unsigned long long total_memory;
} CUDADeviceInfo;

/* Per-device miner context */
typedef struct {
    int device_index;
    unsigned char *d_table;
    unsigned char *d_qbase;
    unsigned char *d_prefix;
    unsigned char *d_suffix;
    unsigned char *d_result_addr;
    unsigned long long *d_result_nonce;
    int *d_found;
    int batch_size;
    char device_name[256];
    int table_loaded;
    int qbase_loaded;
    unsigned char cached_qbase[64];
} NickCUDAContext;

extern "C" {

int get_cuda_device_count() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        return -((int)err);
    }
    return count;
}

int get_cuda_device_info(int index, CUDADeviceInfo *info) {
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, index);
    if (err != cudaSuccess) {
        return -1;
    }
    info->index = index;
    strncpy(info->name, prop.name, 255);
    info->name[255] = '\0';
    info->compute_units = prop.multiProcessorCount;
    info->max_threads_per_block = prop.maxThreadsPerBlock;
    info->total_memory = (unsigned long long)prop.totalGlobalMem;
    return 0;
}

NickCUDAContext *nick_cuda_init(int device_index, int batch_size) {
    if (cudaSetDevice(device_index) != cudaSuccess) return NULL;

    NickCUDAContext *ctx = (NickCUDAContext *)malloc(sizeof(NickCUDAContext));
    if (!ctx) return NULL;
    memset(ctx, 0, sizeof(NickCUDAContext));
    ctx->device_index = device_index;
    ctx->batch_size = batch_size;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_index);
    strncpy(ctx->device_name, prop.name, 255);
    ctx->device_name[255] = '\0';

    int ok = 1;
    ok &= (cudaMalloc(&ctx->d_table, NICK_TABLE_BYTES) == cudaSuccess);
    ok &= (cudaMalloc(&ctx->d_qbase, 64) == cudaSuccess);
    ok &= (cudaMalloc(&ctx->d_prefix, 20) == cudaSuccess);
    ok &= (cudaMalloc(&ctx->d_suffix, 20) == cudaSuccess);
    ok &= (cudaMalloc(&ctx->d_result_addr, 20) == cudaSuccess);
    ok &= (cudaMalloc(&ctx->d_result_nonce, sizeof(unsigned long long)) == cudaSuccess);
    ok &= (cudaMalloc(&ctx->d_found, sizeof(int)) == cudaSuccess);
    if (!ok) {
        if (ctx->d_table) cudaFree(ctx->d_table);
        if (ctx->d_qbase) cudaFree(ctx->d_qbase);
        if (ctx->d_prefix) cudaFree(ctx->d_prefix);
        if (ctx->d_suffix) cudaFree(ctx->d_suffix);
        if (ctx->d_result_addr) cudaFree(ctx->d_result_addr);
        if (ctx->d_result_nonce) cudaFree(ctx->d_result_nonce);
        if (ctx->d_found) cudaFree(ctx->d_found);
        free(ctx);
        return NULL;
    }
    return ctx;
}

void nick_cuda_close(NickCUDAContext *ctx) {
    if (!ctx) return;
    cudaSetDevice(ctx->device_index);
    cudaFree(ctx->d_table);
    cudaFree(ctx->d_qbase);
    cudaFree(ctx->d_prefix);
    cudaFree(ctx->d_suffix);
    cudaFree(ctx->d_result_addr);
    cudaFree(ctx->d_result_nonce);
    cudaFree(ctx->d_found);
    free(ctx);
}

/* Upload the constant comb table once per run. Returns 0 on success. */
int nick_cuda_set_table(NickCUDAContext *ctx, const unsigned char *table, int table_len) {
    if (table_len != NICK_TABLE_BYTES) return -1;
    if (cudaSetDevice(ctx->device_index) != cudaSuccess) return -1;
    if (cudaMemcpy(ctx->d_table, table, NICK_TABLE_BYTES, cudaMemcpyHostToDevice) != cudaSuccess)
        return -1;
    ctx->table_loaded = 1;
    return 0;
}

/* Run one batch. Returns 1 if found, 0 if not, -1 on error. */
int nick_cuda_mine(
    NickCUDAContext *ctx,
    const unsigned char *qbase,
    const unsigned char *prefix, int prefix_len,
    const unsigned char *suffix, int suffix_len,
    unsigned long long start_nonce,
    unsigned char *result_addr,
    unsigned long long *result_nonce) {
    if (!ctx->table_loaded) return -1;
    if (cudaSetDevice(ctx->device_index) != cudaSuccess) return -1;

    if (!ctx->qbase_loaded || memcmp(ctx->cached_qbase, qbase, 64) != 0) {
        if (cudaMemcpy(ctx->d_qbase, qbase, 64, cudaMemcpyHostToDevice) != cudaSuccess) return -1;
        memcpy(ctx->cached_qbase, qbase, 64);
        ctx->qbase_loaded = 1;
    }
    if (prefix_len > 0 &&
        cudaMemcpy(ctx->d_prefix, prefix, prefix_len, cudaMemcpyHostToDevice) != cudaSuccess)
        return -1;
    if (suffix_len > 0 &&
        cudaMemcpy(ctx->d_suffix, suffix, suffix_len, cudaMemcpyHostToDevice) != cudaSuccess)
        return -1;

    if (cudaMemset(ctx->d_found, 0, sizeof(int)) != cudaSuccess) return -1;

    int block_size = NICK_BLOCK;
    int num_blocks = (ctx->batch_size + block_size - 1) / block_size;

    mine_nick<<<num_blocks, block_size>>>(
        ctx->d_table, ctx->d_qbase,
        ctx->d_prefix, prefix_len,
        ctx->d_suffix, suffix_len,
        start_nonce,
        ctx->d_result_addr, ctx->d_result_nonce, ctx->d_found);

    if (cudaDeviceSynchronize() != cudaSuccess) return -1;

    int found = 0;
    if (cudaMemcpy(&found, ctx->d_found, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess)
        return -1;
    if (found) {
        cudaMemcpy(result_addr, ctx->d_result_addr, 20, cudaMemcpyDeviceToHost);
        cudaMemcpy(result_nonce, ctx->d_result_nonce, sizeof(unsigned long long),
                   cudaMemcpyDeviceToHost);
        return 1;
    }
    return 0;
}

} /* extern "C" */
