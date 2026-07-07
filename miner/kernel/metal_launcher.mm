/*
 * metal_launcher.mm - Objective-C++ implementation of the Metal compute backend.
 *
 * The Metal shader (nick.metal) is compiled at runtime via newLibraryWithSource:
 * so the binary needs no precompiled .metallib and builds with only the Command
 * Line Tools (no full Xcode). The 128 KB comb table and 64-byte base pubkey are
 * uploaded once into shared (unified-memory) buffers; each batch only rebinds
 * the start nonce and target pattern.
 */
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <string.h>
#include "metal_launcher.h"

struct MetalMiner {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> pso;

    id<MTLBuffer> dTable;   // 128 KB, uploaded once
    id<MTLBuffer> qBase;    // 64 bytes, uploaded once
    id<MTLBuffer> resAddr;  // 20 bytes
    id<MTLBuffer> resNonce; // 8 bytes
    id<MTLBuffer> found;    // 4 bytes (atomic_int)

    int kernelIters;
    int tgSize;
};

static NSArray<id<MTLDevice>> *allDevices() {
#if TARGET_OS_OSX
    NSArray<id<MTLDevice>> *devs = MTLCopyAllDevices();
    if (devs.count > 0) return devs;
#endif
    id<MTLDevice> d = MTLCreateSystemDefaultDevice();
    return d ? @[d] : @[];
}

int metal_list_devices(char names[][128], int cap) {
    @autoreleasepool {
        NSArray<id<MTLDevice>> *devs = allDevices();
        int n = (int)devs.count;
        if (names) {
            for (int i = 0; i < n && i < cap; i++) {
                const char *nm = [[devs[i] name] UTF8String];
                strncpy(names[i], nm, 127);
                names[i][127] = '\0';
            }
        }
        return n;
    }
}

MetalMiner *metal_new(int device_index, const char *msl_src,
                      int kernel_iters, int threadgroup_size,
                      char *err, int errlen) {
    @autoreleasepool {
        NSArray<id<MTLDevice>> *devs = allDevices();
        if (device_index < 0 || device_index >= (int)devs.count) {
            snprintf(err, errlen, "metal device index %d out of range (%d devices)",
                     device_index, (int)devs.count);
            return nullptr;
        }
        id<MTLDevice> dev = devs[device_index];

        NSError *nserr = nil;
        NSString *src = [NSString stringWithUTF8String:msl_src];
        // The kernel is pure integer code, so float fast-math settings are
        // irrelevant; default options keep the build warning-free across SDKs.
        MTLCompileOptions *opt = [MTLCompileOptions new];
        id<MTLLibrary> lib = [dev newLibraryWithSource:src options:opt error:&nserr];
        if (!lib) {
            snprintf(err, errlen, "shader compile failed: %s",
                     [[nserr localizedDescription] UTF8String]);
            return nullptr;
        }
        id<MTLFunction> fn = [lib newFunctionWithName:@"mine_nick"];
        if (!fn) {
            snprintf(err, errlen, "kernel function mine_nick not found");
            return nullptr;
        }
        id<MTLComputePipelineState> pso =
            [dev newComputePipelineStateWithFunction:fn error:&nserr];
        if (!pso) {
            snprintf(err, errlen, "pipeline creation failed: %s",
                     [[nserr localizedDescription] UTF8String]);
            return nullptr;
        }

        MetalMiner *m = new MetalMiner();
        m->device = dev;
        m->queue = [dev newCommandQueue];
        m->pso = pso;
        m->kernelIters = kernel_iters;

        int maxTg = (int)pso.maxTotalThreadsPerThreadgroup;
        int tg = threadgroup_size > 0 ? threadgroup_size : 32;
        if (tg > maxTg) tg = maxTg;
        int width = (int)pso.threadExecutionWidth;
        if (width > 0 && tg % width != 0) tg = (tg / width) * width;
        if (tg < width) tg = width;
        m->tgSize = tg;

        m->dTable   = [dev newBufferWithLength:8 * 256 * 64 options:MTLResourceStorageModeShared];
        m->qBase    = [dev newBufferWithLength:64 options:MTLResourceStorageModeShared];
        m->resAddr  = [dev newBufferWithLength:20 options:MTLResourceStorageModeShared];
        m->resNonce = [dev newBufferWithLength:8  options:MTLResourceStorageModeShared];
        m->found    = [dev newBufferWithLength:4  options:MTLResourceStorageModeShared];
        return m;
    }
}

void metal_device_name(MetalMiner *m, char *out, int outlen) {
    @autoreleasepool {
        const char *nm = [[m->device name] UTF8String];
        strncpy(out, nm, outlen - 1);
        out[outlen - 1] = '\0';
    }
}

void metal_upload(MetalMiner *m, const uint8_t *d_table, int d_table_len,
                  const uint8_t *q_base) {
    memcpy([m->dTable contents], d_table, d_table_len);
    memcpy([m->qBase contents], q_base, 64);
}

int metal_mine(MetalMiner *m, int threads,
               const uint8_t *prefix, int prefix_len,
               const uint8_t *suffix, int suffix_len,
               uint64_t start_nonce,
               uint8_t *out_addr, uint64_t *out_nonce) {
    @autoreleasepool {
        // Reset the result/found slots.
        *(int *)[m->found contents] = 0;
        memset([m->resAddr contents], 0, 20);
        *(uint64_t *)[m->resNonce contents] = 0;

        uint8_t pfx[20] = {0}, sfx[20] = {0};
        if (prefix_len > 20) prefix_len = 20;
        if (suffix_len > 20) suffix_len = 20;
        memcpy(pfx, prefix, prefix_len);
        memcpy(sfx, suffix, suffix_len);

        id<MTLCommandBuffer> cb = [m->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:m->pso];
        [enc setBuffer:m->dTable offset:0 atIndex:0];
        [enc setBuffer:m->qBase  offset:0 atIndex:1];
        [enc setBytes:pfx length:20 atIndex:2];
        [enc setBytes:&prefix_len length:sizeof(int) atIndex:3];
        [enc setBytes:sfx length:20 atIndex:4];
        [enc setBytes:&suffix_len length:sizeof(int) atIndex:5];
        [enc setBytes:&start_nonce length:sizeof(uint64_t) atIndex:6];
        [enc setBuffer:m->resAddr  offset:0 atIndex:7];
        [enc setBuffer:m->resNonce offset:0 atIndex:8];
        [enc setBuffer:m->found    offset:0 atIndex:9];

        MTLSize grid = MTLSizeMake((NSUInteger)threads, 1, 1);
        MTLSize tg = MTLSizeMake((NSUInteger)m->tgSize, 1, 1);
        [enc dispatchThreads:grid threadsPerThreadgroup:tg];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        if (cb.status == MTLCommandBufferStatusError) {
            return -1;
        }
        if (*(int *)[m->found contents] != 0) {
            memcpy(out_addr, [m->resAddr contents], 20);
            *out_nonce = *(uint64_t *)[m->resNonce contents];
            return 1;
        }
        return 0;
    }
}

void metal_free(MetalMiner *m) {
    if (!m) return;
    @autoreleasepool {
        m->device = nil; m->queue = nil; m->pso = nil;
        m->dTable = nil; m->qBase = nil;
        m->resAddr = nil; m->resNonce = nil; m->found = nil;
    }
    delete m;
}
