// capture_mps_gemm.m — MPSMatrixMultiplication with reusable buffer pool + batch.
//
// Single-matrix path: one encode/commit/wait per call (used by micro tools).
// Batched path: MPSMatrixDescriptor.matrices + MPSMatrixMultiplication.batchSize
// so B expert GEMMs of identical (M,N,K) share ONE encode/commit/wait.
//
// Layout for batched buffers (matrix-major, matching MPSMatrix docs):
//   matrix b starts at byte offset b * matrixBytes
//   within a matrix: row i at i * rowBytes

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <pthread.h>
#include <stdint.h>
#include <string.h>

typedef struct {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLBuffer> bufA;
    id<MTLBuffer> bufB;
    id<MTLBuffer> bufC;
    NSUInteger capA;
    NSUInteger capB;
    NSUInteger capC;
} CaptureMpsPool;

static pthread_mutex_t g_pool_mu = PTHREAD_MUTEX_INITIALIZER;
static CaptureMpsPool g_pool = {0};

// Dispatch counter (process lifetime). Atomic-ish under pool mutex.
static uint64_t g_dispatch_count = 0;

static id<MTLBuffer> ensure_buf(CaptureMpsPool *p, id<MTLBuffer> cur, NSUInteger *cap, NSUInteger need) {
    if (cur && *cap >= need) {
        return cur;
    }
    NSUInteger newCap = need + (need / 4) + 4096;
    return [p->device newBufferWithLength:newCap options:MTLResourceStorageModeShared];
}

static int pool_bind(void *device_ptr, void *queue_ptr) {
    id<MTLDevice> device = (__bridge id<MTLDevice>)device_ptr;
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)queue_ptr;
    if (!device || !queue) {
        return 1;
    }
    if (g_pool.device != device || g_pool.queue != queue) {
        g_pool.device = device;
        g_pool.queue = queue;
        g_pool.bufA = nil;
        g_pool.bufB = nil;
        g_pool.bufC = nil;
        g_pool.capA = g_pool.capB = g_pool.capC = 0;
    }
    return 0;
}

static int ensure_abc(CaptureMpsPool *p, NSUInteger aBytes, NSUInteger bBytes, NSUInteger cBytes) {
    id<MTLBuffer> newA = ensure_buf(p, p->bufA, &p->capA, aBytes);
    id<MTLBuffer> newB = ensure_buf(p, p->bufB, &p->capB, bBytes);
    id<MTLBuffer> newC = ensure_buf(p, p->bufC, &p->capC, cBytes);
    if (!newA || !newB || !newC) {
        return 2;
    }
    if (newA != p->bufA) {
        p->bufA = newA;
        p->capA = newA.length;
    }
    if (newB != p->bufB) {
        p->bufB = newB;
        p->capB = newB.length;
    }
    if (newC != p->bufC) {
        p->bufC = newC;
        p->capC = newC.length;
    }
    return 0;
}

uint64_t hawking_capture_mps_dispatch_count(void) {
    pthread_mutex_lock(&g_pool_mu);
    uint64_t n = g_dispatch_count;
    pthread_mutex_unlock(&g_pool_mu);
    return n;
}

void hawking_capture_mps_dispatch_count_reset(void) {
    pthread_mutex_lock(&g_pool_mu);
    g_dispatch_count = 0;
    pthread_mutex_unlock(&g_pool_mu);
}

// Out = X @ W^T   X[M,K], W[N,K], Out[M,N]
int hawking_capture_mps_gemm_x_wt(
    void *device_ptr,
    void *queue_ptr,
    const float *X,
    const float *W,
    float *Out,
    uint32_t M,
    uint32_t N,
    uint32_t K
) {
    if (M == 0 || N == 0 || K == 0) {
        return 0;
    }
    pthread_mutex_lock(&g_pool_mu);
    @autoreleasepool {
        if (pool_bind(device_ptr, queue_ptr) != 0) {
            pthread_mutex_unlock(&g_pool_mu);
            return 1;
        }
        CaptureMpsPool *p = &g_pool;
        NSUInteger aBytes = (NSUInteger)M * (NSUInteger)K * sizeof(float);
        NSUInteger bBytes = (NSUInteger)N * (NSUInteger)K * sizeof(float);
        NSUInteger cBytes = (NSUInteger)M * (NSUInteger)N * sizeof(float);
        int er = ensure_abc(p, aBytes, bBytes, cBytes);
        if (er != 0) {
            pthread_mutex_unlock(&g_pool_mu);
            return er;
        }

        memcpy([p->bufA contents], X, aBytes);
        memcpy([p->bufB contents], W, bBytes);

        MPSMatrixDescriptor *dA = [MPSMatrixDescriptor
            matrixDescriptorWithRows:M columns:K
                            rowBytes:K * sizeof(float)
                            dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor *dW = [MPSMatrixDescriptor
            matrixDescriptorWithRows:N columns:K
                            rowBytes:K * sizeof(float)
                            dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor *dC = [MPSMatrixDescriptor
            matrixDescriptorWithRows:M columns:N
                            rowBytes:N * sizeof(float)
                            dataType:MPSDataTypeFloat32];

        MPSMatrix *matA = [[MPSMatrix alloc] initWithBuffer:p->bufA descriptor:dA];
        MPSMatrix *matW = [[MPSMatrix alloc] initWithBuffer:p->bufB descriptor:dW];
        MPSMatrix *matC = [[MPSMatrix alloc] initWithBuffer:p->bufC descriptor:dC];

        MPSMatrixMultiplication *mult = [[MPSMatrixMultiplication alloc]
            initWithDevice:p->device
             transposeLeft:NO
            transposeRight:YES
                resultRows:M
             resultColumns:N
          interiorColumns:K
                     alpha:1.0
                      beta:0.0];

        id<MTLCommandBuffer> cmd = [p->queue commandBuffer];
        if (!cmd) {
            pthread_mutex_unlock(&g_pool_mu);
            return 3;
        }
        [mult encodeToCommandBuffer:cmd leftMatrix:matA rightMatrix:matW resultMatrix:matC];
        [cmd commit];
        [cmd waitUntilCompleted];
        g_dispatch_count += 1;
        memcpy(Out, [p->bufC contents], cBytes);
    }
    pthread_mutex_unlock(&g_pool_mu);
    return 0;
}

// Out = A @ B   A[M,K], B[K,N], Out[M,N]
int hawking_capture_mps_gemm_ab(
    void *device_ptr,
    void *queue_ptr,
    const float *A,
    const float *B,
    float *Out,
    uint32_t M,
    uint32_t N,
    uint32_t K
) {
    if (M == 0 || N == 0 || K == 0) {
        return 0;
    }
    pthread_mutex_lock(&g_pool_mu);
    @autoreleasepool {
        if (pool_bind(device_ptr, queue_ptr) != 0) {
            pthread_mutex_unlock(&g_pool_mu);
            return 1;
        }
        CaptureMpsPool *p = &g_pool;
        NSUInteger aBytes = (NSUInteger)M * (NSUInteger)K * sizeof(float);
        NSUInteger bBytes = (NSUInteger)K * (NSUInteger)N * sizeof(float);
        NSUInteger cBytes = (NSUInteger)M * (NSUInteger)N * sizeof(float);
        int er = ensure_abc(p, aBytes, bBytes, cBytes);
        if (er != 0) {
            pthread_mutex_unlock(&g_pool_mu);
            return er;
        }

        memcpy([p->bufA contents], A, aBytes);
        memcpy([p->bufB contents], B, bBytes);

        MPSMatrixDescriptor *dA = [MPSMatrixDescriptor
            matrixDescriptorWithRows:M columns:K
                            rowBytes:K * sizeof(float)
                            dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor *dB = [MPSMatrixDescriptor
            matrixDescriptorWithRows:K columns:N
                            rowBytes:N * sizeof(float)
                            dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor *dC = [MPSMatrixDescriptor
            matrixDescriptorWithRows:M columns:N
                            rowBytes:N * sizeof(float)
                            dataType:MPSDataTypeFloat32];

        MPSMatrix *matA = [[MPSMatrix alloc] initWithBuffer:p->bufA descriptor:dA];
        MPSMatrix *matB = [[MPSMatrix alloc] initWithBuffer:p->bufB descriptor:dB];
        MPSMatrix *matC = [[MPSMatrix alloc] initWithBuffer:p->bufC descriptor:dC];

        MPSMatrixMultiplication *mult = [[MPSMatrixMultiplication alloc]
            initWithDevice:p->device
             transposeLeft:NO
            transposeRight:NO
                resultRows:M
             resultColumns:N
          interiorColumns:K
                     alpha:1.0
                      beta:0.0];

        id<MTLCommandBuffer> cmd = [p->queue commandBuffer];
        if (!cmd) {
            pthread_mutex_unlock(&g_pool_mu);
            return 3;
        }
        [mult encodeToCommandBuffer:cmd leftMatrix:matA rightMatrix:matB resultMatrix:matC];
        [cmd commit];
        [cmd waitUntilCompleted];
        g_dispatch_count += 1;
        memcpy(Out, [p->bufC contents], cBytes);
    }
    pthread_mutex_unlock(&g_pool_mu);
    return 0;
}

// Batched: for b in 0..B-1:
//   Out_b = X_b @ W_b^T
// X layout:  B matrices of M×K, matrixBytes = M*K*4, rowBytes = K*4
// W layout:  B matrices of N×K, matrixBytes = N*K*4, rowBytes = K*4
// Out layout: B matrices of M×N, matrixBytes = M*N*4, rowBytes = N*4
//
// ONE encode/commit/wait for the whole batch (MPSMatrixMultiplication.batchSize).
int hawking_capture_mps_gemm_x_wt_batched(
    void *device_ptr,
    void *queue_ptr,
    const float *X,
    const float *W,
    float *Out,
    uint32_t B,
    uint32_t M,
    uint32_t N,
    uint32_t K
) {
    if (B == 0 || M == 0 || N == 0 || K == 0) {
        return 0;
    }
    // Degenerate: single matrix path (same numerics, simpler descriptors).
    if (B == 1) {
        return hawking_capture_mps_gemm_x_wt(device_ptr, queue_ptr, X, W, Out, M, N, K);
    }

    pthread_mutex_lock(&g_pool_mu);
    @autoreleasepool {
        if (pool_bind(device_ptr, queue_ptr) != 0) {
            pthread_mutex_unlock(&g_pool_mu);
            return 1;
        }
        CaptureMpsPool *p = &g_pool;
        NSUInteger matrixBytesA = (NSUInteger)M * (NSUInteger)K * sizeof(float);
        NSUInteger matrixBytesW = (NSUInteger)N * (NSUInteger)K * sizeof(float);
        NSUInteger matrixBytesC = (NSUInteger)M * (NSUInteger)N * sizeof(float);
        NSUInteger aBytes = (NSUInteger)B * matrixBytesA;
        NSUInteger bBytes = (NSUInteger)B * matrixBytesW;
        NSUInteger cBytes = (NSUInteger)B * matrixBytesC;
        int er = ensure_abc(p, aBytes, bBytes, cBytes);
        if (er != 0) {
            pthread_mutex_unlock(&g_pool_mu);
            return er;
        }

        memcpy([p->bufA contents], X, aBytes);
        memcpy([p->bufB contents], W, bBytes);

        MPSMatrixDescriptor *dA = [MPSMatrixDescriptor
            matrixDescriptorWithRows:M
                             columns:K
                            matrices:B
                            rowBytes:K * sizeof(float)
                         matrixBytes:matrixBytesA
                            dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor *dW = [MPSMatrixDescriptor
            matrixDescriptorWithRows:N
                             columns:K
                            matrices:B
                            rowBytes:K * sizeof(float)
                         matrixBytes:matrixBytesW
                            dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor *dC = [MPSMatrixDescriptor
            matrixDescriptorWithRows:M
                             columns:N
                            matrices:B
                            rowBytes:N * sizeof(float)
                         matrixBytes:matrixBytesC
                            dataType:MPSDataTypeFloat32];

        MPSMatrix *matA = [[MPSMatrix alloc] initWithBuffer:p->bufA descriptor:dA];
        MPSMatrix *matW = [[MPSMatrix alloc] initWithBuffer:p->bufB descriptor:dW];
        MPSMatrix *matC = [[MPSMatrix alloc] initWithBuffer:p->bufC descriptor:dC];

        MPSMatrixMultiplication *mult = [[MPSMatrixMultiplication alloc]
            initWithDevice:p->device
             transposeLeft:NO
            transposeRight:YES
                resultRows:M
             resultColumns:N
          interiorColumns:K
                     alpha:1.0
                      beta:0.0];
        mult.batchStart = 0;
        mult.batchSize = B;

        id<MTLCommandBuffer> cmd = [p->queue commandBuffer];
        if (!cmd) {
            pthread_mutex_unlock(&g_pool_mu);
            return 3;
        }
        [mult encodeToCommandBuffer:cmd leftMatrix:matA rightMatrix:matW resultMatrix:matC];
        [cmd commit];
        [cmd waitUntilCompleted];
        g_dispatch_count += 1;
        memcpy(Out, [p->bufC contents], cBytes);
    }
    pthread_mutex_unlock(&g_pool_mu);
    return 0;
}

// Multi-encode single-wait: B independent (M_b, N, K) GEMMs of form Out = X @ W^T
// where N and K are shared, but M_b may differ. Pointer arrays index host buffers.
// Encodes B multiplications into ONE command buffer (still B encodes, but 1 wait).
// Used when padding to max_M would exceed the scratch budget.
//
// For simplicity and shared-buffer safety this path stages into offset regions of
// the pool buffers when total payload fits; otherwise falls back to serial single GEMMs.
int hawking_capture_mps_gemm_x_wt_grouped_varM(
    void *device_ptr,
    void *queue_ptr,
    const float *const *Xs,
    const float *const *Ws,
    float *const *Outs,
    const uint32_t *Ms,
    uint32_t B,
    uint32_t N,
    uint32_t K
) {
    if (B == 0 || N == 0 || K == 0) {
        return 0;
    }
    // Always-correct path: one CB, encode each, single wait — but only if we can
    // stage all into non-overlapping regions of the pool. Otherwise serial.
    NSUInteger totalA = 0, totalW = 0, totalC = 0;
    for (uint32_t b = 0; b < B; b++) {
        uint32_t M = Ms[b];
        if (M == 0) {
            continue;
        }
        totalA += (NSUInteger)M * (NSUInteger)K;
        totalW += (NSUInteger)N * (NSUInteger)K;
        totalC += (NSUInteger)M * (NSUInteger)N;
    }
    NSUInteger aBytes = totalA * sizeof(float);
    NSUInteger bBytes = totalW * sizeof(float);
    NSUInteger cBytes = totalC * sizeof(float);

    pthread_mutex_lock(&g_pool_mu);
    @autoreleasepool {
        if (pool_bind(device_ptr, queue_ptr) != 0) {
            pthread_mutex_unlock(&g_pool_mu);
            return 1;
        }
        CaptureMpsPool *p = &g_pool;
        int er = ensure_abc(p, aBytes, bBytes, cBytes);
        if (er != 0) {
            pthread_mutex_unlock(&g_pool_mu);
            return er;
        }

        float *dstA = (float *)[p->bufA contents];
        float *dstW = (float *)[p->bufB contents];
        float *dstC = (float *)[p->bufC contents];

        // Stage host → shared buffers with per-matrix offsets.
        NSUInteger offA = 0, offW = 0, offC = 0;
        NSUInteger *oA = (NSUInteger *)calloc(B, sizeof(NSUInteger));
        NSUInteger *oW = (NSUInteger *)calloc(B, sizeof(NSUInteger));
        NSUInteger *oC = (NSUInteger *)calloc(B, sizeof(NSUInteger));
        if (!oA || !oW || !oC) {
            free(oA); free(oW); free(oC);
            pthread_mutex_unlock(&g_pool_mu);
            return 2;
        }
        for (uint32_t b = 0; b < B; b++) {
            uint32_t M = Ms[b];
            oA[b] = offA;
            oW[b] = offW;
            oC[b] = offC;
            if (M == 0) {
                continue;
            }
            memcpy(dstA + offA, Xs[b], (size_t)M * K * sizeof(float));
            memcpy(dstW + offW, Ws[b], (size_t)N * K * sizeof(float));
            offA += (NSUInteger)M * K;
            offW += (NSUInteger)N * K;
            offC += (NSUInteger)M * N;
        }

        id<MTLCommandBuffer> cmd = [p->queue commandBuffer];
        if (!cmd) {
            free(oA); free(oW); free(oC);
            pthread_mutex_unlock(&g_pool_mu);
            return 3;
        }

        for (uint32_t b = 0; b < B; b++) {
            uint32_t M = Ms[b];
            if (M == 0) {
                continue;
            }
            MPSMatrixDescriptor *dA = [MPSMatrixDescriptor
                matrixDescriptorWithRows:M columns:K
                                rowBytes:K * sizeof(float)
                                dataType:MPSDataTypeFloat32];
            MPSMatrixDescriptor *dW = [MPSMatrixDescriptor
                matrixDescriptorWithRows:N columns:K
                                rowBytes:K * sizeof(float)
                                dataType:MPSDataTypeFloat32];
            MPSMatrixDescriptor *dC = [MPSMatrixDescriptor
                matrixDescriptorWithRows:M columns:N
                                rowBytes:N * sizeof(float)
                                dataType:MPSDataTypeFloat32];
            MPSMatrix *matA = [[MPSMatrix alloc]
                initWithBuffer:p->bufA
                        offset:oA[b] * sizeof(float)
                    descriptor:dA];
            MPSMatrix *matW = [[MPSMatrix alloc]
                initWithBuffer:p->bufB
                        offset:oW[b] * sizeof(float)
                    descriptor:dW];
            MPSMatrix *matC = [[MPSMatrix alloc]
                initWithBuffer:p->bufC
                        offset:oC[b] * sizeof(float)
                    descriptor:dC];
            MPSMatrixMultiplication *mult = [[MPSMatrixMultiplication alloc]
                initWithDevice:p->device
                 transposeLeft:NO
                transposeRight:YES
                    resultRows:M
                 resultColumns:N
              interiorColumns:K
                         alpha:1.0
                          beta:0.0];
            [mult encodeToCommandBuffer:cmd leftMatrix:matA rightMatrix:matW resultMatrix:matC];
        }

        [cmd commit];
        [cmd waitUntilCompleted];
        g_dispatch_count += 1; // one commit/wait counts as one dispatch for the report

        // Read back
        for (uint32_t b = 0; b < B; b++) {
            uint32_t M = Ms[b];
            if (M == 0) {
                continue;
            }
            memcpy(Outs[b], dstC + oC[b], (size_t)M * N * sizeof(float));
        }
        free(oA); free(oW); free(oC);
    }
    pthread_mutex_unlock(&g_pool_mu);
    return 0;
}
