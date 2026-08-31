// Metal / IOKit / GPURawCounter enumerator for tools/future/memory_traffic_probe.py.
// Capability query plus a tiny compute copy. Does not count DRAM bytes; it
// reports which surfaces exist and what they returned on this device.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <IOKit/IOKitLib.h>
#import <objc/runtime.h>
#import <dlfcn.h>
#import <string.h>
#import <stdio.h>

static id JSONNull(void) { return [NSNull null]; }

static id describeError(NSError *error) {
    if (!error) return JSONNull();
    return @{
        @"domain": error.domain ?: @"",
        @"code": @(error.code),
        @"description": error.localizedDescription ?: @"",
    };
}

static NSArray<NSString *> *methodNames(Class cls) {
    unsigned int count = 0;
    Method *list = class_copyMethodList(cls, &count);
    NSMutableArray *names = [NSMutableArray arrayWithCapacity:count];
    for (unsigned int i = 0; i < count; i++) {
        [names addObject:NSStringFromSelector(method_getName(list[i]))];
    }
    if (list) free(list);
    return [names sortedArrayUsingSelector:@selector(compare:)];
}

static NSString *jsonString(id obj) {
    NSError *err = nil;
    NSData *data = [NSJSONSerialization dataWithJSONObject:obj
                                                   options:NSJSONWritingPrettyPrinted | NSJSONWritingSortedKeys
                                                     error:&err];
    if (!data) return [NSString stringWithFormat:@"{\"json_error\":\"%@\"}", err.localizedDescription];
    return [[NSString alloc] initWithData:data encoding:NSUTF8StringEncoding];
}

static id jsonSafe(id v) {
    if (!v || v == [NSNull null]) return JSONNull();
    if ([v isKindOfClass:[NSNumber class]] || [v isKindOfClass:[NSString class]] ||
        [v isKindOfClass:[NSArray class]] || [v isKindOfClass:[NSDictionary class]]) {
        return v;
    }
    return [v description];
}

int main(void) {
    @autoreleasepool {
        NSMutableDictionary *out = [NSMutableDictionary dictionary];
        out[@"probe"] = @"mtl_traffic_probe";
        out[@"uname"] = NSProcessInfo.processInfo.operatingSystemVersionString;

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            out[@"error"] = @"MTLCreateSystemDefaultDevice returned nil";
            printf("%s\n", jsonString(out).UTF8String);
            return 1;
        }

        NSMutableDictionary *deviceInfo = [@{
            @"name": device.name ?: @"",
            @"registryID": @(device.registryID),
            @"hasUnifiedMemory": @(device.hasUnifiedMemory),
            @"recommendedMaxWorkingSetSize": @(device.recommendedMaxWorkingSetSize),
            @"maxBufferLength": @(device.maxBufferLength),
            @"currentAllocatedSize": @(device.currentAllocatedSize),
            @"maxThreadgroupMemoryLength": @(device.maxThreadgroupMemoryLength),
        } mutableCopy];
        out[@"device"] = deviceInfo;

        NSMutableDictionary *sampling = [NSMutableDictionary dictionary];
        sampling[@"AtStageBoundary"] = @([device supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary]);
        sampling[@"AtDrawBoundary"] = @([device supportsCounterSampling:MTLCounterSamplingPointAtDrawBoundary]);
        sampling[@"AtDispatchBoundary"] = @([device supportsCounterSampling:MTLCounterSamplingPointAtDispatchBoundary]);
        sampling[@"AtTileDispatchBoundary"] = @([device supportsCounterSampling:MTLCounterSamplingPointAtTileDispatchBoundary]);
        sampling[@"AtBlitBoundary"] = @([device supportsCounterSampling:MTLCounterSamplingPointAtBlitBoundary]);
        out[@"supports_counter_sampling"] = sampling;

        out[@"mtl_common_counter_set_constants"] = @{
            @"MTLCommonCounterSetTimestamp": (NSString *)MTLCommonCounterSetTimestamp,
            @"MTLCommonCounterSetStageUtilization": (NSString *)MTLCommonCounterSetStageUtilization,
            @"MTLCommonCounterSetStatistic": (NSString *)MTLCommonCounterSetStatistic,
        };
        out[@"mtl_common_counter_constants"] = @{
            @"MTLCommonCounterTimestamp": (NSString *)MTLCommonCounterTimestamp,
            @"MTLCommonCounterTessellationInputPatches": (NSString *)MTLCommonCounterTessellationInputPatches,
            @"MTLCommonCounterVertexInvocations": (NSString *)MTLCommonCounterVertexInvocations,
            @"MTLCommonCounterPostTessellationVertexInvocations": (NSString *)MTLCommonCounterPostTessellationVertexInvocations,
            @"MTLCommonCounterClipperInvocations": (NSString *)MTLCommonCounterClipperInvocations,
            @"MTLCommonCounterClipperPrimitivesOut": (NSString *)MTLCommonCounterClipperPrimitivesOut,
            @"MTLCommonCounterFragmentInvocations": (NSString *)MTLCommonCounterFragmentInvocations,
            @"MTLCommonCounterFragmentsPassed": (NSString *)MTLCommonCounterFragmentsPassed,
            @"MTLCommonCounterComputeKernelInvocations": (NSString *)MTLCommonCounterComputeKernelInvocations,
            @"MTLCommonCounterTotalCycles": (NSString *)MTLCommonCounterTotalCycles,
            @"MTLCommonCounterVertexCycles": (NSString *)MTLCommonCounterVertexCycles,
            @"MTLCommonCounterTessellationCycles": (NSString *)MTLCommonCounterTessellationCycles,
            @"MTLCommonCounterPostTessellationVertexCycles": (NSString *)MTLCommonCounterPostTessellationVertexCycles,
            @"MTLCommonCounterFragmentCycles": (NSString *)MTLCommonCounterFragmentCycles,
            @"MTLCommonCounterRenderTargetWriteCycles": (NSString *)MTLCommonCounterRenderTargetWriteCycles,
        };

        NSArray<id<MTLCounterSet>> *sets = device.counterSets ?: @[];
        NSMutableArray *setsOut = [NSMutableArray array];
        NSMutableArray *setNames = [NSMutableArray array];
        for (id<MTLCounterSet> cset in sets) {
            [setNames addObject:cset.name ?: @""];
            NSMutableArray *counters = [NSMutableArray array];
            NSMutableArray *counterNames = [NSMutableArray array];
            for (id<MTLCounter> c in cset.counters) {
                [counters addObject:@{@"name": c.name ?: @""}];
                [counterNames addObject:c.name ?: @""];
            }
            MTLCounterSampleBufferDescriptor *desc = [MTLCounterSampleBufferDescriptor new];
            desc.counterSet = cset;
            desc.sampleCount = 4;
            desc.storageMode = MTLStorageModeShared;
            desc.label = [NSString stringWithFormat:@"probe-%@", cset.name];
            NSError *err = nil;
            id<MTLCounterSampleBuffer> buf = [device newCounterSampleBufferWithDescriptor:desc error:&err];
            [setsOut addObject:@{
                @"name": cset.name ?: @"",
                @"counters": counters,
                @"counter_names": counterNames,
                @"sample_buffer_created": @(buf != nil),
                @"sample_buffer_error": describeError(err),
            }];
        }
        out[@"counter_sets"] = setsOut;
        out[@"counter_set_names"] = setNames;
        out[@"n_counter_sets"] = @(sets.count);

        NSMutableDictionary *mtl4 = [@{
            @"header_types": @[@"MTL4CounterHeapTypeInvalid", @"MTL4CounterHeapTypeTimestamp"],
            @"memory_traffic_heap_type_in_public_api": @NO,
        } mutableCopy];
        if (@available(macOS 26.0, *)) {
            mtl4[@"macos26_available"] = @YES;
            mtl4[@"MTL4CounterHeapTypeTimestamp_raw"] = @(MTL4CounterHeapTypeTimestamp);
            mtl4[@"MTL4CounterHeapTypeInvalid_raw"] = @(MTL4CounterHeapTypeInvalid);
        } else {
            mtl4[@"macos26_available"] = @NO;
        }
        out[@"mtl4_counter_heap"] = mtl4;

        NSUInteger beforeAlloc = device.currentAllocatedSize;
        id<MTLBuffer> scratch = [device newBufferWithLength:16 * 1024 * 1024 options:MTLResourceStorageModeShared];
        NSUInteger afterAlloc = device.currentAllocatedSize;
        out[@"current_allocated_size"] = @{
            @"before_16mib_buffer": @(beforeAlloc),
            @"after_16mib_buffer": @(afterAlloc),
            @"delta": @((long long)afterAlloc - (long long)beforeAlloc),
            @"meaning": @"bytes currently allocated by this process on the device, not bytes transferred",
            @"is_memory_traffic": @NO,
        };

        NSMutableDictionary *residency = [@{@"api_available": @NO} mutableCopy];
        if (@available(macOS 15.0, *)) {
            residency[@"api_available"] = @YES;
            if (scratch) {
                MTLResidencySetDescriptor *rdesc = [MTLResidencySetDescriptor new];
                rdesc.label = @"probe-residency";
                rdesc.initialCapacity = 1;
                NSError *rerr = nil;
                id<MTLResidencySet> rset = [device newResidencySetWithDescriptor:rdesc error:&rerr];
                if (rset) {
                    [rset addAllocation:scratch];
                    [rset commit];
                    residency[@"allocatedSize_after_commit"] = @(rset.allocatedSize);
                    residency[@"is_memory_traffic"] = @NO;
                    residency[@"meaning"] = @"residency set footprint at last commit, not bytes transferred";
                } else {
                    residency[@"error"] = describeError(rerr);
                }
            }
        }
        out[@"residency_set"] = residency;

        NSString *src = @"#include <metal_stdlib>\n"
                         "using namespace metal;\n"
                         "kernel void copy_u32(device const uint* in [[buffer(0)]],\n"
                         "                     device uint* out [[buffer(1)]],\n"
                         "                     uint i [[thread_position_in_grid]]) {\n"
                         "    out[i] = in[i];\n"
                         "}\n";
        NSError *libErr = nil;
        id<MTLLibrary> lib = [device newLibraryWithSource:src options:nil error:&libErr];
        id<MTLFunction> fn = [lib newFunctionWithName:@"copy_u32"];
        NSError *pipeErr = nil;
        id<MTLComputePipelineState> pipeline = fn ? [device newComputePipelineStateWithFunction:fn error:&pipeErr] : nil;
        id<MTLCommandQueue> queue = [device newCommandQueue];

        NSMutableArray *samples = [NSMutableArray array];
        if (pipeline && queue) {
            NSArray<NSNumber *> *payloads = @[ @(1 << 20), @(4 << 20) ];
            BOOL dispatchSample = [device supportsCounterSampling:MTLCounterSamplingPointAtDispatchBoundary];
            for (id<MTLCounterSet> cset in sets) {
                for (NSNumber *payload in payloads) {
                    NSUInteger bytes = payload.unsignedIntegerValue;
                    NSMutableDictionary *result = [@{
                        @"set": cset.name ?: @"",
                        @"payload_bytes": @(bytes),
                    } mutableCopy];
                    MTLCounterSampleBufferDescriptor *desc = [MTLCounterSampleBufferDescriptor new];
                    desc.counterSet = cset;
                    desc.sampleCount = 4;
                    desc.storageMode = MTLStorageModeShared;
                    NSError *err = nil;
                    id<MTLCounterSampleBuffer> sbuf = [device newCounterSampleBufferWithDescriptor:desc error:&err];
                    if (!sbuf) {
                        result[@"error"] = describeError(err);
                        [samples addObject:result];
                        continue;
                    }
                    id<MTLBuffer> inBuf = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
                    id<MTLBuffer> outBuf = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
                    if (!inBuf || !outBuf) {
                        result[@"error"] = @"buffer alloc failed";
                        [samples addObject:result];
                        continue;
                    }
                    memset(inBuf.contents, 1, bytes);
                    id<MTLCommandBuffer> cmd = [queue commandBuffer];
                    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
                    [enc setComputePipelineState:pipeline];
                    [enc setBuffer:inBuf offset:0 atIndex:0];
                    [enc setBuffer:outBuf offset:0 atIndex:1];
                    result[@"sampled_via"] = @"encoder_sampleCountersInBuffer_dispatch_boundary";
                    if (dispatchSample) {
                        [enc sampleCountersInBuffer:sbuf atSampleIndex:0 withBarrier:YES];
                    }
                    NSUInteger n = bytes / 4;
                    [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
                    if (dispatchSample) {
                        [enc sampleCountersInBuffer:sbuf atSampleIndex:1 withBarrier:YES];
                    }
                    [enc endEncoding];
                    [cmd commit];
                    [cmd waitUntilCompleted];
                    result[@"gpu_start_time"] = @(cmd.GPUStartTime);
                    result[@"gpu_end_time"] = @(cmd.GPUEndTime);
                    result[@"gpu_duration_s"] = @(cmd.GPUEndTime - cmd.GPUStartTime);
                    NSData *data = [sbuf resolveCounterRange:NSMakeRange(0, 2)];
                    if (data) {
                        const uint64_t *vals = (const uint64_t *)data.bytes;
                        NSUInteger nU64 = data.length / 8;
                        NSMutableArray *nums = [NSMutableArray array];
                        NSMutableArray *hex = [NSMutableArray array];
                        BOOL sawErr = NO;
                        for (NSUInteger i = 0; i < nU64; i++) {
                            if (vals[i] == UINT64_MAX) {
                                sawErr = YES;
                                [nums addObject:@(-1)];
                            } else {
                                [nums addObject:@(vals[i])];
                            }
                            [hex addObject:[NSString stringWithFormat:@"0x%llx", (unsigned long long)vals[i]]];
                        }
                        result[@"resolved_u64"] = nums;
                        result[@"resolved_u64_hex"] = hex;
                        result[@"resolved_byte_count"] = @(data.length);
                        result[@"contains_mtl_counter_error_value"] = @(sawErr);
                    } else {
                        result[@"resolved_u64"] = JSONNull();
                        result[@"resolve_returned_nil"] = @YES;
                    }
                    [samples addObject:result];
                }
            }
            // Stage-boundary sampling is the only supported point on this device.
            // Use MTLComputePassDescriptor sampleBufferAttachments, not
            // encoder sampleCountersInBuffer (that is dispatch-boundary).
            BOOL stageSample = [device supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary];
            NSMutableArray *stageSamples = [NSMutableArray array];
            if (stageSample) {
                for (id<MTLCounterSet> cset in sets) {
                    for (NSNumber *payload in payloads) {
                        NSUInteger bytes = payload.unsignedIntegerValue;
                        NSMutableDictionary *result = [@{
                            @"set": cset.name ?: @"",
                            @"payload_bytes": @(bytes),
                            @"sampled_via": @"compute_pass_sampleBufferAttachments_stage_boundary",
                        } mutableCopy];
                        MTLCounterSampleBufferDescriptor *desc = [MTLCounterSampleBufferDescriptor new];
                        desc.counterSet = cset;
                        desc.sampleCount = 4;
                        desc.storageMode = MTLStorageModeShared;
                        NSError *err = nil;
                        id<MTLCounterSampleBuffer> sbuf = [device newCounterSampleBufferWithDescriptor:desc error:&err];
                        if (!sbuf) {
                            result[@"error"] = describeError(err);
                            [stageSamples addObject:result];
                            continue;
                        }
                        id<MTLBuffer> inBuf = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
                        id<MTLBuffer> outBuf = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
                        if (!inBuf || !outBuf) {
                            result[@"error"] = @"buffer alloc failed";
                            [stageSamples addObject:result];
                            continue;
                        }
                        memset(inBuf.contents, 1, bytes);
                        MTLComputePassDescriptor *pass = [MTLComputePassDescriptor computePassDescriptor];
                        pass.sampleBufferAttachments[0].sampleBuffer = sbuf;
                        pass.sampleBufferAttachments[0].startOfEncoderSampleIndex = 0;
                        pass.sampleBufferAttachments[0].endOfEncoderSampleIndex = 1;
                        id<MTLCommandBuffer> cmd = [queue commandBuffer];
                        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoderWithDescriptor:pass];
                        if (!enc) {
                            result[@"error"] = @"computeCommandEncoderWithDescriptor returned nil";
                            [stageSamples addObject:result];
                            continue;
                        }
                        [enc setComputePipelineState:pipeline];
                        [enc setBuffer:inBuf offset:0 atIndex:0];
                        [enc setBuffer:outBuf offset:0 atIndex:1];
                        NSUInteger n = bytes / 4;
                        [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
                        [enc endEncoding];
                        [cmd commit];
                        [cmd waitUntilCompleted];
                        result[@"gpu_start_time"] = @(cmd.GPUStartTime);
                        result[@"gpu_end_time"] = @(cmd.GPUEndTime);
                        result[@"gpu_duration_s"] = @(cmd.GPUEndTime - cmd.GPUStartTime);
                        NSData *data = [sbuf resolveCounterRange:NSMakeRange(0, 2)];
                        if (data) {
                            const uint64_t *vals = (const uint64_t *)data.bytes;
                            NSUInteger nU64 = data.length / 8;
                            NSMutableArray *nums = [NSMutableArray array];
                            NSMutableArray *hex = [NSMutableArray array];
                            BOOL sawErr = NO;
                            uint64_t t0 = 0, t1 = 0;
                            for (NSUInteger i = 0; i < nU64; i++) {
                                if (vals[i] == UINT64_MAX) {
                                    sawErr = YES;
                                    [nums addObject:@(-1)];
                                } else {
                                    [nums addObject:@(vals[i])];
                                    if (i == 0) t0 = vals[i];
                                    if (i == 1) t1 = vals[i];
                                }
                                [hex addObject:[NSString stringWithFormat:@"0x%llx", (unsigned long long)vals[i]]];
                            }
                            result[@"resolved_u64"] = nums;
                            result[@"resolved_u64_hex"] = hex;
                            result[@"resolved_byte_count"] = @(data.length);
                            result[@"contains_mtl_counter_error_value"] = @(sawErr);
                            if (!sawErr && nU64 >= 2 && t1 >= t0) {
                                result[@"timestamp_delta_ns"] = @(t1 - t0);
                            }
                        } else {
                            result[@"resolved_u64"] = JSONNull();
                            result[@"resolve_returned_nil"] = @YES;
                        }
                        [stageSamples addObject:result];
                    }
                }
            }
            out[@"compute_copy_counter_samples"] = samples;
            out[@"stage_boundary_counter_samples"] = stageSamples;
        } else {
            out[@"compute_copy_counter_samples"] = JSONNull();
            out[@"compute_copy_error"] = @{
                @"library": describeError(libErr),
                @"pipeline": describeError(pipeErr),
            };
        }

        MTLTimestamp cpuTs = 0, gpuTs = 0;
        [device sampleTimestamps:&cpuTs gpuTimestamp:&gpuTs];
        out[@"sample_timestamps"] = @{
            @"cpu": @(cpuTs),
            @"gpu": @(gpuTs),
            @"is_memory_traffic": @NO,
            @"meaning": @"clock samples, not bytes transferred",
        };

        // GPURawCounter
        {
            NSMutableDictionary *info = [NSMutableDictionary dictionary];
            const char *path = "/System/Library/PrivateFrameworks/GPURawCounter.framework/GPURawCounter";
            void *handle = dlopen(path, RTLD_NOW);
            if (!handle) {
                info[@"dlopen"] = @"failed";
                const char *e = dlerror();
                info[@"dlerror"] = e ? @(e) : @"";
            } else {
                info[@"dlopen"] = @"ok";
                typedef NSArray *(*CopyErrFn)(NSError **);
                typedef NSArray *(*CopyFn)(void);
                CopyErrFn copyErr = (CopyErrFn)dlsym(handle, "GRCCopyAllCounterSourceGroupWithError");
                CopyFn copyPlain = (CopyFn)dlsym(handle, "GRCCopyAllCounterSourceGroup");
                NSError *cerr = nil;
                NSArray *groups = nil;
                if (copyErr) {
                    groups = copyErr(&cerr);
                    if (cerr) info[@"copy_error"] = describeError(cerr);
                } else if (copyPlain) {
                    groups = copyPlain();
                } else {
                    info[@"symbol"] = @"GRCCopyAllCounterSourceGroup* not found";
                }
                if (groups) {
                    info[@"n_groups"] = @(groups.count);
                    NSMutableArray *groupRows = [NSMutableArray array];
                    for (id group in groups) {
                        NSMutableDictionary *grow = [NSMutableDictionary dictionary];
                        grow[@"class"] = NSStringFromClass([group class]);
                        grow[@"instance_methods"] = methodNames([group class]);
                        if ([group respondsToSelector:@selector(name)]) {
                            grow[@"name"] = jsonSafe([group valueForKey:@"name"]);
                        }
                        if ([group respondsToSelector:NSSelectorFromString(@"features")]) {
                            grow[@"features"] = jsonSafe([group valueForKey:@"features"]);
                        }
                        NSMutableArray *sourceRows = [NSMutableArray array];
                        if ([group respondsToSelector:NSSelectorFromString(@"sourceList")]) {
                            NSArray *sources = [group valueForKey:@"sourceList"];
                            for (id srcObj in sources) {
                                NSMutableDictionary *srow = [NSMutableDictionary dictionary];
                                srow[@"class"] = NSStringFromClass([srcObj class]);
                                srow[@"instance_methods"] = methodNames([srcObj class]);
                                if ([srcObj respondsToSelector:@selector(name)]) {
                                    srow[@"name"] = jsonSafe([srcObj valueForKey:@"name"]);
                                }
                                if ([srcObj respondsToSelector:NSSelectorFromString(@"availableCounters")]) {
                                    NSArray *counters = [srcObj valueForKey:@"availableCounters"];
                                    NSMutableArray *crow = [NSMutableArray array];
                                    NSMutableArray *cnames = [NSMutableArray array];
                                    for (id c in counters) {
                                        NSMutableDictionary *rec = [NSMutableDictionary dictionary];
                                        rec[@"class"] = NSStringFromClass([c class]);
                                        if ([c respondsToSelector:@selector(name)]) {
                                            id n = [c valueForKey:@"name"];
                                            rec[@"name"] = jsonSafe(n);
                                            if ([n isKindOfClass:[NSString class]]) [cnames addObject:n];
                                        }
                                        if ([c respondsToSelector:NSSelectorFromString(@"counterValueType")]) {
                                            rec[@"counterValueType"] = jsonSafe([c valueForKey:@"counterValueType"]);
                                        }
                                        rec[@"description"] = jsonSafe([c description]);
                                        [crow addObject:rec];
                                    }
                                    srow[@"available_counters"] = crow;
                                    srow[@"available_counter_names"] = cnames;
                                } else {
                                    srow[@"available_counters"] = @"selector_absent";
                                }
                                [sourceRows addObject:srow];
                            }
                        }
                        grow[@"sources"] = sourceRows;
                        [groupRows addObject:grow];
                    }
                    info[@"groups"] = groupRows;
                } else {
                    info[@"groups"] = JSONNull();
                    if (!info[@"copy_error"] && !info[@"symbol"]) {
                        info[@"note"] = @"copy returned nil without NSError";
                    }
                }
            }
            out[@"gpu_raw_counter"] = info;
        }

        // IOAccelMemoryInfo
        {
            NSMutableDictionary *info = [NSMutableDictionary dictionary];
            const char *path = "/System/Library/PrivateFrameworks/IOAccelMemoryInfo.framework/IOAccelMemoryInfo";
            void *handle = dlopen(path, RTLD_NOW);
            if (!handle) {
                info[@"dlopen"] = @"failed";
                const char *e = dlerror();
                info[@"dlerror"] = e ? @(e) : @"";
            } else {
                info[@"dlopen"] = @"ok";
                Class cls = NSClassFromString(@"IOAccelMemoryInfo");
                if (!cls) {
                    info[@"class"] = JSONNull();
                } else {
                    info[@"class"] = @"IOAccelMemoryInfo";
                    info[@"class_methods"] = methodNames(object_getClass(cls));
                    info[@"instance_methods"] = methodNames(cls);
                    NSArray *candidates = @[@"memoryInfos", @"collectMemoryInfos", @"newMemoryInfos", @"allMemoryInfos"];
                    NSMutableDictionary *calls = [NSMutableDictionary dictionary];
                    for (NSString *name in candidates) {
                        SEL sel = NSSelectorFromString(name);
                        if ([cls respondsToSelector:sel]) {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Warc-performSelector-leaks"
                            id result = [cls performSelector:sel];
#pragma clang diagnostic pop
                            if ([result isKindOfClass:[NSArray class]]) {
                                NSArray *arr = (NSArray *)result;
                                calls[name] = @{
                                    @"count": @(arr.count),
                                    @"first_class": arr.count ? NSStringFromClass([arr[0] class]) : JSONNull(),
                                };
                            } else {
                                calls[name] = jsonSafe(result);
                            }
                        } else {
                            calls[name] = @"selector_absent";
                        }
                    }
                    info[@"tried_class_methods"] = calls;
                }
            }
            info[@"is_memory_traffic"] = @NO;
            info[@"meaning"] = @"per-allocation GPU memory census (footprint), not bytes transferred";
            out[@"ioaccel_memory_info"] = info;
        }

        // IORegistry
        {
            NSMutableDictionary *info = [NSMutableDictionary dictionary];
            io_iterator_t iterator = 0;
            kern_return_t kr = IOServiceGetMatchingServices(kIOMainPortDefault, IOServiceMatching("IOAccelerator"), &iterator);
            info[@"IOServiceGetMatchingServices_IOAccelerator"] = @(kr);
            NSMutableArray *accelEntries = [NSMutableArray array];
            if (kr == KERN_SUCCESS) {
                io_service_t svc;
                while ((svc = IOIteratorNext(iterator))) {
                    CFMutableDictionaryRef props = NULL;
                    kern_return_t pr = IORegistryEntryCreateCFProperties(svc, &props, kCFAllocatorDefault, 0);
                    NSMutableDictionary *row = [NSMutableDictionary dictionary];
                    row[@"create_properties"] = @(pr);
                    if (props) {
                        NSDictionary *dict = (__bridge_transfer NSDictionary *)props;
                        row[@"IOClass"] = jsonSafe(dict[@"IOClass"]);
                        row[@"CFBundleIdentifier"] = jsonSafe(dict[@"CFBundleIdentifier"]);
                        row[@"gpu-core-count"] = jsonSafe(dict[@"gpu-core-count"]);
                        row[@"GPURawCounterBundleName"] = jsonSafe(dict[@"GPURawCounterBundleName"]);
                        row[@"GPURawCounterPluginClassName"] = jsonSafe(dict[@"GPURawCounterPluginClassName"]);
                        id perf = dict[@"PerformanceStatistics"];
                        if ([perf isKindOfClass:[NSDictionary class]]) {
                            NSDictionary *pd = (NSDictionary *)perf;
                            NSMutableDictionary *perfOut = [NSMutableDictionary dictionary];
                            for (NSString *k in pd) perfOut[k] = jsonSafe(pd[k]);
                            row[@"PerformanceStatistics"] = perfOut;
                            row[@"PerformanceStatistics_keys"] = [[pd allKeys] sortedArrayUsingSelector:@selector(compare:)];
                        } else {
                            row[@"PerformanceStatistics"] = JSONNull();
                        }
                        id legend = dict[@"IOReportLegend"];
                        if ([legend isKindOfClass:[NSArray class]]) {
                            NSMutableArray *names = [NSMutableArray array];
                            for (id item in (NSArray *)legend) {
                                if (![item isKindOfClass:[NSDictionary class]]) continue;
                                NSArray *channels = item[@"IOReportChannels"];
                                if (![channels isKindOfClass:[NSArray class]]) continue;
                                for (id ch in channels) {
                                    if ([ch isKindOfClass:[NSArray class]] && [(NSArray *)ch count] >= 3) {
                                        id n = ch[2];
                                        if ([n isKindOfClass:[NSString class]]) [names addObject:n];
                                    }
                                }
                            }
                            row[@"IOReport_channel_names"] = names;
                        }
                    }
                    IOObjectRelease(svc);
                    [accelEntries addObject:row];
                }
                IOObjectRelease(iterator);
            }
            info[@"ioaccelerator_entries"] = accelEntries;

            io_iterator_t agxIt = 0;
            kern_return_t agxKr = IOServiceGetMatchingServices(kIOMainPortDefault, IOServiceMatching("AGXAcceleratorG15X"), &agxIt);
            info[@"IOServiceGetMatchingServices_AGXAcceleratorG15X"] = @(agxKr);
            NSMutableArray *agxEntries = [NSMutableArray array];
            if (agxKr == KERN_SUCCESS) {
                io_service_t svc;
                while ((svc = IOIteratorNext(agxIt))) {
                    CFMutableDictionaryRef props = NULL;
                    IORegistryEntryCreateCFProperties(svc, &props, kCFAllocatorDefault, 0);
                    NSMutableDictionary *row = [NSMutableDictionary dictionary];
                    if (props) {
                        NSDictionary *dict = (__bridge_transfer NSDictionary *)props;
                        row[@"IOClass"] = jsonSafe(dict[@"IOClass"]);
                        row[@"CFBundleIdentifier"] = jsonSafe(dict[@"CFBundleIdentifier"]);
                        row[@"gpu-core-count"] = jsonSafe(dict[@"gpu-core-count"]);
                        row[@"GPURawCounterBundleName"] = jsonSafe(dict[@"GPURawCounterBundleName"]);
                        id perf = dict[@"PerformanceStatistics"];
                        if ([perf isKindOfClass:[NSDictionary class]]) {
                            NSDictionary *pd = (NSDictionary *)perf;
                            NSMutableDictionary *perfOut = [NSMutableDictionary dictionary];
                            for (NSString *k in pd) perfOut[k] = jsonSafe(pd[k]);
                            row[@"PerformanceStatistics"] = perfOut;
                            row[@"PerformanceStatistics_keys"] = [[pd allKeys] sortedArrayUsingSelector:@selector(compare:)];
                        }
                        id legend = dict[@"IOReportLegend"];
                        if ([legend isKindOfClass:[NSArray class]]) {
                            NSMutableArray *names = [NSMutableArray array];
                            for (id item in (NSArray *)legend) {
                                if (![item isKindOfClass:[NSDictionary class]]) continue;
                                NSArray *channels = item[@"IOReportChannels"];
                                if (![channels isKindOfClass:[NSArray class]]) continue;
                                for (id ch in channels) {
                                    if ([ch isKindOfClass:[NSArray class]] && [(NSArray *)ch count] >= 3) {
                                        id n = ch[2];
                                        if ([n isKindOfClass:[NSString class]]) [names addObject:n];
                                    }
                                }
                            }
                            row[@"IOReport_channel_names"] = names;
                        }
                    }
                    IOObjectRelease(svc);
                    [agxEntries addObject:row];
                }
                IOObjectRelease(agxIt);
            }
            info[@"agxaccelerator_g15x_entries"] = agxEntries;
            out[@"iokit"] = info;
        }

        {
            NSMutableDictionary *info = [NSMutableDictionary dictionary];
            const char *path = "/System/Library/PrivateFrameworks/MetalMetrics.framework/MetalMetrics";
            void *h = dlopen(path, RTLD_NOW);
            if (!h) {
                info[@"dlopen"] = @"failed";
                const char *e = dlerror();
                info[@"dlerror"] = e ? @(e) : @"";
            } else {
                info[@"dlopen"] = @"ok";
                info[@"note"] = @"dyld export table was empty; no public counter API discovered";
            }
            out[@"metal_metrics"] = info;
        }

        printf("%s\n", jsonString(out).UTF8String);
    }
    return 0;
}
