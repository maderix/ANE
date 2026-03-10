// ane_runtime.h — Reusable ANE in-memory compile/load/eval wrapper
// Uses _ANEInMemoryModel via private AppleNeuralEngine.framework
#pragma once
#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <IOSurface/IOSurface.h>
#import <sys/mman.h>
#import <sys/stat.h>
#import <fcntl.h>
#import <sys/sysctl.h>

// Chip Detection and MIL Version Selection

typedef NS_ENUM(NSInteger, ANEChipType) {
    ANE_CHIP_UNKNOWN = 0,
    ANE_CHIP_M1, ANE_CHIP_M1_PRO, ANE_CHIP_M1_MAX, ANE_CHIP_M1_ULTRA,
    ANE_CHIP_M2, ANE_CHIP_M2_PRO, ANE_CHIP_M2_MAX, ANE_CHIP_M2_ULTRA,
    ANE_CHIP_M3, ANE_CHIP_M3_PRO, ANE_CHIP_M3_MAX, ANE_CHIP_M3_ULTRA,
    ANE_CHIP_M4, ANE_CHIP_M4_PRO, ANE_CHIP_M4_MAX,
    ANE_CHIP_M5
};

static const size_t SYSCTL_BUFFER_SIZE = 256;
static const int ASCII_DIGIT_OFFSET = '0';
static const int BASE_CHIP_GENERATION_MULTIPLIER = 10;
static const int PRO_VARIANT_OFFSET = 1;
static const int MAX_VARIANT_OFFSET = 2;
static const int ULTRA_VARIANT_OFFSET = 3;

static const char* SYSCTL_BRAND_STRING_KEY = "machdep.cpu.brand_string";
static const char* APPLE_M_PREFIX = "Apple M";
static const size_t APPLE_M_PREFIX_LENGTH = 7;

static const char* VARIANT_PRO = "Pro";
static const char* VARIANT_MAX = "Max";
static const char* VARIANT_ULTRA = "Ultra";
static const size_t VARIANT_PRO_MAX_LENGTH = 3;
static const size_t VARIANT_ULTRA_LENGTH = 5;

static ANEChipType parse_base_chip_generation(const char *generation_string) {
    int generation = 0;
    if (generation_string[0] >= '1' && generation_string[0] <= '9') {
        generation = generation_string[0] - ASCII_DIGIT_OFFSET;
        if (generation_string[1] >= '0' && generation_string[1] <= '9') {
            generation = generation * BASE_CHIP_GENERATION_MULTIPLIER + (generation_string[1] - ASCII_DIGIT_OFFSET);
        }
    }
    
    switch (generation) {
        case 1: return ANE_CHIP_M1;
        case 2: return ANE_CHIP_M2;
        case 3: return ANE_CHIP_M3;
        case 4: return ANE_CHIP_M4;
        case 5: return ANE_CHIP_M5;
        default: return ANE_CHIP_UNKNOWN;
    }
}

static ANEChipType parse_chip_variant(ANEChipType base_chip, const char *variant_string) {
    if (strncmp(variant_string, VARIANT_PRO, VARIANT_PRO_MAX_LENGTH) == 0) {
        return (ANEChipType)(base_chip + PRO_VARIANT_OFFSET);
    }
    if (strncmp(variant_string, VARIANT_MAX, VARIANT_PRO_MAX_LENGTH) == 0) {
        return (ANEChipType)(base_chip + MAX_VARIANT_OFFSET);
    }
    if (strncmp(variant_string, VARIANT_ULTRA, VARIANT_ULTRA_LENGTH) == 0) {
        return (ANEChipType)(base_chip + ULTRA_VARIANT_OFFSET);
    }
    return base_chip;
}

static ANEChipType ane_get_chip_type(void) {
    static ANEChipType cached_chip = ANE_CHIP_UNKNOWN;
    static bool initialized = false;
    
    if (initialized) return cached_chip;
    initialized = true;
    
    char brand[SYSCTL_BUFFER_SIZE] = {0};
    size_t brand_size = sizeof(brand);
    
    if (sysctlbyname(SYSCTL_BRAND_STRING_KEY, brand, &brand_size, NULL, 0) == 0) {
        if (strncmp(brand, APPLE_M_PREFIX, APPLE_M_PREFIX_LENGTH) == 0) {
            const char *generation_pointer = brand + APPLE_M_PREFIX_LENGTH;
            ANEChipType base_chip = parse_base_chip_generation(generation_pointer);
            
            if (base_chip != ANE_CHIP_UNKNOWN) {
                const char *variant_pointer = generation_pointer + 1;
                if (generation_pointer[1] >= '0' && generation_pointer[1] <= '9') {
                    variant_pointer++;
                }
                while (*variant_pointer == ' ') {
                    variant_pointer++;
                }
                cached_chip = parse_chip_variant(base_chip, variant_pointer);
            }
        }
    }
    
    return cached_chip;
}

static bool ane_supports_mil_1_5(void) {
    return (ane_get_chip_type() >= ANE_CHIP_M5);
}

static const char *ane_get_mil_version(void) {
    return ane_supports_mil_1_5() ? "1.5" : "1.3";
}

static const char *ane_get_mil_ios_target(void) {
    return ane_supports_mil_1_5() ? "ios18" : "ios17";
}

static const char *ane_get_chip_name(void) {
    switch (ane_get_chip_type()) {
        case ANE_CHIP_M1: return "M1";
        case ANE_CHIP_M1_PRO: return "M1 Pro";
        case ANE_CHIP_M1_MAX: return "M1 Max";
        case ANE_CHIP_M1_ULTRA: return "M1 Ultra";
        case ANE_CHIP_M2: return "M2";
        case ANE_CHIP_M2_PRO: return "M2 Pro";
        case ANE_CHIP_M2_MAX: return "M2 Max";
        case ANE_CHIP_M2_ULTRA: return "M2 Ultra";
        case ANE_CHIP_M3: return "M3";
        case ANE_CHIP_M3_PRO: return "M3 Pro";
        case ANE_CHIP_M3_MAX: return "M3 Max";
        case ANE_CHIP_M3_ULTRA: return "M3 Ultra";
        case ANE_CHIP_M4: return "M4";
        case ANE_CHIP_M4_PRO: return "M4 Pro";
        case ANE_CHIP_M4_MAX: return "M4 Max";
        case ANE_CHIP_M5: return "M5";
        default: return "Unknown";
    }
}

typedef struct {
    id model;               // _ANEInMemoryModel
    IOSurfaceRef *ioInputs;
    IOSurfaceRef *ioOutputs;
    IOSurfaceRef weightsSurface;  // Optional: dynamic weights IOSurface
    id weightsBuffer;              // Optional: _ANEIOSurfaceObject for weights
    id request;             // _ANERequest
    NSString *tmpDir;
    int nInputs, nOutputs;
    size_t *inputBytes;
    size_t *outputBytes;
    size_t weightsBytes;    // Size of weights surface
} ANEKernel;

static Class g_ANEDesc, g_ANEInMem, g_ANEReq, g_ANEIO;
static bool g_ane_loaded = false;

static void ane_init(void) {
    if (g_ane_loaded) return;
    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
    g_ANEDesc  = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    g_ANEInMem = NSClassFromString(@"_ANEInMemoryModel");
    g_ANEReq   = NSClassFromString(@"_ANERequest");
    g_ANEIO    = NSClassFromString(@"_ANEIOSurfaceObject");
    g_ane_loaded = true;
}

static IOSurfaceRef ane_create_surface(size_t bytes) {
    size_t aligned = ((bytes + 127) / 128) * 128;
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth: @(aligned),
        (id)kIOSurfaceHeight: @1,
        (id)kIOSurfaceBytesPerElement: @1,
        (id)kIOSurfaceBytesPerRow: @(aligned),
        (id)kIOSurfaceAllocSize: @(aligned),
        (id)kIOSurfacePixelFormat: @0
    });
}

// Create an IOSurface specifically for dynamic weights.
// Uses the same 128-byte alignment as regular surfaces.
static IOSurfaceRef ane_create_weights_surface(size_t bytes) {
    size_t aligned = ((bytes + 127) / 128) * 128;
    if (aligned < 128) aligned = 128;
    
    NSMutableDictionary *props = [NSMutableDictionary dictionaryWithObjectsAndKeys:
        @(aligned), (id)kIOSurfaceWidth,
        @1, (id)kIOSurfaceHeight,
        @1, (id)kIOSurfaceBytesPerElement,
        @(aligned), (id)kIOSurfaceBytesPerRow,
        @(aligned), (id)kIOSurfaceAllocSize,
        @0, (id)kIOSurfacePixelFormat,
        nil];
    
    // Enable global access for ANE hardware
    [props setObject:@YES forKey:(id)kIOSurfaceIsGlobal];
    
    return IOSurfaceCreate((__bridge CFDictionaryRef)props);
}

static ANEKernel *ane_compile_with_weights(NSData *milText, NSData *weightData,
                               int nInputs, size_t *inputSizes,
                               int nOutputs, size_t *outputSizes,
                               IOSurfaceRef weightsSurface) {
    ane_init();
    NSError *e = nil;

    NSDictionary *wdict = nil;
    if (weightData) {
        wdict = @{@"@model_path/weights/weight.bin": @{@"offset": @0, @"data": weightData}};
    }
    id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
        g_ANEDesc, @selector(modelWithMILText:weights:optionsPlist:),
        milText, wdict, nil);
    if (!desc) return NULL;

    id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(
        g_ANEInMem, @selector(inMemoryModelWithDescriptor:), desc);

    // Pre-populate temp dir with MIL + weights
    id hx = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
    NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
    NSFileManager *fm = [NSFileManager defaultManager];
    [fm createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"]
        withIntermediateDirectories:YES attributes:nil error:nil];
    [milText writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
    if (weightData)
        [weightData writeToFile:[td stringByAppendingPathComponent:@"weights/weight.bin"] atomically:YES];

    if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
            mdl, @selector(compileWithQoS:options:error:), 21, @{}, &e)) {
        fprintf(stderr, "ANE compile failed: %s\n", [[e description] UTF8String]);
        [fm removeItemAtPath:td error:nil];
        return NULL;
    }
    if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
            mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e)) {
        fprintf(stderr, "ANE load failed: %s\n", [[e description] UTF8String]);
        [fm removeItemAtPath:td error:nil];
        return NULL;
    }

    ANEKernel *k = calloc(1, sizeof(ANEKernel));
    k->model = mdl;
    k->tmpDir = td;
    k->nInputs = nInputs;
    k->nOutputs = nOutputs;
    k->inputBytes = malloc(nInputs * sizeof(size_t));
    k->outputBytes = malloc(nOutputs * sizeof(size_t));
    memcpy(k->inputBytes, inputSizes, nInputs * sizeof(size_t));
    memcpy(k->outputBytes, outputSizes, nOutputs * sizeof(size_t));

    // Create IOSurfaces for inputs/outputs
    k->ioInputs = malloc(nInputs * sizeof(IOSurfaceRef));
    k->ioOutputs = malloc(nOutputs * sizeof(IOSurfaceRef));
    for (int i = 0; i < nInputs; i++)
        k->ioInputs[i] = ane_create_surface(inputSizes[i]);
    for (int i = 0; i < nOutputs; i++)
        k->ioOutputs[i] = ane_create_surface(outputSizes[i]);

    // Handle optional weights surface for dynamic weight injection
    id weightsBufferObj = nil;
    if (weightsSurface) {
        k->weightsSurface = weightsSurface;
        CFRetain(weightsSurface);
        k->weightsBytes = IOSurfaceGetAllocSize(weightsSurface);
        weightsBufferObj = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
            g_ANEIO, @selector(objectWithIOSurface:), weightsSurface);
        k->weightsBuffer = weightsBufferObj;
    }

    // Build request with optional weights buffer
    NSMutableArray *wIns = [NSMutableArray arrayWithCapacity:nInputs];
    NSMutableArray *iIdx = [NSMutableArray arrayWithCapacity:nInputs];
    for (int i = 0; i < nInputs; i++) {
        [wIns addObject:((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
            g_ANEIO, @selector(objectWithIOSurface:), k->ioInputs[i])];
        [iIdx addObject:@(i)];
    }
    NSMutableArray *wOuts = [NSMutableArray arrayWithCapacity:nOutputs];
    NSMutableArray *oIdx = [NSMutableArray arrayWithCapacity:nOutputs];
    for (int i = 0; i < nOutputs; i++) {
        [wOuts addObject:((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
            g_ANEIO, @selector(objectWithIOSurface:), k->ioOutputs[i])];
        [oIdx addObject:@(i)];
    }
    k->request = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(
        g_ANEReq, @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
        wIns, iIdx, wOuts, oIdx, weightsBufferObj, nil, @0);

    return k;
}

static ANEKernel *ane_compile(NSData *milText, NSData *weightData,
                                      int nInputs, size_t *inputSizes,
                                      int nOutputs, size_t *outputSizes) {
    return ane_compile_with_weights(milText, weightData, nInputs, inputSizes, nOutputs, outputSizes, NULL);
}

static int ane_load_weights(ANEKernel *k, const void *data, size_t bytes) {
    if (!k || !k->weightsSurface) {
        fprintf(stderr, "ane_load_weights: kernel has no weights surface\n");
        return -1;
    }
    
    size_t surfaceSize = IOSurfaceGetAllocSize(k->weightsSurface);
    if (bytes > surfaceSize) {
        fprintf(stderr, "ane_load_weights: data size %zu exceeds surface size %zu\n",
                bytes, surfaceSize);
        return -1;
    }
    
    IOSurfaceLock(k->weightsSurface, 0, NULL);
    memcpy(IOSurfaceGetBaseAddress(k->weightsSurface), data, bytes);
    IOSurfaceUnlock(k->weightsSurface, 0, NULL);
    
    return 0;
}

static void *ane_weights_lock(ANEKernel *k) {
    if (!k || !k->weightsSurface) return NULL;
    IOSurfaceLock(k->weightsSurface, 0, NULL);
    return IOSurfaceGetBaseAddress(k->weightsSurface);
}

static void ane_weights_unlock(ANEKernel *k) {
    if (!k || !k->weightsSurface) return;
    IOSurfaceUnlock(k->weightsSurface, 0, NULL);
}

static void ane_write_input(ANEKernel *k, int idx, const void *data, size_t bytes) {
    IOSurfaceLock(k->ioInputs[idx], 0, NULL);
    memcpy(IOSurfaceGetBaseAddress(k->ioInputs[idx]), data, bytes);
    IOSurfaceUnlock(k->ioInputs[idx], 0, NULL);
}

static void ane_read_output(ANEKernel *k, int idx, void *data, size_t bytes) {
    IOSurfaceLock(k->ioOutputs[idx], kIOSurfaceLockReadOnly, NULL);
    memcpy(data, IOSurfaceGetBaseAddress(k->ioOutputs[idx]), bytes);
    IOSurfaceUnlock(k->ioOutputs[idx], kIOSurfaceLockReadOnly, NULL);
}

static bool ane_eval(ANEKernel *k) {
    NSError *e = nil;
    BOOL result = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
        k->model, @selector(evaluateWithQoS:options:request:error:),
        21, @{}, k->request, &e);
    
    if (!result && e) {
        fprintf(stderr, "ANE evaluation failed: %s\n", [[e localizedDescription] UTF8String]);
    }
    
    return result;
}

static void ane_free(ANEKernel *k) {
    if (!k) return;
    NSError *e = nil;
    ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(
        k->model, @selector(unloadWithQoS:error:), 21, &e);
    for (int i = 0; i < k->nInputs; i++) CFRelease(k->ioInputs[i]);
    for (int i = 0; i < k->nOutputs; i++) CFRelease(k->ioOutputs[i]);
    if (k->weightsSurface) CFRelease(k->weightsSurface);
    [[NSFileManager defaultManager] removeItemAtPath:k->tmpDir error:nil];
    free(k->ioInputs); free(k->ioOutputs);
    free(k->inputBytes); free(k->outputBytes);
    free(k);
}
