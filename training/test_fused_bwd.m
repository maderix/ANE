// Test: fused backward dx kernels
// 1. Fused QKV backward: concat(Wq^T@dq, Wk^T@dk, Wv^T@dv) — 3 inputs, 1 output
//    Problem: 3 separate gradient inputs. Can we concat them as input?
//    Input: [1, DIM*3, 1, SEQ] = concat(dq, dk, dv)
//    Use 3 separate convs on slices? MIL has slice_by_size.
// 2. Fused W1b+W3b: input concat(dh1, dh3) [1, HIDDEN*2, 1, SEQ]
//    Two convs on slices, add results → [1, DIM, 1, SEQ]
#import <Foundation/Foundation.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <IOSurface/IOSurface.h>
#include <math.h>

#define DIM 768
#define HIDDEN 2048
#define SEQ 64

static int g_fp16_io = 0;  // M1/M2: cast op unsupported, use fp16 I/O directly

static Class g_D, g_I, g_AR, g_AIO;
static void ane_init(void) {
    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
    g_D  = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    g_I  = NSClassFromString(@"_ANEInMemoryModel");
    g_AR = NSClassFromString(@"_ANERequest");
    g_AIO= NSClassFromString(@"_ANEIOSurfaceObject");
}
static IOSurfaceRef make_surface(size_t bytes) {
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth:@(bytes), (id)kIOSurfaceHeight:@1,
        (id)kIOSurfaceBytesPerElement:@1, (id)kIOSurfaceBytesPerRow:@(bytes),
        (id)kIOSurfaceAllocSize:@(bytes), (id)kIOSurfacePixelFormat:@0});
}
static NSData *build_blob_t(const float *w, int rows, int cols) {
    int wsize = cols * rows * 2, total = 128 + wsize;
    uint8_t *buf = (uint8_t*)calloc(total, 1);
    buf[0]=1; buf[4]=2; buf[64]=0xEF; buf[65]=0xBE; buf[66]=0xAD; buf[67]=0xDE; buf[68]=1;
    *(uint32_t*)(buf+72)=wsize; *(uint32_t*)(buf+80)=128;
    _Float16 *fp16 = (_Float16*)(buf+128);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            fp16[j*rows+i] = (_Float16)w[i*cols+j];
    return [NSData dataWithBytesNoCopy:buf length:total freeWhenDone:YES];
}

int main() {
    @autoreleasepool {
        setbuf(stdout, NULL);
        ane_init();

        srand48(42);
        float *W1 = (float*)malloc(HIDDEN*DIM*sizeof(float));
        float *W3 = (float*)malloc(HIDDEN*DIM*sizeof(float));
        float sc = 1.0f/sqrtf(HIDDEN);
        for (int i = 0; i < HIDDEN*DIM; i++) { W1[i]=sc*(2*drand48()-1); W3[i]=sc*(2*drand48()-1); }

        // Test: fused W1b+W3b backward
        // Input: concat(dh1, dh3) [1, HIDDEN*2, 1, SEQ]
        // Output: W1^T@dh1 + W3^T@dh3 [1, DIM, 1, SEQ]
        // MIL: slice input → 2 convs → add
        printf("=== Fused W1b+W3b backward (slice+conv+add) ===\n");

        retry_compile:;
        NSString *mil;
        if (g_fp16_io) {
            mil = [NSString stringWithFormat:
                @"program(1.0)\n[buildInfo = dict<tensor<string, []>, tensor<string, []>>({{\"coremlc-version\", \"3505.4.1\"}})]\n{\n"
                "    func main<ios16>(tensor<fp16, [1, %d, 1, %d]> x) {\n"
                "        tensor<int32, [4]> b1 = const()[name = tensor<string, []>(\"b1\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
                "        tensor<int32, [4]> s1 = const()[name = tensor<string, []>(\"s1\"), val = tensor<int32, [4]>([1, %d, 1, %d])];\n"
                "        tensor<fp16, [1, %d, 1, %d]> dh1 = slice_by_size(x = x, begin = b1, size = s1)[name = tensor<string, []>(\"sl1\")];\n"
                "        tensor<int32, [4]> b3 = const()[name = tensor<string, []>(\"b3\"), val = tensor<int32, [4]>([0, %d, 0, 0])];\n"
                "        tensor<int32, [4]> s3 = const()[name = tensor<string, []>(\"s3\"), val = tensor<int32, [4]>([1, %d, 1, %d])];\n"
                "        tensor<fp16, [1, %d, 1, %d]> dh3 = slice_by_size(x = x, begin = b3, size = s3)[name = tensor<string, []>(\"sl3\")];\n"
                "        tensor<string, []> pt = const()[name = tensor<string, []>(\"pt\"), val = tensor<string, []>(\"valid\")];\n"
                "        tensor<int32, [2]> st = const()[name = tensor<string, []>(\"st\"), val = tensor<int32, [2]>([1, 1])];\n"
                "        tensor<int32, [4]> pd = const()[name = tensor<string, []>(\"pd\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
                "        tensor<int32, [2]> dl = const()[name = tensor<string, []>(\"dl\"), val = tensor<int32, [2]>([1, 1])];\n"
                "        tensor<int32, []> gr = const()[name = tensor<string, []>(\"gr\"), val = tensor<int32, []>(1)];\n"
                "        tensor<fp16, [%d, %d, 1, 1]> W1t = const()[name = tensor<string, []>(\"W1t\"), "
                "val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = tensor<string, []>(\"@model_path/weights/w1t.bin\"), offset = tensor<uint64, []>(64)))];\n"
                "        tensor<fp16, [%d, %d, 1, 1]> W3t = const()[name = tensor<string, []>(\"W3t\"), "
                "val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = tensor<string, []>(\"@model_path/weights/w3t.bin\"), offset = tensor<uint64, []>(64)))];\n"
                "        tensor<fp16, [1, %d, 1, %d]> dx1 = conv(dilations = dl, groups = gr, pad = pd, "
                "pad_type = pt, strides = st, weight = W1t, x = dh1)[name = tensor<string, []>(\"cv1\")];\n"
                "        tensor<fp16, [1, %d, 1, %d]> dx3 = conv(dilations = dl, groups = gr, pad = pd, "
                "pad_type = pt, strides = st, weight = W3t, x = dh3)[name = tensor<string, []>(\"cv3\")];\n"
                "        tensor<fp16, [1, %d, 1, %d]> y = add(x = dx1, y = dx3)[name = tensor<string, []>(\"ad\")];\n"
                "    } -> (y);\n}\n",
                HIDDEN*2, SEQ,
                HIDDEN, SEQ, HIDDEN, SEQ,
                HIDDEN, HIDDEN, SEQ, HIDDEN, SEQ,
                DIM, HIDDEN, DIM, HIDDEN,
                DIM, HIDDEN, DIM, HIDDEN,
                DIM, SEQ, DIM, SEQ,
                DIM, SEQ];
        } else {
            mil = [NSString stringWithFormat:
                @"program(1.0)\n[buildInfo = dict<tensor<string, []>, tensor<string, []>>({{\"coremlc-version\", \"3505.4.1\"}})]\n{\n"
                "    func main<ios16>(tensor<fp32, [1, %d, 1, %d]> x) {\n"
                "        tensor<string, []> d1 = const()[name = tensor<string, []>(\"d1\"), val = tensor<string, []>(\"fp16\")];\n"
                "        tensor<fp16, [1, %d, 1, %d]> x16 = cast(dtype = d1, x = x)[name = tensor<string, []>(\"cx\")];\n"
                "        tensor<int32, [4]> b1 = const()[name = tensor<string, []>(\"b1\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
                "        tensor<int32, [4]> s1 = const()[name = tensor<string, []>(\"s1\"), val = tensor<int32, [4]>([1, %d, 1, %d])];\n"
                "        tensor<fp16, [1, %d, 1, %d]> dh1 = slice_by_size(x = x16, begin = b1, size = s1)[name = tensor<string, []>(\"sl1\")];\n"
                "        tensor<int32, [4]> b3 = const()[name = tensor<string, []>(\"b3\"), val = tensor<int32, [4]>([0, %d, 0, 0])];\n"
                "        tensor<int32, [4]> s3 = const()[name = tensor<string, []>(\"s3\"), val = tensor<int32, [4]>([1, %d, 1, %d])];\n"
                "        tensor<fp16, [1, %d, 1, %d]> dh3 = slice_by_size(x = x16, begin = b3, size = s3)[name = tensor<string, []>(\"sl3\")];\n"
                "        tensor<string, []> pt = const()[name = tensor<string, []>(\"pt\"), val = tensor<string, []>(\"valid\")];\n"
                "        tensor<int32, [2]> st = const()[name = tensor<string, []>(\"st\"), val = tensor<int32, [2]>([1, 1])];\n"
                "        tensor<int32, [4]> pd = const()[name = tensor<string, []>(\"pd\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
                "        tensor<int32, [2]> dl = const()[name = tensor<string, []>(\"dl\"), val = tensor<int32, [2]>([1, 1])];\n"
                "        tensor<int32, []> gr = const()[name = tensor<string, []>(\"gr\"), val = tensor<int32, []>(1)];\n"
                "        tensor<fp16, [%d, %d, 1, 1]> W1t = const()[name = tensor<string, []>(\"W1t\"), "
                "val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = tensor<string, []>(\"@model_path/weights/w1t.bin\"), offset = tensor<uint64, []>(64)))];\n"
                "        tensor<fp16, [%d, %d, 1, 1]> W3t = const()[name = tensor<string, []>(\"W3t\"), "
                "val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = tensor<string, []>(\"@model_path/weights/w3t.bin\"), offset = tensor<uint64, []>(64)))];\n"
                "        tensor<fp16, [1, %d, 1, %d]> dx1 = conv(dilations = dl, groups = gr, pad = pd, "
                "pad_type = pt, strides = st, weight = W1t, x = dh1)[name = tensor<string, []>(\"cv1\")];\n"
                "        tensor<fp16, [1, %d, 1, %d]> dx3 = conv(dilations = dl, groups = gr, pad = pd, "
                "pad_type = pt, strides = st, weight = W3t, x = dh3)[name = tensor<string, []>(\"cv3\")];\n"
                "        tensor<fp16, [1, %d, 1, %d]> sum = add(x = dx1, y = dx3)[name = tensor<string, []>(\"ad\")];\n"
                "        tensor<string, []> d2 = const()[name = tensor<string, []>(\"d2\"), val = tensor<string, []>(\"fp32\")];\n"
                "        tensor<fp32, [1, %d, 1, %d]> y = cast(dtype = d2, x = sum)[name = tensor<string, []>(\"co\")];\n"
                "    } -> (y);\n}\n",
                HIDDEN*2, SEQ, HIDDEN*2, SEQ,
                HIDDEN, SEQ, HIDDEN, SEQ,
                HIDDEN, HIDDEN, SEQ, HIDDEN, SEQ,
                DIM, HIDDEN, DIM, HIDDEN,
                DIM, HIDDEN, DIM, HIDDEN,
                DIM, SEQ, DIM, SEQ,
                DIM, SEQ, DIM, SEQ];
        }

        NSDictionary *wd = @{
            @"@model_path/weights/w1t.bin": @{@"offset":@0, @"data":build_blob_t(W1, HIDDEN, DIM)},
            @"@model_path/weights/w3t.bin": @{@"offset":@0, @"data":build_blob_t(W3, HIDDEN, DIM)}
        };

        NSData *md = [mil dataUsingEncoding:NSUTF8StringEncoding];
        id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(g_D, @selector(modelWithMILText:weights:optionsPlist:), md, wd, nil);
        if (!desc) { printf("desc=NULL\n"); return 1; }
        id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(g_I, @selector(inMemoryModelWithDescriptor:), desc);
        id hx = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
        NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
        [[NSFileManager defaultManager] createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"] withIntermediateDirectories:YES attributes:nil error:nil];
        [md writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
        for (NSString *path in wd) {
            [wd[path][@"data"] writeToFile:[td stringByAppendingPathComponent:[path stringByReplacingOccurrencesOfString:@"@model_path/" withString:@""]] atomically:YES];
        }

        NSError *e = nil;
        BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl, @selector(compileWithQoS:options:error:), 21, @{}, &e);
        if (!ok && !g_fp16_io) {
            printf("[ANE] fp32 compile failed, retrying with fp16 I/O (M1/M2 fallback)\n");
            g_fp16_io = 1;
            [[NSFileManager defaultManager] removeItemAtPath:td error:nil];
            goto retry_compile;
        }
        printf("Compile: %s\n", ok?"OK":"FAIL");
        if (!ok) { printf("  %s\n", e?[[e description] UTF8String]:""); return 1; }
        ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e);
        printf("Load: %s\n", ok?"OK":"FAIL");
        if (!ok) return 1;

        // Prepare input: concat(dh1, dh3) in channel-first layout
        float *dh1 = (float*)malloc(SEQ*HIDDEN*sizeof(float));
        float *dh3 = (float*)malloc(SEQ*HIDDEN*sizeof(float));
        for (int i = 0; i < SEQ*HIDDEN; i++) { dh1[i]=0.01f*sinf(i*0.007f); dh3[i]=0.01f*cosf(i*0.011f); }

        size_t bpe = g_fp16_io ? 2 : 4;
        IOSurfaceRef ioI = make_surface(HIDDEN*2*SEQ*bpe), ioO = make_surface(DIM*SEQ*bpe);
        IOSurfaceLock(ioI, 0, NULL);
        if (g_fp16_io) {
            _Float16 *dst = (_Float16*)IOSurfaceGetBaseAddress(ioI);
            for (int t = 0; t < SEQ; t++) {
                for (int c = 0; c < HIDDEN; c++) dst[c*SEQ+t] = (_Float16)dh1[t*HIDDEN+c];
                for (int c = 0; c < HIDDEN; c++) dst[(HIDDEN+c)*SEQ+t] = (_Float16)dh3[t*HIDDEN+c];
            }
        } else {
            float *dst = (float*)IOSurfaceGetBaseAddress(ioI);
            for (int t = 0; t < SEQ; t++) {
                for (int c = 0; c < HIDDEN; c++) dst[c*SEQ+t] = dh1[t*HIDDEN+c];
                for (int c = 0; c < HIDDEN; c++) dst[(HIDDEN+c)*SEQ+t] = dh3[t*HIDDEN+c];
            }
        }
        IOSurfaceUnlock(ioI, 0, NULL);

        id wI = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioI);
        id wO = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioO);
        id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_AR,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            @[wI], @[@0], @[wO], @[@0], nil, nil, @0);

        ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
            mdl, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &e);
        printf("Eval: %s\n", ok?"OK":"FAIL");
        if (!ok) { printf("  %s\n", e?[[e description] UTF8String]:""); return 1; }

        // CPU reference: dx = W1^T @ dh1 + W3^T @ dh3
        float *ref = (float*)calloc(SEQ*DIM, sizeof(float));
        for (int t = 0; t < SEQ; t++)
            for (int i = 0; i < DIM; i++) {
                float s = 0;
                for (int j = 0; j < HIDDEN; j++) {
                    s += W1[j*DIM+i] * dh1[t*HIDDEN+j]; // W1^T[i,j] = W1[j,i]
                    s += W3[j*DIM+i] * dh3[t*HIDDEN+j];
                }
                ref[t*DIM+i] = s;
            }

        IOSurfaceLock(ioO, kIOSurfaceLockReadOnly, NULL);
        float maxd = 0;
        if (g_fp16_io) {
            _Float16 *src = (_Float16*)IOSurfaceGetBaseAddress(ioO);
            for (int t = 0; t < SEQ; t++)
                for (int c = 0; c < DIM; c++) {
                    float d = fabsf((float)src[c*SEQ+t] - ref[t*DIM+c]);
                    if (d > maxd) maxd = d;
                }
        } else {
            float *src = (float*)IOSurfaceGetBaseAddress(ioO);
            for (int t = 0; t < SEQ; t++)
                for (int c = 0; c < DIM; c++) {
                    float d = fabsf(src[c*SEQ+t] - ref[t*DIM+c]);
                    if (d > maxd) maxd = d;
                }
        }
        IOSurfaceUnlock(ioO, kIOSurfaceLockReadOnly, NULL);
        printf("dx max diff: %.6f\n", maxd);

        ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(mdl, @selector(unloadWithQoS:error:), 21, &e);
        [[NSFileManager defaultManager] removeItemAtPath:td error:nil];
        CFRelease(ioI); CFRelease(ioO);
        free(W1); free(W3); free(dh1); free(dh3); free(ref);
        printf("\nDONE\n");
    }
    return 0;
}
