// Debug: why causal mask doesn't apply. Try different approaches.
#import <Foundation/Foundation.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <IOSurface/IOSurface.h>
#include <math.h>

#define HEADS 12
#define HD 64
#define SEQ 8  // small for readable output

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

// Build inline mask string for MIL: tensor<fp16, [1,1,S,S]>([v00, v01, ...])
static NSString *build_inline_causal_mask(int s) {
    NSMutableString *vals = [NSMutableString string];
    for (int t = 0; t < s; t++) {
        for (int t2 = 0; t2 < s; t2++) {
            if (t > 0 || t2 > 0) [vals appendString:@", "];
            [vals appendString:(t2 <= t) ? @"0" : @"-65504"];  // fp16 -inf
        }
    }
    return [NSString stringWithFormat:
        @"tensor<fp16, [1, 1, %d, %d]>([%@])", s, s, vals];
}

static NSData *build_mask_blob(int seq) {
    int wsize = seq * seq * 2;
    int total = 128 + wsize;
    uint8_t *buf = (uint8_t*)calloc(total, 1);
    buf[0]=1; buf[4]=2; buf[64]=0xEF; buf[65]=0xBE; buf[66]=0xAD; buf[67]=0xDE; buf[68]=1;
    *(uint32_t*)(buf+72)=wsize; *(uint32_t*)(buf+80)=128;
    _Float16 *fp16 = (_Float16*)(buf+128);
    for (int t = 0; t < seq; t++)
        for (int t2 = 0; t2 < seq; t2++)
            fp16[t*seq + t2] = (t2 <= t) ? (_Float16)0.0f : (_Float16)(-65504.0f);
    return [NSData dataWithBytesNoCopy:buf length:total freeWhenDone:YES];
}

typedef struct { id model; NSString *td; } Model;

static Model compile_model(NSString *mil, NSDictionary *wd) {
    Model m = {nil, nil};
    NSData *md = [mil dataUsingEncoding:NSUTF8StringEncoding];
    id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(g_D, @selector(modelWithMILText:weights:optionsPlist:), md, wd ?: @{}, nil);
    if (!desc) { printf("  desc=NULL\n"); return m; }
    id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(g_I, @selector(inMemoryModelWithDescriptor:), desc);
    id hx = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
    NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
    [[NSFileManager defaultManager] createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"] withIntermediateDirectories:YES attributes:nil error:nil];
    [md writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
    for (NSString *path in wd) {
        [wd[path][@"data"] writeToFile:[td stringByAppendingPathComponent:[path stringByReplacingOccurrencesOfString:@"@model_path/" withString:@""]] atomically:YES];
    }
    NSError *e = nil;
    if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl, @selector(compileWithQoS:options:error:), 21, @{}, &e)) {
        printf("  compile FAIL: %s\n", e?[[[e localizedDescription] substringToIndex:MIN(300,(int)[[e localizedDescription] length])] UTF8String]:"");
        [[NSFileManager defaultManager] removeItemAtPath:td error:nil]; return m;
    }
    if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e)) {
        printf("  load FAIL\n"); [[NSFileManager defaultManager] removeItemAtPath:td error:nil]; return m;
    }
    m.model = mdl; m.td = td;
    return m;
}

static void cleanup_model(Model *m) {
    if (!m->model) return;
    NSError *e = nil;
    ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(m->model, @selector(unloadWithQoS:error:), 21, &e);
    [[NSFileManager defaultManager] removeItemAtPath:m->td error:nil];
}

int main() {
    @autoreleasepool {
        setbuf(stdout, NULL);
        ane_init();

        srand48(42);
        int total = HEADS * SEQ * HD;
        _Float16 *Q = (_Float16*)malloc(total * 2);
        _Float16 *K = (_Float16*)malloc(total * 2);
        _Float16 *V = (_Float16*)malloc(total * 2);
        for (int i = 0; i < total; i++) {
            Q[i] = (_Float16)(0.5f * (2*drand48()-1));
            K[i] = (_Float16)(0.5f * (2*drand48()-1));
            V[i] = (_Float16)(0.5f * (2*drand48()-1));
        }

        size_t bytes = total * 2;
        IOSurfaceRef ioQ = make_surface(bytes), ioK = make_surface(bytes);
        IOSurfaceRef ioV = make_surface(bytes);
        IOSurfaceLock(ioQ, 0, NULL); memcpy(IOSurfaceGetBaseAddress(ioQ), Q, bytes); IOSurfaceUnlock(ioQ, 0, NULL);
        IOSurfaceLock(ioK, 0, NULL); memcpy(IOSurfaceGetBaseAddress(ioK), K, bytes); IOSurfaceUnlock(ioK, 0, NULL);
        IOSurfaceLock(ioV, 0, NULL); memcpy(IOSurfaceGetBaseAddress(ioV), V, bytes); IOSurfaceUnlock(ioV, 0, NULL);
        id wQ = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioQ);
        id wK = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioK);
        id wV = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioV);

        // CPU references
        float scale = 1.0f / sqrtf((float)HD);
        float *cpu_causal = (float*)calloc(total, sizeof(float));
        float *cpu_nocausal = (float*)calloc(total, sizeof(float));
        for (int h = 0; h < HEADS; h++)
            for (int t = 0; t < SEQ; t++) {
                // Causal
                float scores[SEQ], maxs = -1e30f;
                for (int t2 = 0; t2 <= t; t2++) {
                    float s = 0;
                    for (int d = 0; d < HD; d++) s += (float)Q[h*SEQ*HD+t*HD+d]*(float)K[h*SEQ*HD+t2*HD+d];
                    s *= scale; scores[t2] = s; if(s>maxs) maxs=s;
                }
                float sum = 0;
                for (int t2 = 0; t2 <= t; t2++) { scores[t2]=expf(scores[t2]-maxs); sum+=scores[t2]; }
                for (int t2 = 0; t2 <= t; t2++) scores[t2]/=sum;
                for (int d = 0; d < HD; d++) {
                    float r = 0;
                    for (int t2 = 0; t2 <= t; t2++) r += scores[t2]*(float)V[h*SEQ*HD+t2*HD+d];
                    cpu_causal[h*SEQ*HD+t*HD+d] = r;
                }
                // Non-causal
                maxs = -1e30f;
                for (int t2 = 0; t2 < SEQ; t2++) {
                    float s = 0;
                    for (int d = 0; d < HD; d++) s += (float)Q[h*SEQ*HD+t*HD+d]*(float)K[h*SEQ*HD+t2*HD+d];
                    s *= scale; scores[t2] = s; if(s>maxs) maxs=s;
                }
                sum = 0;
                for (int t2 = 0; t2 < SEQ; t2++) { scores[t2]=expf(scores[t2]-maxs); sum+=scores[t2]; }
                for (int t2 = 0; t2 < SEQ; t2++) scores[t2]/=sum;
                for (int d = 0; d < HD; d++) {
                    float r = 0;
                    for (int t2 = 0; t2 < SEQ; t2++) r += scores[t2]*(float)V[h*SEQ*HD+t2*HD+d];
                    cpu_nocausal[h*SEQ*HD+t*HD+d] = r;
                }
            }

        // Helper: eval and compare
        void (^eval_and_compare)(const char*, Model*, int nInputs, IOSurfaceRef*) =
            ^(const char *label, Model *m, int nInputs, IOSurfaceRef *inputs) {
            IOSurfaceRef ioO = make_surface(bytes);
            id wO = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioO);
            NSMutableArray *inArr = [NSMutableArray array];
            NSMutableArray *inIdx = [NSMutableArray array];
            for (int i = 0; i < nInputs; i++) {
                [inArr addObject:((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), inputs[i])];
                [inIdx addObject:@(i)];
            }
            id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_AR,
                @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
                inArr, inIdx, @[wO], @[@0], nil, nil, @0);
            NSError *e = nil;
            BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                m->model, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &e);
            if (!ok) {
                printf("  %s: eval FAIL: %s\n", label, e?[[[e localizedDescription] substringToIndex:MIN(200,(int)[[e localizedDescription] length])] UTF8String]:"");
                CFRelease(ioO); return;
            }
            IOSurfaceLock(ioO, kIOSurfaceLockReadOnly, NULL);
            _Float16 *out = (_Float16*)IOSurfaceGetBaseAddress(ioO);
            float dc=0, dnc=0;
            for (int i = 0; i < total; i++) {
                float v = (float)out[i];
                float d1 = fabsf(v - cpu_causal[i]); if(d1>dc) dc=d1;
                float d2 = fabsf(v - cpu_nocausal[i]); if(d2>dnc) dnc=d2;
            }
            IOSurfaceUnlock(ioO, kIOSurfaceLockReadOnly, NULL);
            printf("  %s: diff_causal=%.6f diff_nocausal=%.6f → %s\n", label, dc, dnc,
                   dc < dnc ? "CAUSAL" : (dc > dnc ? "NON-CAUSAL" : "SAME"));
            CFRelease(ioO);
        };

        // === Test 1: No mask (should be non-causal) ===
        printf("Test 1: no mask\n");
        {
            NSString *mil = [NSString stringWithFormat:
                @"program(1.0)\n[buildInfo = dict<tensor<string, []>, tensor<string, []>>({{\"coremlc-version\", \"3505.4.1\"}})]\n{\n"
                "    func main<ios16>(tensor<fp16, [1, %d, %d, %d]> q, "
                "tensor<fp16, [1, %d, %d, %d]> k, tensor<fp16, [1, %d, %d, %d]> v) {\n"
                "        tensor<fp16, [1, %d, %d, %d]> att = scaled_dot_product_attention("
                "query = q, key = k, value = v)[name = tensor<string, []>(\"sdpa\")];\n"
                "    } -> (att);\n}\n",
                HEADS, SEQ, HD, HEADS, SEQ, HD, HEADS, SEQ, HD, HEADS, SEQ, HD];
            Model m = compile_model(mil, nil);
            if (m.model) {
                IOSurfaceRef ins[] = {ioQ, ioK, ioV};
                eval_and_compare("no-mask", &m, 3, ins);
                cleanup_model(&m);
            }
        }

        // === Test 2: Inline causal mask ===
        printf("\nTest 2: inline causal mask\n");
        {
            NSString *maskStr = build_inline_causal_mask(SEQ);
            NSString *mil = [NSString stringWithFormat:
                @"program(1.0)\n[buildInfo = dict<tensor<string, []>, tensor<string, []>>({{\"coremlc-version\", \"3505.4.1\"}})]\n{\n"
                "    func main<ios16>(tensor<fp16, [1, %d, %d, %d]> q, "
                "tensor<fp16, [1, %d, %d, %d]> k, tensor<fp16, [1, %d, %d, %d]> v) {\n"
                "        %@ mask = const()[name = tensor<string, []>(\"mask\"), val = %@];\n"
                "        tensor<fp16, [1, %d, %d, %d]> att = scaled_dot_product_attention("
                "query = q, key = k, value = v, attn_mask = mask)[name = tensor<string, []>(\"sdpa\")];\n"
                "    } -> (att);\n}\n",
                HEADS, SEQ, HD, HEADS, SEQ, HD, HEADS, SEQ, HD,
                [NSString stringWithFormat:@"tensor<fp16, [1, 1, %d, %d]>", SEQ, SEQ], maskStr,
                HEADS, SEQ, HD];
            Model m = compile_model(mil, nil);
            if (m.model) {
                IOSurfaceRef ins[] = {ioQ, ioK, ioV};
                eval_and_compare("inline-mask", &m, 3, ins);
                cleanup_model(&m);
            }
        }

        // === Test 3: BLOBFILE mask ===
        printf("\nTest 3: BLOBFILE causal mask\n");
        {
            NSString *mil = [NSString stringWithFormat:
                @"program(1.0)\n[buildInfo = dict<tensor<string, []>, tensor<string, []>>({{\"coremlc-version\", \"3505.4.1\"}})]\n{\n"
                "    func main<ios16>(tensor<fp16, [1, %d, %d, %d]> q, "
                "tensor<fp16, [1, %d, %d, %d]> k, tensor<fp16, [1, %d, %d, %d]> v) {\n"
                "        tensor<fp16, [1, 1, %d, %d]> mask = const()[name = tensor<string, []>(\"mask\"), "
                "val = tensor<fp16, [1, 1, %d, %d]>(BLOBFILE(path = tensor<string, []>(\"@model_path/weights/mask.bin\"), offset = tensor<uint64, []>(64)))];\n"
                "        tensor<fp16, [1, %d, %d, %d]> att = scaled_dot_product_attention("
                "query = q, key = k, value = v, attn_mask = mask)[name = tensor<string, []>(\"sdpa\")];\n"
                "    } -> (att);\n}\n",
                HEADS, SEQ, HD, HEADS, SEQ, HD, HEADS, SEQ, HD,
                SEQ, SEQ, SEQ, SEQ, HEADS, SEQ, HD];
            NSDictionary *wd = @{@"@model_path/weights/mask.bin": @{@"offset":@0, @"data":build_mask_blob(SEQ)}};
            Model m = compile_model(mil, wd);
            if (m.model) {
                IOSurfaceRef ins[] = {ioQ, ioK, ioV};
                eval_and_compare("blob-mask", &m, 3, ins);
                cleanup_model(&m);
            }
        }

        // === Test 4: mask as runtime input ===
        printf("\nTest 4: mask as runtime input\n");
        {
            NSString *mil = [NSString stringWithFormat:
                @"program(1.0)\n[buildInfo = dict<tensor<string, []>, tensor<string, []>>({{\"coremlc-version\", \"3505.4.1\"}})]\n{\n"
                "    func main<ios16>(tensor<fp16, [1, %d, %d, %d]> q, "
                "tensor<fp16, [1, %d, %d, %d]> k, tensor<fp16, [1, %d, %d, %d]> v, "
                "tensor<fp16, [1, 1, %d, %d]> mask) {\n"
                "        tensor<fp16, [1, %d, %d, %d]> att = scaled_dot_product_attention("
                "query = q, key = k, value = v, attn_mask = mask)[name = tensor<string, []>(\"sdpa\")];\n"
                "    } -> (att);\n}\n",
                HEADS, SEQ, HD, HEADS, SEQ, HD, HEADS, SEQ, HD,
                SEQ, SEQ, HEADS, SEQ, HD];
            Model m = compile_model(mil, nil);
            if (m.model) {
                // Create mask IOSurface
                size_t mbytes = SEQ * SEQ * 2;
                IOSurfaceRef ioM = make_surface(mbytes);
                IOSurfaceLock(ioM, 0, NULL);
                _Float16 *mp = (_Float16*)IOSurfaceGetBaseAddress(ioM);
                for (int t = 0; t < SEQ; t++)
                    for (int t2 = 0; t2 < SEQ; t2++)
                        mp[t*SEQ+t2] = (t2 <= t) ? (_Float16)0.0f : (_Float16)(-65504.0f);
                IOSurfaceUnlock(ioM, 0, NULL);

                IOSurfaceRef ins[] = {ioQ, ioK, ioV, ioM};
                eval_and_compare("runtime-mask", &m, 4, ins);
                CFRelease(ioM);
                cleanup_model(&m);
            }
        }

        CFRelease(ioQ); CFRelease(ioK); CFRelease(ioV);
        free(Q); free(K); free(V);
        free(cpu_causal); free(cpu_nocausal);
        printf("\nDONE\n");
    }
    return 0;
}
