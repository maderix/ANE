// test_ane_model.m — Experiments E-H: _ANEModel loading, ANECompiler, chaining, shared events
// Build: make test_ane_model && ./test_ane_model
#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <IOSurface/IOSurface.h>
#import <mach/mach_time.h>
#include <limits.h>

static mach_timebase_info_data_t g_tb;
static double tb_ms(uint64_t t) { return (double)t * g_tb.numer / g_tb.denom / 1e6; }
__attribute__((unused)) static int g_fp16_io = 1;

#pragma mark - Helpers

static void dump_class(const char *name) {
    Class cls = NSClassFromString([NSString stringWithUTF8String:name]);
    if (!cls) { printf("  %s: NOT FOUND\n", name); return; }
    printf("\n=== %s ===\n", name);

    unsigned int count;
    Method *methods = class_copyMethodList(object_getClass(cls), &count);
    if (count) printf("  Class methods (%u):\n", count);
    for (unsigned int i = 0; i < count; i++) {
        SEL s = method_getName(methods[i]);
        const char *enc = method_getTypeEncoding(methods[i]);
        printf("    + %s  [%s]\n", sel_getName(s), enc ? enc : "?");
    }
    free(methods);

    methods = class_copyMethodList(cls, &count);
    if (count) printf("  Instance methods (%u):\n", count);
    for (unsigned int i = 0; i < count; i++) {
        SEL s = method_getName(methods[i]);
        const char *enc = method_getTypeEncoding(methods[i]);
        printf("    - %s  [%s]\n", sel_getName(s), enc ? enc : "?");
    }
    free(methods);

    unsigned int pcount;
    objc_property_t *props = class_copyPropertyList(cls, &pcount);
    if (pcount) printf("  Properties (%u):\n", pcount);
    for (unsigned int i = 0; i < pcount; i++) {
        const char *pname = property_getName(props[i]);
        const char *pattr = property_getAttributes(props[i]);
        printf("    @property %s  [%s]\n", pname, pattr ? pattr : "?");
    }
    free(props);
}

static void dump_all_properties(id obj, Class cls) {
    if (!obj) return;
    unsigned int pcount;
    objc_property_t *props = class_copyPropertyList(cls, &pcount);
    for (unsigned int i = 0; i < pcount; i++) {
        const char *pname = property_getName(props[i]);
        @try {
            id val = [obj valueForKey:[NSString stringWithUTF8String:pname]];
            printf("    %s = %s\n", pname, val ? [[val description] UTF8String] : "nil");
        } @catch (NSException *ex) {
            printf("    %s = <exception: %s>\n", pname, [[ex reason] UTF8String]);
        }
    }
    free(props);
}

static void list_dir_recursive(NSString *path, int depth) {
    NSFileManager *fm = [NSFileManager defaultManager];
    NSArray *items = [fm contentsOfDirectoryAtPath:path error:nil];
    for (NSString *item in items) {
        NSString *full = [path stringByAppendingPathComponent:item];
        BOOL isDir = NO;
        [fm fileExistsAtPath:full isDirectory:&isDir];
        NSDictionary *attrs = [fm attributesOfItemAtPath:full error:nil];
        unsigned long long sz = [attrs fileSize];
        for (int i = 0; i < depth; i++) printf("  ");
        if (isDir) {
            printf("  [DIR] %s/\n", [item UTF8String]);
            list_dir_recursive(full, depth + 1);
        } else {
            printf("  %s (%llu bytes)\n", [item UTF8String], sz);
        }
    }
}

static IOSurfaceRef make_surface(size_t bytes) {
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth:@(bytes), (id)kIOSurfaceHeight:@1,
        (id)kIOSurfaceBytesPerElement:@1, (id)kIOSurfaceBytesPerRow:@(bytes),
        (id)kIOSurfaceAllocSize:@(bytes), (id)kIOSurfacePixelFormat:@0});
}

#pragma mark - MIL Generation (FP16 conv)

static NSString *gen_conv_mil(int ch, int sp) {
    return [NSString stringWithFormat:
        @"program(1.0)\n[buildInfo = dict<tensor<string, []>, tensor<string, []>>"
        "({{\"coremlc-version\", \"3505.4.1\"}})]\n{\n"
        "    func main<ios16>(tensor<fp16, [1, %d, 1, %d]> x) {\n"
        "        tensor<string, []> pt = const()[name=tensor<string, []>(\"pt\"),"
        " val=tensor<string, []>(\"valid\")];\n"
        "        tensor<int32, [2]> st = const()[name=tensor<string, []>(\"st\"),"
        " val=tensor<int32, [2]>([1,1])];\n"
        "        tensor<int32, [4]> pd = const()[name=tensor<string, []>(\"pd\"),"
        " val=tensor<int32, [4]>([0,0,0,0])];\n"
        "        tensor<int32, [2]> dl = const()[name=tensor<string, []>(\"dl\"),"
        " val=tensor<int32, [2]>([1,1])];\n"
        "        tensor<int32, []> gr = const()[name=tensor<string, []>(\"gr\"),"
        " val=tensor<int32, []>(1)];\n"
        "        tensor<fp16, [%d,%d,1,1]> W = const()[name=tensor<string, []>(\"W\"), "
        "val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=tensor<string, []>"
        "(\"@model_path/weights/weight.bin\"), offset=tensor<uint64, []>(64)))];\n"
        "        tensor<fp16, [1,%d,1,%d]> y = conv(dilations=dl,groups=gr,"
        "pad=pd,pad_type=pt,strides=st,weight=W,x=x)"
        "[name=tensor<string, []>(\"conv\")];\n"
        "    } -> (y);\n}\n", ch, sp, ch, ch, ch, ch, ch, sp];
}

#pragma mark - Kernel Compilation

typedef struct {
    id model;
    IOSurfaceRef ioIn, ioOut;
    NSString *tmpDir;
    NSString *hexId;
    int ch, sp;
} CompiledKernel;

static CompiledKernel compile_kernel(int ch, int sp) {
    CompiledKernel k = {0};
    k.ch = ch; k.sp = sp;

    Class gD = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    Class gI = NSClassFromString(@"_ANEInMemoryModel");
    if (!gD || !gI) { printf("  ERROR: ANE classes not found\n"); return k; }

    int ws = ch * ch * 2;
    int tot = 128 + ws;
    uint8_t *blob = (uint8_t *)calloc((size_t)tot, 1);
    blob[0] = 1; blob[4] = 2;
    blob[64] = 0xEF; blob[65] = 0xBE; blob[66] = 0xAD; blob[67] = 0xDE;
    blob[68] = 1;
    *(uint32_t *)(blob + 72) = (uint32_t)ws;
    *(uint32_t *)(blob + 80) = 128;
    _Float16 *wp = (_Float16 *)(blob + 128);
    for (int i = 0; i < ch; i++) wp[i * ch + i] = (_Float16)1.0f;
    NSData *wdata = [NSData dataWithBytesNoCopy:blob length:(NSUInteger)tot
                                   freeWhenDone:YES];

    NSString *mil = gen_conv_mil(ch, sp);
    NSData *md = [mil dataUsingEncoding:NSUTF8StringEncoding];

    id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(gD,
        @selector(modelWithMILText:weights:optionsPlist:),
        md, @{@"@model_path/weights/weight.bin": @{@"offset":@0, @"data":wdata}}, nil);
    id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(gI,
        @selector(inMemoryModelWithDescriptor:), desc);
    if (!mdl) { printf("  ERROR: inMemoryModel creation failed\n"); return k; }

    id hx = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
    k.hexId = hx;
    NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
    NSFileManager *fm = [NSFileManager defaultManager];
    [fm createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"]
        withIntermediateDirectories:YES attributes:nil error:nil];
    [md writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
    [wdata writeToFile:[td stringByAppendingPathComponent:@"weights/weight.bin"]
        atomically:YES];

    NSError *e = nil;
    BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
        mdl, @selector(compileWithQoS:options:error:), 21, @{}, &e);
    if (!ok) {
        printf("  Compile failed: %s\n", e ? [[e description] UTF8String] : "unknown");
        return k;
    }

    ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
        mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e);
    if (!ok) {
        printf("  Load failed: %s\n", e ? [[e description] UTF8String] : "unknown");
        return k;
    }

    k.model = mdl;
    k.ioIn = make_surface((size_t)ch * sp * 2);
    k.ioOut = make_surface((size_t)ch * sp * 2);
    k.tmpDir = td;
    return k;
}

static void free_kernel(CompiledKernel *k) {
    if (!k->model) return;
    NSError *e = nil;
    ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(
        k->model, @selector(unloadWithQoS:error:), 21, &e);
    if (k->ioIn) CFRelease(k->ioIn);
    if (k->ioOut) CFRelease(k->ioOut);
}

#pragma mark - Main

int main(int argc, const char *argv[]) {
    (void)argc; (void)argv;
    @autoreleasepool {
        mach_timebase_info(&g_tb);
        printf("==============================================================\n");
        printf("  ANE Experiments E-H: _ANEModel, Compiler, Chaining\n");
        printf("==============================================================\n\n");

        void *handle = dlopen(
            "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/"
            "AppleNeuralEngine", RTLD_NOW);
        if (!handle) { printf("FATAL: dlopen ANE framework failed\n"); return 1; }

        Class gAIO = NSClassFromString(@"_ANEIOSurfaceObject");
        Class gAR = NSClassFromString(@"_ANERequest");
        Class gBuf = NSClassFromString(@"_ANEBuffer");
        Class gOutSets = NSClassFromString(@"_ANEIOSurfaceOutputSets");
        Class gChain = NSClassFromString(@"_ANEChainingRequest");
        id client = [NSClassFromString(@"_ANEClient")
            performSelector:@selector(sharedConnection)];

        printf("=== Compiling test kernels (64x32 FP16 conv) ===\n");
        CompiledKernel k1 = compile_kernel(64, 32);
        CompiledKernel k2 = compile_kernel(64, 32);
        if (!k1.model || !k2.model) {
            printf("FATAL: Kernel compilation failed\n");
            return 1;
        }
        printf("  k1: hexId=%s\n       tmpDir=%s\n",
               [k1.hexId UTF8String], [k1.tmpDir UTF8String]);
        printf("  k2: hexId=%s\n       tmpDir=%s\n",
               [k2.hexId UTF8String], [k2.tmpDir UTF8String]);

        // =================================================================
        // EXPERIMENT E: Load _ANEModel from compiled temp directory
        // =================================================================
        printf("\n------------------------------------------------------------\n");
        printf("  EXPERIMENT E: Load _ANEModel from compiled temp dir\n");
        printf("------------------------------------------------------------\n");

        id diskModel1 = nil;
        id diskModel2 = nil;
        Class gANEModel = NSClassFromString(@"_ANEModel");

        if (!gANEModel) {
            printf("  FATAL: _ANEModel class not found\n");
        } else {
            printf("\n  --- E.1: Full _ANEModel API surface ---\n");
            dump_class("_ANEModel");

            printf("\n  --- E.2: Temp dir contents (k1) ---\n");
            printf("  Path: %s\n", [k1.tmpDir UTF8String]);
            list_dir_recursive(k1.tmpDir, 0);

            printf("\n  --- E.3: Factory method probing ---\n");

            unsigned int mcount;
            Method *cmethods = class_copyMethodList(
                object_getClass(gANEModel), &mcount);
            printf("  All _ANEModel class methods (%u):\n", mcount);
            for (unsigned int i = 0; i < mcount; i++) {
                printf("    + %s\n", sel_getName(method_getName(cmethods[i])));
            }
            free(cmethods);

            NSURL *dirURL1 = [NSURL fileURLWithPath:k1.tmpDir];
            NSString *hexKey1 = k1.hexId;
            printf("\n  URL: %s\n", [[dirURL1 absoluteString] UTF8String]);
            printf("  key (hexId): %s\n", [hexKey1 UTF8String]);

            // E.3a: modelAtURL:key:
            printf("\n  --- E.3a: modelAtURL:key: ---\n");
            @try {
                SEL sel = NSSelectorFromString(@"modelAtURL:key:");
                if ([gANEModel respondsToSelector:sel]) {
                    diskModel1 = ((id(*)(Class,SEL,id,id))objc_msgSend)(
                        gANEModel, sel, dirURL1, hexKey1);
                    printf("  Result: %s\n",
                           diskModel1 ? [[diskModel1 description] UTF8String] : "nil");
                } else {
                    printf("  NOT available\n");
                }
            } @catch (NSException *ex) {
                printf("  EXCEPTION: %s\n", [[ex reason] UTF8String]);
            }

            // E.3b: modelAtURL:key:modelAttributes:
            if (!diskModel1) {
                printf("\n  --- E.3b: modelAtURL:key:modelAttributes: ---\n");
                @try {
                    SEL sel = NSSelectorFromString(
                        @"modelAtURL:key:modelAttributes:");
                    if ([gANEModel respondsToSelector:sel]) {
                        diskModel1 = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
                            gANEModel, sel, dirURL1, hexKey1, @{});
                        printf("  empty attrs: %s\n",
                               diskModel1 ? [[diskModel1 description] UTF8String]
                                          : "nil");
                        if (!diskModel1) {
                            diskModel1 = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
                                gANEModel, sel, dirURL1, hexKey1, nil);
                            printf("  nil attrs: %s\n",
                                   diskModel1 ? [[diskModel1 description] UTF8String]
                                              : "nil");
                        }
                    } else {
                        printf("  NOT available\n");
                    }
                } @catch (NSException *ex) {
                    printf("  EXCEPTION: %s\n", [[ex reason] UTF8String]);
                }
            }

            // E.3c: modelWithCacheURLIdentifier:
            if (!diskModel1) {
                printf("\n  --- E.3c: modelWithCacheURLIdentifier: ---\n");
                @try {
                    SEL sel = NSSelectorFromString(@"modelWithCacheURLIdentifier:");
                    if ([gANEModel respondsToSelector:sel]) {
                        diskModel1 = ((id(*)(Class,SEL,id))objc_msgSend)(
                            gANEModel, sel, hexKey1);
                        printf("  hexId: %s\n",
                               diskModel1 ? [[diskModel1 description] UTF8String]
                                          : "nil");
                        if (!diskModel1) {
                            diskModel1 = ((id(*)(Class,SEL,id))objc_msgSend)(
                                gANEModel, sel, k1.tmpDir);
                            printf("  tmpDir: %s\n",
                                   diskModel1 ? [[diskModel1 description] UTF8String]
                                              : "nil");
                        }
                    } else {
                        printf("  NOT available\n");
                    }
                } @catch (NSException *ex) {
                    printf("  EXCEPTION: %s\n", [[ex reason] UTF8String]);
                }
            }

            // E.3d: modelAtURLWithSourceURL:sourceURL:key:cacheURLIdentifier:
            if (!diskModel1) {
                printf("\n  --- E.3d: modelAtURLWithSourceURL:... ---\n");
                @try {
                    SEL sel = NSSelectorFromString(
                        @"modelAtURLWithSourceURL:sourceURL:key:cacheURLIdentifier:");
                    if ([gANEModel respondsToSelector:sel]) {
                        diskModel1 = ((id(*)(Class,SEL,id,id,id,id))objc_msgSend)(
                            gANEModel, sel, dirURL1, dirURL1, hexKey1, hexKey1);
                        printf("  Result: %s\n",
                               diskModel1 ? [[diskModel1 description] UTF8String]
                                          : "nil");
                    } else {
                        printf("  NOT available\n");
                    }
                } @catch (NSException *ex) {
                    printf("  EXCEPTION: %s\n", [[ex reason] UTF8String]);
                }
            }

            // E.3e: alloc/init variants
            if (!diskModel1) {
                printf("\n  --- E.3e: alloc/init variants ---\n");
                unsigned int imcount;
                Method *imethods = class_copyMethodList(gANEModel, &imcount);
                printf("  Init-like instance methods:\n");
                for (unsigned int i = 0; i < imcount; i++) {
                    const char *mname = sel_getName(method_getName(imethods[i]));
                    if (strstr(mname, "init") || strstr(mname, "Init")) {
                        printf("    - %s  [%s]\n", mname,
                               method_getTypeEncoding(imethods[i]));
                    }
                }
                free(imethods);

                @try {
                    SEL initSel = NSSelectorFromString(@"initWithURL:key:");
                    if ([gANEModel instancesRespondToSelector:initSel]) {
                        id obj = [gANEModel alloc];
                        diskModel1 = ((id(*)(id,SEL,id,id))objc_msgSend)(
                            obj, initSel, dirURL1, hexKey1);
                        printf("  initWithURL:key: %s\n",
                               diskModel1 ? [[diskModel1 description] UTF8String]
                                          : "nil");
                    }
                } @catch (NSException *ex) {
                    printf("  initWithURL:key: EXCEPTION: %s\n",
                           [[ex reason] UTF8String]);
                }
            }

            // E.3f: Search for .hwx files
            if (!diskModel1) {
                printf("\n  --- E.3f: Search for .hwx / .plist files ---\n");
                NSFileManager *fm = [NSFileManager defaultManager];
                NSDirectoryEnumerator *dirEnum = [fm enumeratorAtPath:k1.tmpDir];
                NSString *file;
                while ((file = [dirEnum nextObject])) {
                    NSString *ext = [file pathExtension];
                    if ([ext isEqualToString:@"hwx"] ||
                        [ext isEqualToString:@"plist"] ||
                        [ext isEqualToString:@"espresso"]) {
                        NSString *fp = [k1.tmpDir
                            stringByAppendingPathComponent:file];
                        NSDictionary *attrs = [fm attributesOfItemAtPath:fp
                                                                  error:nil];
                        printf("  Found: %s (%llu bytes)\n",
                               [file UTF8String], [attrs fileSize]);
                    }
                }

                NSString *netPlist = [k1.tmpDir
                    stringByAppendingPathComponent:@"net.plist"];
                if ([fm fileExistsAtPath:netPlist]) {
                    printf("  net.plist found! Reading...\n");
                    @try {
                        NSDictionary *plist = [NSDictionary
                            dictionaryWithContentsOfFile:netPlist];
                        if (plist) {
                            printf("  net.plist keys: %s\n",
                                   [[[plist allKeys] description] UTF8String]);
                        } else {
                            NSData *raw = [NSData dataWithContentsOfFile:netPlist];
                            printf("  net.plist: binary (%lu bytes)\n",
                                   (unsigned long)raw.length);
                        }
                    } @catch (NSException *ex) {
                        printf("  net.plist EXCEPTION: %s\n",
                               [[ex reason] UTF8String]);
                    }
                }
            }

            // E.3g: Try constructing from programHandle
            if (!diskModel1) {
                printf("\n  --- E.3g: programHandle-based construction ---\n");
                @try {
                    id progHandle = [k1.model valueForKey:@"programHandle"];
                    printf("  k1 programHandle = %s\n",
                           progHandle ? [[progHandle description] UTF8String]
                                      : "nil");
                    unsigned int mct;
                    Method *cls_m = class_copyMethodList(
                        object_getClass(gANEModel), &mct);
                    for (unsigned int i = 0; i < mct; i++) {
                        const char *mn = sel_getName(method_getName(cls_m[i]));
                        if (strstr(mn, "Handle") || strstr(mn, "handle") ||
                            strstr(mn, "program") || strstr(mn, "Program")) {
                            printf("  Relevant factory: +%s\n", mn);
                        }
                    }
                    free(cls_m);
                } @catch (NSException *ex) {
                    printf("  EXCEPTION: %s\n", [[ex reason] UTF8String]);
                }
            }

            // E.4: If loaded, query critical methods
            if (diskModel1) {
                printf("\n  ======================================================\n");
                printf("  _ANEModel LOADED SUCCESSFULLY!\n");
                printf("  ======================================================\n");

                printf("\n  --- E.4a: All properties ---\n");
                dump_all_properties(diskModel1, gANEModel);

                printf("\n  --- E.4b: getUUID ---\n");
                @try {
                    SEL uuidSel = NSSelectorFromString(@"getUUID");
                    if ([diskModel1 respondsToSelector:uuidSel]) {
                        id uuid = ((id(*)(id,SEL))objc_msgSend)(
                            diskModel1, uuidSel);
                        printf("  getUUID: %s\n",
                               uuid ? [[uuid description] UTF8String] : "nil");
                    } else {
                        printf("  getUUID: NOT available\n");
                    }
                } @catch (NSException *ex) {
                    printf("  getUUID EXCEPTION: %s\n",
                           [[ex reason] UTF8String]);
                }

                printf("\n  --- E.4c: Symbol indices ---\n");
                @try {
                    SEL inSel = NSSelectorFromString(
                        @"inputSymbolIndicesForProcedureIndex:");
                    if ([diskModel1 respondsToSelector:inSel]) {
                        id idx = ((id(*)(id,SEL,NSUInteger))objc_msgSend)(
                            diskModel1, inSel, (NSUInteger)0);
                        printf("  inputSymbolIndices(0): %s\n",
                               idx ? [[idx description] UTF8String] : "nil");
                    }
                    SEL outSel = NSSelectorFromString(
                        @"outputSymbolIndicesForProcedureIndex:");
                    if ([diskModel1 respondsToSelector:outSel]) {
                        id idx = ((id(*)(id,SEL,NSUInteger))objc_msgSend)(
                            diskModel1, outSel, (NSUInteger)0);
                        printf("  outputSymbolIndices(0): %s\n",
                               idx ? [[idx description] UTF8String] : "nil");
                    }
                } @catch (NSException *ex) {
                    printf("  EXCEPTION: %s\n", [[ex reason] UTF8String]);
                }

                printf("\n  --- E.4d: mapper ---\n");
                @try {
                    id mapper = [diskModel1 valueForKey:@"mapper"];
                    printf("  mapper: %s\n",
                           mapper ? [[mapper description] UTF8String] : "nil");
                } @catch (NSException *ex) {
                    printf("  EXCEPTION: %s\n", [[ex reason] UTF8String]);
                }

                printf("\n  --- E.4e: program ---\n");
                @try {
                    id prog = [diskModel1 valueForKey:@"program"];
                    printf("  program: %s\n",
                           prog ? [[prog description] UTF8String] : "nil");
                } @catch (NSException *ex) {
                    printf("  EXCEPTION: %s\n", [[ex reason] UTF8String]);
                }

                // E.4f: Copy programHandle + program from InMemoryModel
                printf("\n  --- E.4f: Populate _ANEModel from InMemoryModel ---\n");
                @try {
                    id imProgHandle = [k1.model valueForKey:@"programHandle"];
                    id imProgram = [k1.model valueForKey:@"program"];
                    id imMapper = nil;
                    @try { imMapper = [k1.model valueForKey:@"mapper"]; }
                    @catch (NSException *ex) { (void)ex; }

                    printf("  InMemoryModel programHandle: %s\n",
                           imProgHandle ? [[imProgHandle description] UTF8String]
                                        : "nil");
                    printf("  InMemoryModel program: %s\n",
                           imProgram ? [[imProgram description] UTF8String]
                                     : "nil");

                    if (imProgHandle) {
                        uint64_t ph = [imProgHandle unsignedLongLongValue];
                        ((void(*)(id,SEL,uint64_t))objc_msgSend)(
                            diskModel1,
                            @selector(setProgramHandle:), ph);
                        printf("  Set programHandle on _ANEModel: %llu\n", ph);
                    }
                    if (imProgram) {
                        ((void(*)(id,SEL,id))objc_msgSend)(
                            diskModel1, @selector(setProgram:), imProgram);
                        printf("  Set program on _ANEModel\n");
                    }

                    // Verify
                    id newPH = [diskModel1 valueForKey:@"programHandle"];
                    id newProg = [diskModel1 valueForKey:@"program"];
                    printf("  _ANEModel programHandle now: %s\n",
                           newPH ? [[newPH description] UTF8String] : "nil");
                    printf("  _ANEModel program now: %s\n",
                           newProg ? [[newProg description] UTF8String] : "nil");

                    // Re-check symbol indices after populating
                    printf("\n  Re-checking symbol indices...\n");
                    SEL inSel2 = NSSelectorFromString(
                        @"inputSymbolIndicesForProcedureIndex:");
                    if ([diskModel1 respondsToSelector:inSel2]) {
                        id idx = ((id(*)(id,SEL,unsigned int))objc_msgSend)(
                            diskModel1, inSel2, (unsigned int)0);
                        printf("  inputSymbolIndices(0): %s\n",
                               idx ? [[idx description] UTF8String] : "nil");
                    }
                    SEL outSel2 = NSSelectorFromString(
                        @"outputSymbolIndicesForProcedureIndex:");
                    if ([diskModel1 respondsToSelector:outSel2]) {
                        id idx = ((id(*)(id,SEL,unsigned int))objc_msgSend)(
                            diskModel1, outSel2, (unsigned int)0);
                        printf("  outputSymbolIndices(0): %s\n",
                               idx ? [[idx description] UTF8String] : "nil");
                    }

                    // Try getUUID again
                    id uuid2 = ((id(*)(id,SEL))objc_msgSend)(
                        diskModel1, NSSelectorFromString(@"getUUID"));
                    printf("  getUUID after populate: %s\n",
                           uuid2 ? [[uuid2 description] UTF8String] : "nil");
                } @catch (NSException *ex) {
                    printf("  Populate EXCEPTION: %s\n",
                           [[ex reason] UTF8String]);
                }

                // Also load k2 and populate it
                printf("\n  --- E.5: Loading k2 as _ANEModel ---\n");
                NSURL *dirURL2 = [NSURL fileURLWithPath:k2.tmpDir];
                @try {
                    SEL sel = NSSelectorFromString(@"modelAtURL:key:");
                    if ([gANEModel respondsToSelector:sel]) {
                        diskModel2 = ((id(*)(Class,SEL,id,id))objc_msgSend)(
                            gANEModel, sel, dirURL2, k2.hexId);
                        printf("  k2 _ANEModel: %s\n",
                               diskModel2 ? "LOADED" : "nil");
                        if (diskModel2) {
                            id k2ph = [k2.model valueForKey:@"programHandle"];
                            id k2prog = [k2.model valueForKey:@"program"];
                            if (k2ph) {
                                ((void(*)(id,SEL,uint64_t))objc_msgSend)(
                                    diskModel2, @selector(setProgramHandle:),
                                    [k2ph unsignedLongLongValue]);
                            }
                            if (k2prog) {
                                ((void(*)(id,SEL,id))objc_msgSend)(
                                    diskModel2, @selector(setProgram:),
                                    k2prog);
                            }
                            printf("  k2 populated with programHandle + program\n");
                        }
                    }
                } @catch (NSException *ex) {
                    printf("  EXCEPTION: %s\n", [[ex reason] UTF8String]);
                }
            } else {
                printf("\n  _ANEModel could NOT be loaded via any factory.\n");
                printf("  Proceeding to Experiment E2 (ANECompiler).\n");
            }
        }

        // =================================================================
        // EXPERIMENT E2: ANECompiler probing
        // =================================================================
        printf("\n------------------------------------------------------------\n");
        printf("  EXPERIMENT E2: ANECompiler / model.hwx generation\n");
        printf("------------------------------------------------------------\n");
        {
            printf("\n  --- E2.1: Looking for ANECompiler ---\n");
            const char *compiler_paths[] = {
                "/System/Library/PrivateFrameworks/"
                "ANECompiler.framework/ANECompiler",
                "/System/Library/PrivateFrameworks/"
                "ANECompiler.framework/Versions/Current/ANECompiler",
                "/System/Library/Frameworks/"
                "CoreML.framework/Versions/A/CoreML",
                NULL
            };
            void *compilerHandle = NULL;
            for (int i = 0; compiler_paths[i]; i++) {
                compilerHandle = dlopen(compiler_paths[i], RTLD_NOW);
                if (compilerHandle) {
                    printf("  Found: %s\n", compiler_paths[i]);
                    break;
                } else {
                    printf("  Not at: %s\n", compiler_paths[i]);
                }
            }

            printf("\n  --- E2.2: Compiler class search ---\n");
            const char *compiler_classes[] = {
                "ANECompiler", "_ANECompiler", "ANECompilerService",
                "_ANECompilerService", "ANECompileOptions",
                "_ANECompileOptions", "ANEModelCompiler",
                "_ANEModelCompiler", "ANECCompiler", NULL
            };
            for (int i = 0; compiler_classes[i]; i++) {
                Class cls = NSClassFromString(
                    [NSString stringWithUTF8String:compiler_classes[i]]);
                if (cls) {
                    printf("  FOUND: %s\n", compiler_classes[i]);
                    dump_class(compiler_classes[i]);
                } else {
                    printf("  %s: not found\n", compiler_classes[i]);
                }
            }

            printf("\n  --- E2.3: Compile with debug_mask ---\n");
            @try {
                Class gD = NSClassFromString(@"_ANEInMemoryModelDescriptor");
                Class gI = NSClassFromString(@"_ANEInMemoryModel");
                int dbg_ch = 32, dbg_ws = dbg_ch * dbg_ch * 2;
                int dbg_tot = 128 + dbg_ws;
                uint8_t *dbg_blob = (uint8_t *)calloc((size_t)dbg_tot, 1);
                dbg_blob[0] = 1; dbg_blob[4] = 2;
                dbg_blob[64] = 0xEF; dbg_blob[65] = 0xBE;
                dbg_blob[66] = 0xAD; dbg_blob[67] = 0xDE;
                dbg_blob[68] = 1;
                *(uint32_t *)(dbg_blob + 72) = (uint32_t)dbg_ws;
                *(uint32_t *)(dbg_blob + 80) = 128;
                _Float16 *dbg_wp = (_Float16 *)(dbg_blob + 128);
                for (int i = 0; i < dbg_ch; i++)
                    dbg_wp[i * dbg_ch + i] = (_Float16)1.0f;
                NSData *wdata = [NSData dataWithBytesNoCopy:dbg_blob
                    length:(NSUInteger)dbg_tot freeWhenDone:YES];

                NSString *mil = gen_conv_mil(32, 16);
                NSData *md = [mil dataUsingEncoding:NSUTF8StringEncoding];
                id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(gD,
                    @selector(modelWithMILText:weights:optionsPlist:),
                    md,
                    @{@"@model_path/weights/weight.bin":
                        @{@"offset":@0, @"data":wdata}},
                    nil);
                id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(gI,
                    @selector(inMemoryModelWithDescriptor:), desc);

                id hx = ((id(*)(id,SEL))objc_msgSend)(mdl,
                    @selector(hexStringIdentifier));
                NSString *td = [NSTemporaryDirectory()
                    stringByAppendingPathComponent:
                        [NSString stringWithFormat:@"debug_%@", hx]];
                NSFileManager *fm = [NSFileManager defaultManager];
                [fm createDirectoryAtPath:
                    [td stringByAppendingPathComponent:@"weights"]
                    withIntermediateDirectories:YES attributes:nil error:nil];
                [md writeToFile:[td stringByAppendingPathComponent:@"model.mil"]
                    atomically:YES];
                [wdata writeToFile:
                    [td stringByAppendingPathComponent:@"weights/weight.bin"]
                    atomically:YES];

                NSDictionary *debugOpts = @{
                    @"debug_mask": @(INT_MAX),
                    @"ANEDebugMask": @(INT_MAX),
                    @"ane_debug_mask": @(INT_MAX),
                };
                printf("  Compiling with debug_mask=%d...\n", INT_MAX);

                NSError *e = nil;
                BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))
                    objc_msgSend)(mdl,
                    @selector(compileWithQoS:options:error:),
                    21, debugOpts, &e);
                printf("  Compile: %s\n", ok ? "SUCCESS" : "FAILED");
                if (!ok && e)
                    printf("  Error: %s\n", [[e description] UTF8String]);

                if (ok) {
                    printf("  Temp dir after debug compile:\n");
                    list_dir_recursive(td, 0);

                    NSDirectoryEnumerator *de = [fm enumeratorAtPath:td];
                    NSString *f;
                    int hwxCount = 0;
                    while ((f = [de nextObject])) {
                        if ([[f pathExtension] isEqualToString:@"hwx"])
                            hwxCount++;
                    }
                    if (hwxCount > 0)
                        printf("  Found %d .hwx file(s)!\n", hwxCount);

                    if (gANEModel && !diskModel1) {
                        @try {
                            SEL sel = NSSelectorFromString(@"modelAtURL:key:");
                            if ([gANEModel respondsToSelector:sel]) {
                                NSURL *dURL = [NSURL fileURLWithPath:td];
                                diskModel1 = ((id(*)(Class,SEL,id,id))
                                    objc_msgSend)(gANEModel, sel, dURL, hx);
                                printf("  modelAtURL:key: on debug dir: %s\n",
                                       diskModel1
                                           ? [[diskModel1 description] UTF8String]
                                           : "nil");
                            }
                        } @catch (NSException *ex) {
                            printf("  EXCEPTION: %s\n",
                                   [[ex reason] UTF8String]);
                        }
                    }
                }
                [fm removeItemAtPath:td error:nil];
            } @catch (NSException *ex) {
                printf("  E2.3 EXCEPTION: %s\n", [[ex reason] UTF8String]);
            }

            printf("\n  --- E2.4: ane_compiler_service search ---\n");
            {
                NSFileManager *fm = [NSFileManager defaultManager];
                const char *svc_paths[] = {
                    "/usr/libexec/ane_compiler_service",
                    "/System/Library/CoreServices/ane_compiler_service",
                    "/System/Library/PrivateFrameworks/"
                    "ANECompiler.framework/ane_compiler_service",
                    NULL
                };
                for (int i = 0; svc_paths[i]; i++) {
                    NSString *p = [NSString stringWithUTF8String:svc_paths[i]];
                    printf("  %s: %s\n", svc_paths[i],
                           [fm fileExistsAtPath:p] ? "FOUND" : "not found");
                }
            }

            printf("\n  --- E2.5: _ANEInMemoryModel compilation methods ---\n");
            {
                Class gI = NSClassFromString(@"_ANEInMemoryModel");
                unsigned int mc;
                Method *ms = class_copyMethodList(gI, &mc);
                for (unsigned int i = 0; i < mc; i++) {
                    const char *mn = sel_getName(method_getName(ms[i]));
                    if (strstr(mn, "compile") || strstr(mn, "Compile") ||
                        strstr(mn, "hwx") || strstr(mn, "HWX") ||
                        strstr(mn, "binary") || strstr(mn, "Binary") ||
                        strstr(mn, "save") || strstr(mn, "Save") ||
                        strstr(mn, "export") || strstr(mn, "Export") ||
                        strstr(mn, "path") || strstr(mn, "Path") ||
                        strstr(mn, "url") || strstr(mn, "URL") ||
                        strstr(mn, "temp") || strstr(mn, "Temp") ||
                        strstr(mn, "cache") || strstr(mn, "Cache")) {
                        printf("    - %s [%s]\n", mn,
                               method_getTypeEncoding(ms[i]));
                    }
                }
                free(ms);
            }
        }

        // =================================================================
        // EXPERIMENT F: Full chaining pipeline with _ANEModel
        // =================================================================
        printf("\n------------------------------------------------------------\n");
        printf("  EXPERIMENT F: Full chaining pipeline\n");
        printf("------------------------------------------------------------\n");
        {
            if (!diskModel1) {
                printf("  SKIPPED: _ANEModel not loaded\n");
                printf("  Fallback: prepareChainingWithModel on InMemoryModel\n\n");

                @try {
                    id ioObj1 = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
                        gAIO, @selector(objectWithIOSurface:), k1.ioIn);
                    id buf1 = ((id(*)(Class,SEL,id,id,long long))objc_msgSend)(
                        gBuf,
                        @selector(bufferWithIOSurfaceObject:symbolIndex:source:),
                        ioObj1, @0, (long long)0);

                    id outIO1 = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
                        gAIO, @selector(objectWithIOSurface:), k1.ioOut);
                    id outBuf1 = ((id(*)(Class,SEL,id,id,long long))objc_msgSend)(
                        gBuf,
                        @selector(bufferWithIOSurfaceObject:symbolIndex:source:),
                        outIO1, @0, (long long)1);

                    IOSurfaceRef statsSurf = make_surface(64);
                    id outSet = ((id(*)(Class,SEL,IOSurfaceRef,id))objc_msgSend)(
                        gOutSets,
                        @selector(objectWithstatsSurRef:outputBuffer:),
                        statsSurf, @[outBuf1]);

                    id chainReq = ((id(*)(Class,SEL,id,id,id,id,id,id,id,id,id))
                        objc_msgSend)(gChain,
                        @selector(chainingRequestWithInputs:outputSets:
                            lbInputSymbolId:lbOutputSymbolId:procedureIndex:
                            signalEvents:transactionHandle:fwEnqueueDelay:
                            memoryPoolId:),
                        @[buf1], @[outSet], @(-1), @(-1), @0,
                        @[], @0, @0, @0);

                    if (chainReq) {
                        BOOL valid = ((BOOL(*)(id,SEL))objc_msgSend)(
                            chainReq, @selector(validate));
                        printf("  validate: %s\n", valid ? "YES" : "NO");

                        NSError *prepErr = nil;
                        BOOL prepOk = ((BOOL(*)(id,SEL,id,id,id,unsigned int,
                            NSError**))objc_msgSend)(client,
                            @selector(prepareChainingWithModel:options:
                                chainingReq:qos:error:),
                            k1.model, @{}, chainReq, (unsigned int)21,
                            &prepErr);
                        printf("  prepareChainingWithModel (InMemory): %s\n",
                               prepOk ? "YES" : "NO");
                        if (!prepOk && prepErr)
                            printf("  Error: %s\n",
                                   [[prepErr description] UTF8String]);
                    }
                    CFRelease(statsSurf);
                } @catch (NSException *ex) {
                    printf("  EXCEPTION: %s\n", [[ex reason] UTF8String]);
                    printf("  (Expected: getUUID unrecognized selector)\n");
                }
            } else {
                printf("  Using _ANEModel for chaining!\n\n");

                NSArray *inputSymbols = nil;
                NSArray *outputSymbols = nil;
                @try {
                    SEL inSel = NSSelectorFromString(
                        @"inputSymbolIndicesForProcedureIndex:");
                    inputSymbols = ((id(*)(id,SEL,NSUInteger))objc_msgSend)(
                        diskModel1, inSel, (NSUInteger)0);
                    SEL outSel = NSSelectorFromString(
                        @"outputSymbolIndicesForProcedureIndex:");
                    outputSymbols = ((id(*)(id,SEL,NSUInteger))objc_msgSend)(
                        diskModel1, outSel, (NSUInteger)0);
                    printf("  Input symbols: %s\n",
                           inputSymbols ? [[inputSymbols description] UTF8String]
                                        : "nil");
                    printf("  Output symbols: %s\n",
                           outputSymbols ? [[outputSymbols description] UTF8String]
                                         : "nil");
                } @catch (NSException *ex) {
                    printf("  Symbol query EXCEPTION: %s\n",
                           [[ex reason] UTF8String]);
                }

                @try {
                    NSNumber *inSymIdx = (inputSymbols.count > 0)
                        ? inputSymbols[0] : @0;
                    NSNumber *outSymIdx = (outputSymbols.count > 0)
                        ? outputSymbols[0] : @0;

                    id ioObj1 = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
                        gAIO, @selector(objectWithIOSurface:), k1.ioIn);
                    id inBuf = ((id(*)(Class,SEL,id,id,long long))objc_msgSend)(
                        gBuf,
                        @selector(bufferWithIOSurfaceObject:symbolIndex:source:),
                        ioObj1, inSymIdx, (long long)0);

                    id outIO = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
                        gAIO, @selector(objectWithIOSurface:), k1.ioOut);
                    id outBuf = ((id(*)(Class,SEL,id,id,long long))objc_msgSend)(
                        gBuf,
                        @selector(bufferWithIOSurfaceObject:symbolIndex:source:),
                        outIO, outSymIdx, (long long)1);

                    IOSurfaceRef statsSurf = make_surface(64);
                    id outSet = ((id(*)(Class,SEL,IOSurfaceRef,id))objc_msgSend)(
                        gOutSets,
                        @selector(objectWithstatsSurRef:outputBuffer:),
                        statsSurf, @[outBuf]);

                    id chainReq = ((id(*)(Class,SEL,id,id,id,id,id,id,id,id,id))
                        objc_msgSend)(gChain,
                        @selector(chainingRequestWithInputs:outputSets:
                            lbInputSymbolId:lbOutputSymbolId:procedureIndex:
                            signalEvents:transactionHandle:fwEnqueueDelay:
                            memoryPoolId:),
                        @[inBuf], @[outSet], @(-1), @(-1), @0,
                        @[], @0, @0, @0);

                    if (chainReq) {
                        BOOL valid = ((BOOL(*)(id,SEL))objc_msgSend)(
                            chainReq, @selector(validate));
                        printf("  validate: %s\n", valid ? "YES" : "NO");

                        NSError *prepErr = nil;
                        BOOL prepOk = ((BOOL(*)(id,SEL,id,id,id,unsigned int,
                            NSError**))objc_msgSend)(client,
                            @selector(prepareChainingWithModel:options:
                                chainingReq:qos:error:),
                            diskModel1, @{}, chainReq, (unsigned int)21,
                            &prepErr);
                        printf("  prepareChainingWithModel: %s\n",
                               prepOk ? "YES" : "NO");
                        if (!prepOk && prepErr)
                            printf("  Error: %s\n",
                                   [[prepErr description] UTF8String]);

                        if (prepOk) {
                            printf("  CHAINING PREPARE SUCCEEDED!\n");

                            @try {
                                NSError *enqErr = nil;
                                BOOL enqOk = ((BOOL(*)(id,SEL,id,id,id,
                                    unsigned int,NSError**))objc_msgSend)(
                                    client,
                                    @selector(enqueueSetsWithModel:outputSet:
                                        options:qos:error:),
                                    diskModel1, outSet, @{},
                                    (unsigned int)21, &enqErr);
                                printf("  enqueueSets: %s\n",
                                       enqOk ? "YES" : "NO");
                                if (!enqOk && enqErr)
                                    printf("  Error: %s\n",
                                           [[enqErr description] UTF8String]);
                            } @catch (NSException *ex) {
                                printf("  enqueueSets EXCEPTION: %s\n",
                                       [[ex reason] UTF8String]);
                            }

                            @try {
                                NSError *rdyErr = nil;
                                BOOL rdyOk = ((BOOL(*)(id,SEL,id,id,id,
                                    unsigned int,NSError**))objc_msgSend)(
                                    client,
                                    @selector(buffersReadyWithModel:
                                        inputBuffers:options:qos:error:),
                                    diskModel1, @[inBuf], @{},
                                    (unsigned int)21, &rdyErr);
                                printf("  buffersReady: %s\n",
                                       rdyOk ? "YES" : "NO");
                                if (!rdyOk && rdyErr)
                                    printf("  Error: %s\n",
                                           [[rdyErr description] UTF8String]);
                            } @catch (NSException *ex) {
                                printf("  buffersReady EXCEPTION: %s\n",
                                       [[ex reason] UTF8String]);
                            }

                            // Benchmark sequential baseline
                            printf("\n  --- Benchmark ---\n");
                            id wI = ((id(*)(Class,SEL,IOSurfaceRef))
                                objc_msgSend)(gAIO,
                                @selector(objectWithIOSurface:), k1.ioIn);
                            id wO = ((id(*)(Class,SEL,IOSurfaceRef))
                                objc_msgSend)(gAIO,
                                @selector(objectWithIOSurface:), k1.ioOut);
                            id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))
                                objc_msgSend)(gAR,
                                @selector(requestWithInputs:inputIndices:
                                    outputs:outputIndices:weightsBuffer:
                                    perfStats:procedureIndex:),
                                @[wI], @[@0], @[wO], @[@0], nil, nil, @0);

                            int iters = 100;
                            NSError *seqErr = nil;
                            uint64_t t0 = mach_absolute_time();
                            for (int i = 0; i < iters; i++) {
                                ((BOOL(*)(id,SEL,unsigned int,id,id,
                                    NSError**))objc_msgSend)(k1.model,
                                    @selector(evaluateWithQoS:options:
                                        request:error:),
                                    21, @{}, req, &seqErr);
                            }
                            double seqMs = tb_ms(
                                mach_absolute_time() - t0) / iters;
                            printf("  Sequential: %.3f ms/kernel\n", seqMs);
                        }
                    }
                    CFRelease(statsSurf);
                } @catch (NSException *ex) {
                    printf("  Chaining EXCEPTION: %s\n",
                           [[ex reason] UTF8String]);
                }
            }
        }

        // =================================================================
        // EXPERIMENT G: IOSurfaceSharedEvent / hardware fences
        // =================================================================
        printf("\n------------------------------------------------------------\n");
        printf("  EXPERIMENT G: IOSurfaceSharedEvent / hardware fences\n");
        printf("------------------------------------------------------------\n");
        {
            Class gSigEvent = NSClassFromString(@"_ANESharedSignalEvent");
            Class gWaitEvent = NSClassFromString(@"_ANESharedWaitEvent");

            printf("\n  --- G.1: Event class API ---\n");
            if (gSigEvent) dump_class("_ANESharedSignalEvent");
            else printf("  _ANESharedSignalEvent: NOT FOUND\n");
            if (gWaitEvent) dump_class("_ANESharedWaitEvent");
            else printf("  _ANESharedWaitEvent: NOT FOUND\n");

            printf("\n  --- G.2: MTLSharedEvent via Metal ---\n");
            @try {
                void *metalH = dlopen(
                    "/System/Library/Frameworks/Metal.framework/Metal",
                    RTLD_NOW);
                if (metalH) {
                    id (*createDev)(void) = dlsym(metalH,
                        "MTLCreateSystemDefaultDevice");
                    if (createDev) {
                        id dev = createDev();
                        printf("  MTLDevice: %s\n",
                               dev ? [[dev description] UTF8String] : "nil");

                        if (dev) {
                            SEL newEvt = NSSelectorFromString(
                                @"newSharedEvent");
                            if ([dev respondsToSelector:newEvt]) {
                                id shEvt = ((id(*)(id,SEL))objc_msgSend)(
                                    dev, newEvt);
                                printf("  MTLSharedEvent: %s\n",
                                       shEvt ? [[shEvt description] UTF8String]
                                             : "nil");

                                if (shEvt && gSigEvent) {
                                    printf("\n  --- G.3: _ANESharedSignalEvent "
                                           "with MTLSharedEvent ---\n");
                                    // Factory: (Q16 I24 q28 @36) =
                                    //  (uint64_t, unsigned int, long long, id)
                                    @try {
                                        SEL sigSel = NSSelectorFromString(
                                            @"signalEventWithValue:symbolIndex:"
                                            "eventType:sharedEvent:");
                                        if ([gSigEvent respondsToSelector:sigSel]) {
                                            for (int et = 0; et <= 2; et++) {
                                                id se = ((id(*)(Class,SEL,
                                                    uint64_t,unsigned int,
                                                    long long,id))
                                                    objc_msgSend)(gSigEvent,
                                                    sigSel, (uint64_t)1,
                                                    (unsigned int)0,
                                                    (long long)et, shEvt);
                                                printf("  eventType=%d: %s\n",
                                                       et,
                                                       se ? [[se description]
                                                           UTF8String] : "nil");
                                                if (se) {
                                                    dump_all_properties(
                                                        se, gSigEvent);
                                                }
                                            }
                                        }
                                    } @catch (NSException *ex) {
                                        printf("  EXCEPTION: %s\n",
                                               [[ex reason] UTF8String]);
                                    }
                                }

                                if (shEvt && gWaitEvent) {
                                    printf("\n  --- G.4: _ANESharedWaitEvent "
                                           "with MTLSharedEvent ---\n");
                                    // Factory: waitEventWithValue:sharedEvent:
                                    //   (Q16 @24) = (uint64_t, id)
                                    // Factory: waitEventWithValue:sharedEvent:eventType:
                                    //   (Q16 @24 Q32) = (uint64_t, id, uint64_t)
                                    @try {
                                        SEL wSel = NSSelectorFromString(
                                            @"waitEventWithValue:sharedEvent:");
                                        if ([gWaitEvent respondsToSelector:wSel]) {
                                            id we = ((id(*)(Class,SEL,
                                                uint64_t,id))
                                                objc_msgSend)(gWaitEvent,
                                                wSel, (uint64_t)1, shEvt);
                                            printf("  waitEvent(2-param): %s\n",
                                                   we ? [[we description]
                                                       UTF8String] : "nil");
                                            if (we)
                                                dump_all_properties(
                                                    we, gWaitEvent);
                                        }
                                    } @catch (NSException *ex) {
                                        printf("  2-param EXCEPTION: %s\n",
                                               [[ex reason] UTF8String]);
                                    }
                                    @try {
                                        SEL wSel3 = NSSelectorFromString(
                                            @"waitEventWithValue:"
                                            "sharedEvent:eventType:");
                                        if ([gWaitEvent respondsToSelector:wSel3]) {
                                            id we = ((id(*)(Class,SEL,
                                                uint64_t,id,uint64_t))
                                                objc_msgSend)(gWaitEvent,
                                                wSel3, (uint64_t)1, shEvt,
                                                (uint64_t)0);
                                            printf("  waitEvent(3-param): %s\n",
                                                   we ? [[we description]
                                                       UTF8String] : "nil");
                                            if (we)
                                                dump_all_properties(
                                                    we, gWaitEvent);
                                        }
                                    } @catch (NSException *ex) {
                                        printf("  3-param EXCEPTION: %s\n",
                                               [[ex reason] UTF8String]);
                                    }
                                }
                            }
                        }
                    }
                }
            } @catch (NSException *ex) {
                printf("  Metal EXCEPTION: %s\n", [[ex reason] UTF8String]);
            }

            printf("\n  --- G.5: IOSurfaceSharedEventCreate ---\n");
            @try {
                void *iosH = dlopen(
                    "/System/Library/Frameworks/"
                    "IOSurface.framework/IOSurface", RTLD_NOW);
                if (iosH) {
                    typedef id (*CreateFunc)(void);
                    CreateFunc fn = dlsym(iosH, "IOSurfaceSharedEventCreate");
                    if (fn) {
                        id iosEvt = fn();
                        printf("  IOSurfaceSharedEventCreate: %s\n",
                               iosEvt ? [[iosEvt description] UTF8String]
                                      : "nil");

                        // Try using IOSurfaceSharedEvent with signal/wait
                        if (iosEvt && gSigEvent) {
                            printf("\n  G.5b: SignalEvent with IOSurfaceSharedEvent\n");
                            @try {
                                SEL sigSel = NSSelectorFromString(
                                    @"signalEventWithValue:symbolIndex:"
                                    "eventType:sharedEvent:");
                                id se = ((id(*)(Class,SEL,uint64_t,
                                    unsigned int,long long,id))
                                    objc_msgSend)(gSigEvent, sigSel,
                                    (uint64_t)1, (unsigned int)0,
                                    (long long)0, iosEvt);
                                printf("  signalEvent: %s\n",
                                       se ? [[se description] UTF8String]
                                          : "nil");
                                if (se) dump_all_properties(se, gSigEvent);
                            } @catch (NSException *ex) {
                                printf("  EXCEPTION: %s\n",
                                       [[ex reason] UTF8String]);
                            }
                        }
                        if (iosEvt && gWaitEvent) {
                            printf("\n  G.5c: WaitEvent with IOSurfaceSharedEvent\n");
                            @try {
                                SEL wSel = NSSelectorFromString(
                                    @"waitEventWithValue:sharedEvent:");
                                id we = ((id(*)(Class,SEL,uint64_t,id))
                                    objc_msgSend)(gWaitEvent, wSel,
                                    (uint64_t)1, iosEvt);
                                printf("  waitEvent: %s\n",
                                       we ? [[we description] UTF8String]
                                          : "nil");
                                if (we) dump_all_properties(we, gWaitEvent);
                            } @catch (NSException *ex) {
                                printf("  EXCEPTION: %s\n",
                                       [[ex reason] UTF8String]);
                            }
                        }
                    } else {
                        printf("  IOSurfaceSharedEventCreate: not found\n");
                    }
                }
            } @catch (NSException *ex) {
                printf("  EXCEPTION: %s\n", [[ex reason] UTF8String]);
            }
        }

        // =================================================================
        // EXPERIMENT H: Alternative chaining preparation
        // =================================================================
        printf("\n------------------------------------------------------------\n");
        printf("  EXPERIMENT H: Alternative chaining preparation\n");
        printf("------------------------------------------------------------\n");
        {
            printf("\n  --- H.1: _ANEClient chaining methods ---\n");
            Class clientCls = NSClassFromString(@"_ANEClient");
            if (clientCls) {
                unsigned int mc;
                Method *ms = class_copyMethodList(clientCls, &mc);
                for (unsigned int i = 0; i < mc; i++) {
                    const char *mn = sel_getName(method_getName(ms[i]));
                    if (strstr(mn, "chain") || strstr(mn, "Chain") ||
                        strstr(mn, "prepare") || strstr(mn, "Prepare") ||
                        strstr(mn, "enqueue") || strstr(mn, "Enqueue") ||
                        strstr(mn, "buffer") || strstr(mn, "Buffer") ||
                        strstr(mn, "ready") || strstr(mn, "Ready") ||
                        strstr(mn, "pipeline") || strstr(mn, "Pipeline") ||
                        strstr(mn, "batch") || strstr(mn, "Batch") ||
                        strstr(mn, "async") || strstr(mn, "Async")) {
                        printf("    - %s [%s]\n", mn,
                               method_getTypeEncoding(ms[i]));
                    }
                }
                free(ms);

                printf("\n  --- H.2: doPrepareChainingWithModel ---\n");
                SEL doPrep = NSSelectorFromString(
                    @"doPrepareChainingWithModel:options:"
                    "chainingReq:qos:error:");
                if ([client respondsToSelector:doPrep]) {
                    printf("  doPrepareChainingWithModel EXISTS\n");

                    @try {
                        id ioObj = ((id(*)(Class,SEL,IOSurfaceRef))
                            objc_msgSend)(gAIO,
                            @selector(objectWithIOSurface:), k1.ioIn);
                        id buf = ((id(*)(Class,SEL,id,id,long long))
                            objc_msgSend)(gBuf,
                            @selector(bufferWithIOSurfaceObject:
                                symbolIndex:source:),
                            ioObj, @0, (long long)0);
                        id outIO = ((id(*)(Class,SEL,IOSurfaceRef))
                            objc_msgSend)(gAIO,
                            @selector(objectWithIOSurface:), k1.ioOut);
                        id outBuf = ((id(*)(Class,SEL,id,id,long long))
                            objc_msgSend)(gBuf,
                            @selector(bufferWithIOSurfaceObject:
                                symbolIndex:source:),
                            outIO, @0, (long long)1);
                        IOSurfaceRef ss = make_surface(64);
                        id os = ((id(*)(Class,SEL,IOSurfaceRef,id))
                            objc_msgSend)(gOutSets,
                            @selector(objectWithstatsSurRef:outputBuffer:),
                            ss, @[outBuf]);
                        id cr = ((id(*)(Class,SEL,id,id,id,id,id,id,id,id,id))
                            objc_msgSend)(gChain,
                            @selector(chainingRequestWithInputs:outputSets:
                                lbInputSymbolId:lbOutputSymbolId:
                                procedureIndex:signalEvents:
                                transactionHandle:fwEnqueueDelay:
                                memoryPoolId:),
                            @[buf], @[os], @(-1), @(-1), @0,
                            @[], @0, @0, @0);

                        NSError *err = nil;
                        printf("  With _ANEInMemoryModel...\n");
                        BOOL ok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,
                            NSError**))objc_msgSend)(client, doPrep,
                            k1.model, @{}, cr, (unsigned int)21, &err);
                        printf("  Result: %s\n", ok ? "YES" : "NO");
                        if (!ok && err)
                            printf("  Error: %s\n",
                                   [[err description] UTF8String]);
                        CFRelease(ss);
                    } @catch (NSException *ex) {
                        printf("  InMemory EXCEPTION: %s\n",
                               [[ex reason] UTF8String]);
                    }

                    if (diskModel1) {
                        @try {
                            id ioObj = ((id(*)(Class,SEL,IOSurfaceRef))
                                objc_msgSend)(gAIO,
                                @selector(objectWithIOSurface:), k1.ioIn);
                            id buf = ((id(*)(Class,SEL,id,id,long long))
                                objc_msgSend)(gBuf,
                                @selector(bufferWithIOSurfaceObject:
                                    symbolIndex:source:),
                                ioObj, @0, (long long)0);
                            id outIO = ((id(*)(Class,SEL,IOSurfaceRef))
                                objc_msgSend)(gAIO,
                                @selector(objectWithIOSurface:), k1.ioOut);
                            id outBuf = ((id(*)(Class,SEL,id,id,long long))
                                objc_msgSend)(gBuf,
                                @selector(bufferWithIOSurfaceObject:
                                    symbolIndex:source:),
                                outIO, @0, (long long)1);
                            IOSurfaceRef ss = make_surface(64);
                            id os = ((id(*)(Class,SEL,IOSurfaceRef,id))
                                objc_msgSend)(gOutSets,
                                @selector(objectWithstatsSurRef:outputBuffer:),
                                ss, @[outBuf]);
                            id cr = ((id(*)(Class,SEL,id,id,id,id,id,id,id,id,id))
                                objc_msgSend)(gChain,
                                @selector(chainingRequestWithInputs:outputSets:
                                    lbInputSymbolId:lbOutputSymbolId:
                                    procedureIndex:signalEvents:
                                    transactionHandle:fwEnqueueDelay:
                                    memoryPoolId:),
                                @[buf], @[os], @(-1), @(-1), @0,
                                @[], @0, @0, @0);

                            NSError *err = nil;
                            printf("  With _ANEModel...\n");
                            BOOL ok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,
                                NSError**))objc_msgSend)(client, doPrep,
                                diskModel1, @{}, cr, (unsigned int)21, &err);
                            printf("  Result: %s\n", ok ? "YES" : "NO");
                            if (!ok && err)
                                printf("  Error: %s\n",
                                       [[err description] UTF8String]);
                            CFRelease(ss);
                        } @catch (NSException *ex) {
                            printf("  ANEModel EXCEPTION: %s\n",
                                   [[ex reason] UTF8String]);
                        }
                    }
                } else {
                    printf("  NOT available\n");
                }

                printf("\n  --- H.3: All _ANEClient methods ---\n");
                unsigned int allC;
                Method *allM = class_copyMethodList(clientCls, &allC);
                printf("  Total: %u\n", allC);
                for (unsigned int i = 0; i < allC; i++) {
                    printf("    - %s\n",
                           sel_getName(method_getName(allM[i])));
                }
                free(allM);
            }
        }

        // =================================================================
        // Experiment K: ChainingRequest Factory Type Encoding Analysis
        // =================================================================
        printf("\n==============================================================\n");
        printf("  Experiment K: Type Encoding Analysis\n");
        printf("==============================================================\n\n");
        {
            Class clientCls = object_getClass(client);

            SEL chainFactorySel = @selector(chainingRequestWithInputs:outputSets:
                lbInputSymbolId:lbOutputSymbolId:procedureIndex:
                signalEvents:transactionHandle:fwEnqueueDelay:memoryPoolId:);
            Method chainFactory = class_getClassMethod(gChain, chainFactorySel);
            if (chainFactory) {
                const char *enc = method_getTypeEncoding(chainFactory);
                printf("  ChainingRequest factory encoding: %s\n", enc ? enc : "nil");

                if (enc) {
                    const char *paramNames[] = {
                        "return", "self", "_cmd",
                        "inputs", "outputSets", "lbInputSymbolId",
                        "lbOutputSymbolId", "procedureIndex", "signalEvents",
                        "transactionHandle", "fwEnqueueDelay", "memoryPoolId"
                    };
                    unsigned int nargs = method_getNumberOfArguments(chainFactory);
                    printf("  Number of arguments: %u\n", nargs);
                    for (unsigned int i = 0; i < nargs && i < 12; i++) {
                        char argType[64] = {0};
                        method_getArgumentType(chainFactory, i, argType, sizeof(argType));
                        printf("    arg[%u] %-20s = %s", i, paramNames[i], argType);
                        if (argType[0] == '@') printf("  (id/object)");
                        else if (argType[0] == 'q') printf("  (int64_t)");
                        else if (argType[0] == 'Q') printf("  (uint64_t)");
                        else if (argType[0] == 'i') printf("  (int32_t)");
                        else if (argType[0] == 'I') printf("  (uint32_t)");
                        else if (argType[0] == 'B') printf("  (BOOL)");
                        else if (argType[0] == 'v') printf("  (void)");
                        else if (argType[0] == ':') printf("  (SEL)");
                        printf("\n");
                    }
                }
            } else {
                printf("  ChainingRequest factory: NOT FOUND\n");
            }

            SEL prepSel = @selector(prepareChainingWithModel:options:
                chainingReq:qos:error:);
            Method prepMethod = class_getInstanceMethod(clientCls, prepSel);
            if (!prepMethod)
                prepMethod = class_getInstanceMethod(
                    NSClassFromString(@"_ANEClient"), prepSel);
            if (prepMethod) {
                const char *enc = method_getTypeEncoding(prepMethod);
                printf("\n  prepareChainingWithModel encoding: %s\n",
                       enc ? enc : "nil");
                if (enc) {
                    const char *pNames[] = {
                        "return", "self", "_cmd",
                        "model", "options", "chainingReq", "qos", "error"
                    };
                    unsigned int nargs = method_getNumberOfArguments(prepMethod);
                    printf("  Number of arguments: %u\n", nargs);
                    for (unsigned int i = 0; i < nargs && i < 8; i++) {
                        char argType[64] = {0};
                        method_getArgumentType(prepMethod, i, argType, sizeof(argType));
                        printf("    arg[%u] %-15s = %s", i, pNames[i], argType);
                        if (argType[0] == '@') printf("  (id/object)");
                        else if (argType[0] == 'q') printf("  (int64_t)");
                        else if (argType[0] == 'Q') printf("  (uint64_t)");
                        else if (argType[0] == 'I') printf("  (uint32_t)");
                        else if (argType[0] == 'B') printf("  (BOOL)");
                        printf("\n");
                    }
                }
            } else {
                printf("\n  prepareChainingWithModel: NOT FOUND\n");
            }

            SEL doPrepSel = NSSelectorFromString(
                @"doPrepareChainingWithModel:options:chainingReq:qos:error:");
            Method doPrepMethod = class_getInstanceMethod(
                NSClassFromString(@"_ANEClient"), doPrepSel);
            if (doPrepMethod) {
                const char *enc = method_getTypeEncoding(doPrepMethod);
                printf("\n  doPrepareChainingWithModel encoding: %s\n",
                       enc ? enc : "nil");
                unsigned int nargs = method_getNumberOfArguments(doPrepMethod);
                printf("  Number of arguments: %u\n", nargs);
                for (unsigned int i = 0; i < nargs; i++) {
                    char argType[64] = {0};
                    method_getArgumentType(doPrepMethod, i, argType, sizeof(argType));
                    printf("    arg[%u] = %s\n", i, argType);
                }
            }

            printf("\n  --- K.2: All _ANEChainingRequest methods type encodings ---\n");
            {
                unsigned int mc;
                Method *cms = class_copyMethodList(object_getClass(gChain), &mc);
                printf("  Class methods (%u):\n", mc);
                for (unsigned int i = 0; i < mc; i++) {
                    const char *name = sel_getName(method_getName(cms[i]));
                    const char *enc = method_getTypeEncoding(cms[i]);
                    printf("    + %s\n      encoding: %s\n", name, enc ? enc : "?");
                }
                free(cms);

                Method *ims = class_copyMethodList(gChain, &mc);
                printf("  Instance methods (%u):\n", mc);
                for (unsigned int i = 0; i < mc; i++) {
                    const char *name = sel_getName(method_getName(ims[i]));
                    const char *enc = method_getTypeEncoding(ims[i]);
                    printf("    - %s\n      encoding: %s\n", name, enc ? enc : "?");
                }
                free(ims);
            }
        }

        // =================================================================
        // Experiment L: Array-Typed ChainingRequest Parameters
        // =================================================================
        printf("\n==============================================================\n");
        printf("  Experiment L: Array-Typed ChainingRequest Parameters\n");
        printf("==============================================================\n\n");
        BOOL chainingPrepSuccess = NO;
        id bestChainReq = nil;
        id bestModel = diskModel1 ? diskModel1 : k1.model;
        {
            id ioObj = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
                gAIO, @selector(objectWithIOSurface:), k1.ioIn);
            id inBuf = ((id(*)(Class,SEL,id,id,long long))objc_msgSend)(
                gBuf, @selector(bufferWithIOSurfaceObject:symbolIndex:source:),
                ioObj, @0, (long long)0);

            id outIO = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
                gAIO, @selector(objectWithIOSurface:), k1.ioOut);
            id outBuf = ((id(*)(Class,SEL,id,id,long long))objc_msgSend)(
                gBuf, @selector(bufferWithIOSurfaceObject:symbolIndex:source:),
                outIO, @0, (long long)1);

            IOSurfaceRef statsSurf = make_surface(64);
            id outSet = ((id(*)(Class,SEL,IOSurfaceRef,id))objc_msgSend)(
                gOutSets, @selector(objectWithstatsSurRef:outputBuffer:),
                statsSurf, @[outBuf]);

            struct {
                const char *label;
                id lbIn; id lbOut; id procIdx;
            } combos[] = {
                { "arrays @[@(-1)]",   @[@(-1)], @[@(-1)], @[@0] },
                { "arrays @[@0]",      @[@0],    @[@0],    @[@0] },
                { "empty arrays @[]",  @[],      @[],      @[]   },
                { "nil values",        nil,      nil,      nil   },
                { "original NSNumber", @(-1),    @(-1),    @0    },
            };
            int ncombos = sizeof(combos)/sizeof(combos[0]);

            for (int ci = 0; ci < ncombos; ci++) {
                printf("  --- L.%d: %s ---\n", ci+1, combos[ci].label);
                @try {
                    id cr = ((id(*)(Class,SEL,id,id,id,id,id,id,id,id,id))
                        objc_msgSend)(gChain,
                        @selector(chainingRequestWithInputs:outputSets:
                            lbInputSymbolId:lbOutputSymbolId:procedureIndex:
                            signalEvents:transactionHandle:fwEnqueueDelay:
                            memoryPoolId:),
                        @[inBuf], @[outSet],
                        combos[ci].lbIn, combos[ci].lbOut, combos[ci].procIdx,
                        @[], @0, @0, @0);

                    if (!cr) {
                        printf("  ChainingRequest: nil\n\n");
                        continue;
                    }

                    BOOL valid = ((BOOL(*)(id,SEL))objc_msgSend)(
                        cr, @selector(validate));
                    printf("  validate: %s\n", valid ? "YES" : "NO");
                    printf("  desc: %s\n",
                           [[cr description] UTF8String]);

                    @try {
                        NSError *prepErr = nil;
                        BOOL prepOk = ((BOOL(*)(id,SEL,id,id,id,unsigned int,
                            NSError**))objc_msgSend)(client,
                            @selector(prepareChainingWithModel:options:
                                chainingReq:qos:error:),
                            bestModel, @{}, cr, (unsigned int)21, &prepErr);
                        printf("  prepareChainingWithModel: %s\n",
                               prepOk ? "YES" : "NO");
                        if (prepErr)
                            printf("  Error: %s\n",
                                   [[prepErr description] UTF8String]);
                        if (prepOk || !prepErr) {
                            chainingPrepSuccess = prepOk;
                            bestChainReq = cr;
                            printf("  *** GOT PAST THE CRASH! ***\n");
                        }
                    } @catch (NSException *prepEx) {
                        printf("  prepare EXCEPTION: %s\n",
                               [[prepEx reason] UTF8String]);
                    }
                } @catch (NSException *ex) {
                    printf("  factory EXCEPTION: %s\n",
                           [[ex reason] UTF8String]);
                }
                printf("\n");
            }

            CFRelease(statsSurf);
        }

        // =================================================================
        // Experiment M: Load Model via _ANEClient
        // =================================================================
        printf("\n==============================================================\n");
        printf("  Experiment M: Load Model via _ANEClient\n");
        printf("==============================================================\n\n");
        id fullyLoadedModel = nil;
        {
            if (!diskModel1) {
                printf("  SKIPPED: no _ANEModel from Experiment E\n");
            } else {
                @try {
                    id st = [diskModel1 valueForKey:@"state"];
                    printf("  diskModel1 state before: %s\n",
                           st ? [[st description] UTF8String] : "nil");
                } @catch (NSException *ex) {
                    printf("  state query exception: %s\n",
                           [[ex reason] UTF8String]);
                }

                printf("\n  --- M.1: compiledModelExistsFor: ---\n");
                @try {
                    SEL existsSel = NSSelectorFromString(
                        @"compiledModelExistsFor:");
                    if ([client respondsToSelector:existsSel]) {
                        BOOL exists = ((BOOL(*)(id,SEL,id))objc_msgSend)(
                            client, existsSel, diskModel1);
                        printf("  compiledModelExistsFor: %s\n",
                               exists ? "YES" : "NO");
                    } else {
                        printf("  compiledModelExistsFor: NOT AVAILABLE\n");
                    }
                } @catch (NSException *ex) {
                    printf("  EXCEPTION: %s\n", [[ex reason] UTF8String]);
                }

                printf("\n  --- M.2: loadModel:options:qos:error: ---\n");
                @try {
                    SEL loadSel = NSSelectorFromString(
                        @"loadModel:options:qos:error:");
                    if ([client respondsToSelector:loadSel]) {
                        NSError *loadErr = nil;
                        BOOL loadOk = ((BOOL(*)(id,SEL,id,id,unsigned int,
                            NSError**))objc_msgSend)(client, loadSel,
                            diskModel1, @{}, (unsigned int)21, &loadErr);
                        printf("  loadModel: %s\n", loadOk ? "YES" : "NO");
                        if (loadErr)
                            printf("  Error: %s\n",
                                   [[loadErr description] UTF8String]);

                        if (loadOk) {
                            fullyLoadedModel = diskModel1;
                            @try {
                                SEL inSel = NSSelectorFromString(
                                    @"inputSymbolIndicesForProcedureIndex:");
                                id inSyms = ((id(*)(id,SEL,NSUInteger))
                                    objc_msgSend)(diskModel1, inSel, 0);
                                SEL outSel = NSSelectorFromString(
                                    @"outputSymbolIndicesForProcedureIndex:");
                                id outSyms = ((id(*)(id,SEL,NSUInteger))
                                    objc_msgSend)(diskModel1, outSel, 0);
                                printf("  After load - inputSymbols: %s\n",
                                    inSyms ? [[inSyms description] UTF8String]
                                           : "nil/empty");
                                printf("  After load - outputSymbols: %s\n",
                                    outSyms ? [[outSyms description] UTF8String]
                                            : "nil/empty");
                            } @catch (NSException *ex) {
                                printf("  Symbol query EXCEPTION: %s\n",
                                       [[ex reason] UTF8String]);
                            }
                        }
                    } else {
                        printf("  loadModel: NOT AVAILABLE\n");
                    }
                } @catch (NSException *ex) {
                    printf("  EXCEPTION: %s\n", [[ex reason] UTF8String]);
                }

                printf("\n  --- M.3: compileModel:options:qos:error: ---\n");
                @try {
                    SEL compileSel = NSSelectorFromString(
                        @"compileModel:options:qos:error:");
                    if ([client respondsToSelector:compileSel]) {
                        NSError *compErr = nil;
                        BOOL compOk = ((BOOL(*)(id,SEL,id,id,unsigned int,
                            NSError**))objc_msgSend)(client, compileSel,
                            diskModel1, @{}, (unsigned int)21, &compErr);
                        printf("  compileModel: %s\n", compOk ? "YES" : "NO");
                        if (compErr)
                            printf("  Error: %s\n",
                                   [[compErr description] UTF8String]);

                        if (compOk) {
                            fullyLoadedModel = diskModel1;
                            @try {
                                id inSyms = ((id(*)(id,SEL,NSUInteger))
                                    objc_msgSend)(diskModel1,
                                    NSSelectorFromString(
                                        @"inputSymbolIndicesForProcedureIndex:"),
                                    0);
                                id outSyms = ((id(*)(id,SEL,NSUInteger))
                                    objc_msgSend)(diskModel1,
                                    NSSelectorFromString(
                                        @"outputSymbolIndicesForProcedureIndex:"),
                                    0);
                                printf("  After compile - inputSymbols: %s\n",
                                    inSyms ? [[inSyms description] UTF8String]
                                           : "nil/empty");
                                printf("  After compile - outputSymbols: %s\n",
                                    outSyms ? [[outSyms description] UTF8String]
                                            : "nil/empty");
                            } @catch (NSException *ex) {
                                printf("  Symbol query EXCEPTION: %s\n",
                                       [[ex reason] UTF8String]);
                            }
                        }
                    } else {
                        printf("  compileModel: NOT AVAILABLE\n");
                    }
                } @catch (NSException *ex) {
                    printf("  EXCEPTION: %s\n", [[ex reason] UTF8String]);
                }

                @try {
                    id st = [diskModel1 valueForKey:@"state"];
                    printf("\n  diskModel1 state after: %s\n",
                           st ? [[st description] UTF8String] : "nil");
                } @catch (NSException *ex) {
                    printf("\n  state query exception: %s\n",
                           [[ex reason] UTF8String]);
                }
            }
        }

        // =================================================================
        // Experiment N: IOSurface Mapping via _ANEProgramIOSurfacesMapper
        // =================================================================
        printf("\n==============================================================\n");
        printf("  Experiment N: IOSurface Mapping\n");
        printf("==============================================================\n\n");
        {
            Class gMapper = NSClassFromString(@"_ANEProgramIOSurfacesMapper");
            if (!gMapper) {
                printf("  _ANEProgramIOSurfacesMapper: NOT FOUND\n");
            } else {
                printf("  _ANEProgramIOSurfacesMapper: FOUND\n");
                dump_class("_ANEProgramIOSurfacesMapper");

                id progHandle = nil;
                @try {
                    progHandle = [k1.model valueForKey:@"programHandle"];
                } @catch (NSException *ex) {
                    printf("  programHandle exception: %s\n",
                           [[ex reason] UTF8String]);
                }

                printf("\n  --- N.1: mapperWithProgramHandle: ---\n");
                id mapper = nil;
                if (progHandle) {
                    uint64_t ph = [progHandle unsignedLongLongValue];
                    printf("  programHandle = %llu\n", ph);
                    @try {
                        SEL mapperSel = NSSelectorFromString(
                            @"mapperWithProgramHandle:");
                        mapper = ((id(*)(Class,SEL,uint64_t))objc_msgSend)(
                            gMapper, mapperSel, ph);
                        printf("  mapper created: %s\n",
                               mapper ? [[mapper description] UTF8String]
                                      : "nil");
                    } @catch (NSException *ex) {
                        printf("  EXCEPTION: %s\n",
                               [[ex reason] UTF8String]);
                    }
                }

                id targetModel = diskModel1 ? diskModel1 : k1.model;
                const char *modelType = diskModel1 ? "_ANEModel" : "InMemoryModel";

                if (mapper) {
                    printf("\n  --- N.2: mapIOSurfacesWithModel: (%s) ---\n",
                           modelType);
                    @try {
                        id ioObj = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
                            gAIO, @selector(objectWithIOSurface:), k1.ioIn);
                        id outIO = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
                            gAIO, @selector(objectWithIOSurface:), k1.ioOut);
                        id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))
                            objc_msgSend)(gAR,
                            @selector(requestWithInputs:inputIndices:
                                outputs:outputIndices:weightsBuffer:
                                perfStats:procedureIndex:),
                            @[ioObj], @[@0], @[outIO], @[@0], nil, nil, @0);

                        if (req) {
                            SEL mapSel = NSSelectorFromString(
                                @"mapIOSurfacesWithModel:request:"
                                "cacheInference:error:");
                            NSError *mapErr = nil;
                            BOOL mapOk = ((BOOL(*)(id,SEL,id,id,BOOL,
                                NSError**))objc_msgSend)(mapper, mapSel,
                                targetModel, req, NO, &mapErr);
                            printf("  mapIOSurfaces: %s\n",
                                   mapOk ? "YES" : "NO");
                            if (mapErr)
                                printf("  Error: %s\n",
                                       [[mapErr description] UTF8String]);

                            if (mapOk) {
                                dump_all_properties(mapper,
                                    [mapper class]);
                            }
                        } else {
                            printf("  Request creation failed\n");
                        }
                    } @catch (NSException *ex) {
                        printf("  EXCEPTION: %s\n",
                               [[ex reason] UTF8String]);
                    }

                    printf("\n  --- N.3: validateRequest:model: ---\n");
                    @try {
                        id ioObj = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
                            gAIO, @selector(objectWithIOSurface:), k1.ioIn);
                        id outIO = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
                            gAIO, @selector(objectWithIOSurface:), k1.ioOut);
                        id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))
                            objc_msgSend)(gAR,
                            @selector(requestWithInputs:inputIndices:
                                outputs:outputIndices:weightsBuffer:
                                perfStats:procedureIndex:),
                            @[ioObj], @[@0], @[outIO], @[@0], nil, nil, @0);

                        SEL valSel = NSSelectorFromString(
                            @"validateRequest:model:");
                        BOOL valOk = ((BOOL(*)(id,SEL,id,id))objc_msgSend)(
                            mapper, valSel, req, targetModel);
                        printf("  validateRequest: %s\n",
                               valOk ? "YES" : "NO");
                    } @catch (NSException *ex) {
                        printf("  EXCEPTION: %s\n",
                               [[ex reason] UTF8String]);
                    }
                }

                if (diskModel1) {
                    printf("\n  --- N.4: _ANEModel.mapper property ---\n");
                    @try {
                        id modelMapper = [diskModel1 valueForKey:@"mapper"];
                        printf("  model.mapper: %s\n",
                               modelMapper
                                   ? [[modelMapper description] UTF8String]
                                   : "nil");
                        if (modelMapper) {
                            dump_all_properties(modelMapper,
                                [modelMapper class]);
                        }
                    } @catch (NSException *ex) {
                        printf("  EXCEPTION: %s\n",
                               [[ex reason] UTF8String]);
                    }
                }
            }
        }

        // =================================================================
        // Experiment O: Procedure Info Extraction
        // =================================================================
        printf("\n==============================================================\n");
        printf("  Experiment O: Procedure Info Extraction\n");
        printf("==============================================================\n\n");
        {
            id targetModel = diskModel1 ? diskModel1 : k1.model;
            const char *modelType = diskModel1 ? "_ANEModel" : "InMemoryModel";
            printf("  Using: %s\n", modelType);

            printf("\n  --- O.1: procedureInfoForProcedureIndex:0 ---\n");
            @try {
                SEL piSel = NSSelectorFromString(
                    @"procedureInfoForProcedureIndex:");
                if ([targetModel respondsToSelector:piSel]) {
                    id pInfo = ((id(*)(id,SEL,NSUInteger))objc_msgSend)(
                        targetModel, piSel, (NSUInteger)0);
                    printf("  procedureInfo: %s\n",
                           pInfo ? [[pInfo description] UTF8String] : "nil");
                    if (pInfo) {
                        printf("  class: %s\n",
                               [NSStringFromClass([pInfo class]) UTF8String]);
                        dump_all_properties(pInfo, [pInfo class]);
                    }
                } else {
                    printf("  procedureInfoForProcedureIndex: NOT AVAILABLE\n");
                }
            } @catch (NSException *ex) {
                printf("  EXCEPTION: %s\n", [[ex reason] UTF8String]);
            }

            printf("\n  --- O.2: procedureCount ---\n");
            @try {
                SEL pcSel = NSSelectorFromString(@"procedureCount");
                if ([targetModel respondsToSelector:pcSel]) {
                    NSUInteger pc = ((NSUInteger(*)(id,SEL))objc_msgSend)(
                        targetModel, pcSel);
                    printf("  procedureCount: %lu\n", (unsigned long)pc);
                } else {
                    printf("  procedureCount: NOT AVAILABLE\n");
                    id pcVal = nil;
                    @try {
                        pcVal = [targetModel valueForKey:@"procedureCount"];
                        printf("  procedureCount (KVC): %s\n",
                               pcVal ? [[pcVal description] UTF8String] : "nil");
                    } @catch (NSException *ex2) {
                        printf("  procedureCount KVC: %s\n",
                               [[ex2 reason] UTF8String]);
                    }
                }
            } @catch (NSException *ex) {
                printf("  EXCEPTION: %s\n", [[ex reason] UTF8String]);
            }

            printf("\n  --- O.3: modelAttributes ---\n");
            @try {
                id attrs = [targetModel valueForKey:@"modelAttributes"];
                printf("  modelAttributes: %s\n",
                       attrs ? [[attrs description] UTF8String] : "nil");
            } @catch (NSException *ex) {
                printf("  EXCEPTION: %s\n", [[ex reason] UTF8String]);
            }

            printf("\n  --- O.4: inputSymbolNames / outputSymbolNames ---\n");
            @try {
                SEL inNamesSel = NSSelectorFromString(@"inputSymbolNames");
                if ([targetModel respondsToSelector:inNamesSel]) {
                    id names = ((id(*)(id,SEL))objc_msgSend)(
                        targetModel, inNamesSel);
                    printf("  inputSymbolNames: %s\n",
                           names ? [[names description] UTF8String] : "nil");
                } else {
                    printf("  inputSymbolNames: NOT AVAILABLE as method\n");
                    @try {
                        id n = [targetModel valueForKey:@"inputSymbolNames"];
                        printf("  inputSymbolNames (KVC): %s\n",
                               n ? [[n description] UTF8String] : "nil");
                    } @catch (NSException *ex2) {
                        printf("  inputSymbolNames KVC: not available\n");
                    }
                }
            } @catch (NSException *ex) {
                printf("  EXCEPTION: %s\n", [[ex reason] UTF8String]);
            }
            @try {
                SEL outNamesSel = NSSelectorFromString(@"outputSymbolNames");
                if ([targetModel respondsToSelector:outNamesSel]) {
                    id names = ((id(*)(id,SEL))objc_msgSend)(
                        targetModel, outNamesSel);
                    printf("  outputSymbolNames: %s\n",
                           names ? [[names description] UTF8String] : "nil");
                } else {
                    printf("  outputSymbolNames: NOT AVAILABLE as method\n");
                    @try {
                        id n = [targetModel valueForKey:@"outputSymbolNames"];
                        printf("  outputSymbolNames (KVC): %s\n",
                               n ? [[n description] UTF8String] : "nil");
                    } @catch (NSException *ex2) {
                        printf("  outputSymbolNames KVC: not available\n");
                    }
                }
            } @catch (NSException *ex) {
                printf("  EXCEPTION: %s\n", [[ex reason] UTF8String]);
            }

            if (diskModel1) {
                printf("\n  --- O.5: Full _ANEModel property dump ---\n");
                dump_all_properties(diskModel1,
                    NSClassFromString(@"_ANEModel"));
            }
        }

        // =================================================================
        // Experiment P: Full Chaining Retry with Fixes
        // =================================================================
        printf("\n==============================================================\n");
        printf("  Experiment P: Full Chaining Retry\n");
        printf("==============================================================\n\n");
        BOOL chainExecuted = NO;
        {
            void *metalHandle = dlopen(
                "/System/Library/Frameworks/Metal.framework/Metal", RTLD_NOW);
            (void)metalHandle;
            id mtlDev = nil;
            {
                id (*createDev)(void) = (id(*)(void))
                    dlsym(RTLD_DEFAULT, "MTLCreateSystemDefaultDevice");
                if (createDev) mtlDev = createDev();
            }

            id signalEvents = @[];
            id shEvt = nil;
            if (mtlDev) {
                printf("  Metal device: %s\n",
                       [[mtlDev description] UTF8String]);
                @try {
                    shEvt = ((id(*)(id,SEL))objc_msgSend)(
                        mtlDev, NSSelectorFromString(@"newSharedEvent"));
                    if (shEvt) {
                        Class gSigEvent = NSClassFromString(
                            @"_ANESharedSignalEvent");
                        if (gSigEvent) {
                            long long et = 0;
                            @try {
                                id etObj = [gSigEvent valueForKey:
                                    @"ANESignalEventTypeMTLSharedEvent"];
                                if (etObj) et = [etObj longLongValue];
                            } @catch (NSException *ex) { (void)ex; }

                            SEL sigSel = NSSelectorFromString(
                                @"signalEventWithValue:symbolIndex:"
                                "eventType:sharedEvent:");
                            id se = ((id(*)(Class,SEL,uint64_t,unsigned int,
                                long long,id))objc_msgSend)(
                                gSigEvent, sigSel, (uint64_t)1,
                                (unsigned int)0, et, shEvt);
                            if (se)
                                signalEvents = @[se];
                            printf("  SignalEvent: %s\n",
                                   se ? "created" : "nil");
                        }
                    }
                } @catch (NSException *ex) {
                    printf("  SharedEvent EXCEPTION: %s\n",
                           [[ex reason] UTF8String]);
                }
            }

            struct {
                const char *label;
                id model;
            } modelCandidates[3];
            int nCandidates = 0;

            CompiledKernel k3 = compile_kernel(64, 32);
            if (k3.model) {
                NSURL *url = [NSURL fileURLWithPath:k3.tmpDir isDirectory:YES];
                Class gANEModel = NSClassFromString(@"_ANEModel");
                id freshDisk = ((id(*)(Class,SEL,id,id))objc_msgSend)(
                    gANEModel, @selector(modelAtURL:key:), url, k3.hexId);
                if (freshDisk) {
                    id ph = [k3.model valueForKey:@"programHandle"];
                    id prog = [k3.model valueForKey:@"program"];
                    if (ph) ((void(*)(id,SEL,uint64_t))objc_msgSend)(
                        freshDisk, @selector(setProgramHandle:),
                        [ph unsignedLongLongValue]);
                    if (prog) ((void(*)(id,SEL,id))objc_msgSend)(
                        freshDisk, @selector(setProgram:), prog);

                    modelCandidates[nCandidates++] = (typeof(modelCandidates[0]))
                        {"fresh _ANEModel (state=1)", freshDisk};
                }
                modelCandidates[nCandidates++] = (typeof(modelCandidates[0]))
                    {"InMemoryModel (k3)", k3.model};
            }
            if (diskModel1) {
                modelCandidates[nCandidates++] = (typeof(modelCandidates[0]))
                    {"populated _ANEModel (from E)", diskModel1};
            }

            for (int mi = 0; mi < nCandidates; mi++) {
                printf("\n  --- P.%d: %s ---\n", mi+1, modelCandidates[mi].label);
                id chainModel = modelCandidates[mi].model;
                printf("  class: %s\n",
                       [NSStringFromClass([chainModel class]) UTF8String]);
                @try {
                    id st = [chainModel valueForKey:@"state"];
                    printf("  state: %s\n",
                           st ? [[st description] UTF8String] : "N/A");
                } @catch (NSException *ex) { (void)ex; }

                id ioObj1 = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
                    gAIO, @selector(objectWithIOSurface:), k1.ioIn);
                id inBuf = ((id(*)(Class,SEL,id,id,long long))objc_msgSend)(
                    gBuf,
                    @selector(bufferWithIOSurfaceObject:symbolIndex:source:),
                    ioObj1, @0, (long long)0);

                id outIO = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
                    gAIO, @selector(objectWithIOSurface:), k1.ioOut);
                id outBuf = ((id(*)(Class,SEL,id,id,long long))objc_msgSend)(
                    gBuf,
                    @selector(bufferWithIOSurfaceObject:symbolIndex:source:),
                    outIO, @0, (long long)1);

                IOSurfaceRef sSurf = make_surface(64);
                id outSet = ((id(*)(Class,SEL,IOSurfaceRef,id))objc_msgSend)(
                    gOutSets, @selector(objectWithstatsSurRef:outputBuffer:),
                    sSurf, @[outBuf]);

                @try {
                    id cr = ((id(*)(Class,SEL,id,id,id,id,id,id,id,id,id))
                        objc_msgSend)(gChain,
                        @selector(chainingRequestWithInputs:outputSets:
                            lbInputSymbolId:lbOutputSymbolId:procedureIndex:
                            signalEvents:transactionHandle:fwEnqueueDelay:
                            memoryPoolId:),
                        @[inBuf], @[outSet], nil, nil, nil,
                        signalEvents, @0, @0, @0);

                    if (!cr) {
                        printf("  ChainingRequest: nil\n");
                        CFRelease(sSurf);
                        continue;
                    }

                    BOOL valid = ((BOOL(*)(id,SEL))objc_msgSend)(
                        cr, @selector(validate));
                    printf("  validate: %s\n", valid ? "YES" : "NO");

                    NSError *prepErr = nil;
                    BOOL prepOk = ((BOOL(*)(id,SEL,id,id,id,unsigned int,
                        NSError**))objc_msgSend)(client,
                        @selector(prepareChainingWithModel:options:
                            chainingReq:qos:error:),
                        chainModel, @{}, cr, (unsigned int)21, &prepErr);
                    printf("  prepare: %s\n", prepOk ? "YES" : "NO");
                    if (prepErr)
                        printf("  prepareError: %s\n",
                               [[prepErr description] UTF8String]);

                    if (prepOk) {
                        printf("  *** PREPARE SUCCEEDED! ***\n");
                        chainingPrepSuccess = YES;
                        chainExecuted = YES;

                        printf("\n  --- enqueueSetsWithModel ---\n");
                        @try {
                            SEL eqSel = NSSelectorFromString(
                                @"enqueueSetsWithModel:outputSet:"
                                "options:qos:error:");
                            NSError *eqErr = nil;
                            BOOL eqOk = ((BOOL(*)(id,SEL,id,id,id,
                                unsigned int,NSError**))objc_msgSend)(
                                client, eqSel, chainModel, outSet, @{},
                                (unsigned int)21, &eqErr);
                            printf("  enqueueSets: %s\n",
                                   eqOk ? "YES" : "NO");
                            if (eqErr)
                                printf("  Error: %s\n",
                                       [[eqErr description] UTF8String]);
                        } @catch (NSException *ex) {
                            printf("  EXCEPTION: %s\n",
                                   [[ex reason] UTF8String]);
                        }

                        printf("\n  --- buffersReadyWithModel ---\n");
                        @try {
                            SEL brSel = NSSelectorFromString(
                                @"buffersReadyWithModel:inputBuffers:"
                                "options:qos:error:");
                            NSError *brErr = nil;
                            BOOL brOk = ((BOOL(*)(id,SEL,id,id,id,
                                unsigned int,NSError**))objc_msgSend)(
                                client, brSel, chainModel, @[inBuf], @{},
                                (unsigned int)21, &brErr);
                            printf("  buffersReady: %s\n",
                                   brOk ? "YES" : "NO");
                            if (brErr)
                                printf("  Error: %s\n",
                                       [[brErr description] UTF8String]);
                        } @catch (NSException *ex) {
                            printf("  EXCEPTION: %s\n",
                                   [[ex reason] UTF8String]);
                        }

                        printf("\n  --- Benchmark ---\n");
                        uint64_t t0 = mach_absolute_time();
                        int niters = 50;
                        for (int i = 0; i < niters; i++) {
                            @try {
                                SEL brSel = NSSelectorFromString(
                                    @"buffersReadyWithModel:inputBuffers:"
                                    "options:qos:error:");
                                ((BOOL(*)(id,SEL,id,id,id,unsigned int,
                                    NSError**))objc_msgSend)(
                                    client, brSel, chainModel, @[inBuf],
                                    @{}, (unsigned int)21, nil);
                            } @catch (NSException *ex) {
                                if (i == 0)
                                    printf("  Bench EXCEPTION: %s\n",
                                           [[ex reason] UTF8String]);
                                break;
                            }
                        }
                        double elapsed = tb_ms(mach_absolute_time() - t0);
                        printf("  %d iters in %.3f ms (%.4f ms/iter)\n",
                               niters, elapsed, elapsed / niters);
                    }
                } @catch (NSException *ex) {
                    printf("  EXCEPTION: %s\n", [[ex reason] UTF8String]);
                }
                CFRelease(sSurf);
            }

            printf("\n  --- P.extra: Try with _ANEInputBuffersReady ---\n");
            @try {
                Class gIBR = NSClassFromString(@"_ANEInputBuffersReady");
                if (gIBR) {
                    dump_class("_ANEInputBuffersReady");
                    SEL ibrSel = NSSelectorFromString(
                        @"inputBuffersWithProcedureIndex:inputBufferInfoIndex:"
                        "inputFreeValue:executionDelay:");
                    if (class_getClassMethod(gIBR, ibrSel)) {
                        Method m = class_getClassMethod(gIBR, ibrSel);
                        const char *enc = method_getTypeEncoding(m);
                        printf("  inputBuffersReady encoding: %s\n",
                               enc ? enc : "?");
                        unsigned int na = method_getNumberOfArguments(m);
                        printf("  args: %u\n", na);
                        for (unsigned int i = 0; i < na; i++) {
                            char at[64] = {0};
                            method_getArgumentType(m, i, at, sizeof(at));
                            printf("    [%u] = %s\n", i, at);
                        }
                    }
                }

                Class gOSE = NSClassFromString(@"_ANEOutputSetEnqueue");
                if (gOSE) {
                    dump_class("_ANEOutputSetEnqueue");
                    SEL oseSel = NSSelectorFromString(
                        @"outputSetWithProcedureIndex:setIndex:signalValue:"
                        "signalNotRequired:isOpenLoop:");
                    if (class_getClassMethod(gOSE, oseSel)) {
                        Method m = class_getClassMethod(gOSE, oseSel);
                        const char *enc = method_getTypeEncoding(m);
                        printf("  outputSetEnqueue encoding: %s\n",
                               enc ? enc : "?");
                        unsigned int na = method_getNumberOfArguments(m);
                        printf("  args: %u\n", na);
                        for (unsigned int i = 0; i < na; i++) {
                            char at[64] = {0};
                            method_getArgumentType(m, i, at, sizeof(at));
                            printf("    [%u] = %s\n", i, at);
                        }
                    }
                }
            } @catch (NSException *ex) {
                printf("  EXCEPTION: %s\n", [[ex reason] UTF8String]);
            }

            if (k3.model) free_kernel(&k3);
        }

        // =================================================================
        // Summary
        // =================================================================
        printf("\n============================================================\n");
        printf("  RESULTS SUMMARY\n");
        printf("============================================================\n");
        printf("  Exp E:  _ANEModel loaded:     %s\n",
               diskModel1 ? "YES" : "NO");
        printf("  Exp E2: ANECompiler found:     (see above)\n");
        printf("  Exp F:  Chaining pipeline:     %s\n",
               diskModel1 ? "ATTEMPTED" : "SKIPPED");
        printf("  Exp G:  SharedEvents:          (see above)\n");
        printf("  Exp H:  Alt preparation:       (see above)\n");
        printf("  Exp K:  Type encodings:        DONE\n");
        printf("  Exp L:  Array params:          %s\n",
               chainingPrepSuccess ? "PREPARE SUCCEEDED" : "see above");
        printf("  Exp M:  Client load model:     %s\n",
               fullyLoadedModel ? "LOADED" : "see above");
        printf("  Exp N:  IOSurface mapping:     DONE\n");
        printf("  Exp O:  Procedure info:        DONE\n");
        printf("  Exp P:  Full chaining retry:   %s\n",
               chainExecuted ? "EXECUTED" : "see above");
        printf("============================================================\n");

        free_kernel(&k1);
        free_kernel(&k2);
        printf("\nDone.\n");
    }
    return 0;
}
