// test_coreml_chaining.m — Experiments Q-S: CoreML-compiled model for ANE chaining
// Build: make test_coreml_chaining && ./test_coreml_chaining
#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <IOSurface/IOSurface.h>
#import <mach/mach_time.h>

static mach_timebase_info_data_t g_tb;
static double tb_ms(uint64_t t) { return (double)t * g_tb.numer / g_tb.denom / 1e6; }

#pragma mark - Helpers

static void dump_class_brief(const char *name) {
    Class cls = NSClassFromString([NSString stringWithUTF8String:name]);
    if (!cls) { printf("  %s: NOT FOUND\n", name); return; }
    unsigned int mc, ic, pc;
    Method *cm = class_copyMethodList(object_getClass(cls), &mc);
    Method *im = class_copyMethodList(cls, &ic);
    objc_property_t *pp = class_copyPropertyList(cls, &pc);
    printf("  %s: %u class, %u instance methods, %u props\n", name, mc, ic, pc);
    free(cm); free(im); free(pp);
}

static void dump_props(id obj) {
    if (!obj) return;
    Class cls = [obj class];
    unsigned int pc;
    objc_property_t *pp = class_copyPropertyList(cls, &pc);
    for (unsigned int i = 0; i < pc; i++) {
        const char *pn = property_getName(pp[i]);
        @try {
            id v = [obj valueForKey:[NSString stringWithUTF8String:pn]];
            NSString *desc = v ? [v description] : @"nil";
            if ([desc length] > 200)
                desc = [[desc substringToIndex:200] stringByAppendingString:@"..."];
            printf("    %s = %s\n", pn, [desc UTF8String]);
        } @catch (NSException *ex) {
            printf("    %s = <exc: %s>\n", pn, [[ex reason] UTF8String]);
        }
    }
    free(pp);
}

static void list_dir(NSString *path) {
    NSFileManager *fm = [NSFileManager defaultManager];
    NSDirectoryEnumerator *en = [fm enumeratorAtPath:path];
    NSString *f;
    while ((f = [en nextObject])) {
        NSString *full = [path stringByAppendingPathComponent:f];
        BOOL isDir;
        [fm fileExistsAtPath:full isDirectory:&isDir];
        if (!isDir) {
            NSDictionary *a = [fm attributesOfItemAtPath:full error:nil];
            printf("    %s (%llu bytes)\n", [f UTF8String],
                   [[a objectForKey:NSFileSize] unsignedLongLongValue]);
        }
    }
}

static IOSurfaceRef make_surface(size_t bytes) {
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth:@(bytes), (id)kIOSurfaceHeight:@1,
        (id)kIOSurfaceBytesPerElement:@1, (id)kIOSurfaceBytesPerRow:@(bytes),
        (id)kIOSurfaceAllocSize:@(bytes), (id)kIOSurfacePixelFormat:@0});
}

#pragma mark - Main

int main(int argc, const char *argv[]) {
    (void)argc; (void)argv;
    @autoreleasepool {
        mach_timebase_info(&g_tb);
        printf("==============================================================\n");
        printf("  Experiments Q-S: CoreML-Compiled Model Chaining\n");
        printf("==============================================================\n\n");

        void *handle = dlopen(
            "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/"
            "AppleNeuralEngine", RTLD_NOW);
        if (!handle) { printf("FATAL: dlopen ANE framework failed\n"); return 1; }

        Class gAIO = NSClassFromString(@"_ANEIOSurfaceObject");
        Class gBuf = NSClassFromString(@"_ANEBuffer");
        Class gOutSets = NSClassFromString(@"_ANEIOSurfaceOutputSets");
        Class gChain = NSClassFromString(@"_ANEChainingRequest");
        Class gAR = NSClassFromString(@"_ANERequest");
        Class gANEModel = NSClassFromString(@"_ANEModel");
        id client = [NSClassFromString(@"_ANEClient")
            performSelector:@selector(sharedConnection)];

        if (!gAIO || !gBuf || !gOutSets || !gChain || !gAR || !gANEModel || !client) {
            printf("FATAL: Missing ANE classes\n");
            return 1;
        }

        // =================================================================
        // Experiment Q: CoreML-compile .mlpackage and extract _ANEModel
        // =================================================================
        printf("==============================================================\n");
        printf("  Experiment Q: CoreML Pipeline -> _ANEModel Extraction\n");
        printf("==============================================================\n\n");

        NSString *pkgPath = @"/tmp/ane_sram_256ch_64sp.mlpackage";
        NSFileManager *fm = [NSFileManager defaultManager];
        if (![fm fileExistsAtPath:pkgPath]) {
            printf("  FATAL: %s not found.\n", [pkgPath UTF8String]);
            printf("  Run: python3 scripts/gen_mlpackages.py\n");
            return 1;
        }

        printf("  --- Q.1: Compile .mlpackage -> .mlmodelc ---\n");
        NSError *err = nil;
        NSURL *srcURL = [NSURL fileURLWithPath:pkgPath];
        NSURL *compiledURL = [MLModel compileModelAtURL:srcURL error:&err];
        if (err || !compiledURL) {
            printf("  Compile FAILED: %s\n",
                   err ? [[err description] UTF8String] : "nil URL");
            return 1;
        }
        printf("  Compiled to: %s\n", [[compiledURL path] UTF8String]);
        fflush(stdout);

        printf("\n  --- Q.2: .mlmodelc contents ---\n");
        list_dir([compiledURL path]);

        printf("\n  --- Q.3: Load MLModel with ANE compute units ---\n");
        MLModelConfiguration *config = [[MLModelConfiguration alloc] init];
        config.computeUnits = MLComputeUnitsAll;
        err = nil;
        MLModel *mlModel = [MLModel modelWithContentsOfURL:compiledURL
                                             configuration:config error:&err];
        if (err || !mlModel) {
            printf("  Load FAILED: %s\n",
                   err ? [[err description] UTF8String] : "nil model");
            return 1;
        }
        printf("  MLModel loaded: %s\n",
               [NSStringFromClass([mlModel class]) UTF8String]);
        fflush(stdout);

        printf("\n  --- Q.4: Extract internal ANE model ---\n");
        fflush(stdout);
        id aneModel = nil;
        id aneProgram = nil;

        NSArray *kvcKeys = @[@"proxy", @"engine", @"aneModel", @"model",
            @"neuralNetworkEngine", @"aneEngine", @"compiledModel",
            @"_aneModel", @"_model", @"_engine"];
        for (NSString *key in kvcKeys) {
            @try {
                id val = [mlModel valueForKey:key];
                if (val) {
                    printf("  mlModel.%s = %s (%s)\n", [key UTF8String],
                           [[val description] UTF8String],
                           [NSStringFromClass([val class]) UTF8String]);
                    if ([val isKindOfClass:gANEModel]) {
                        aneModel = val;
                        printf("  *** Found _ANEModel via '%s' ***\n",
                               [key UTF8String]);
                    }
                }
            } @catch (NSException *ex) { (void)ex; }
        }

        if (!aneModel) {
            printf("\n  Trying deeper traversal...\n");
            for (NSString *key1 in @[@"proxy", @"engine"]) {
                id l1 = nil;
                @try { l1 = [mlModel valueForKey:key1]; }
                @catch (NSException *ex) { continue; }
                if (!l1) continue;
                printf("  L1: %s -> %s\n", [key1 UTF8String],
                       [NSStringFromClass([l1 class]) UTF8String]);

                for (NSString *key2 in @[@"model", @"aneModel", @"engine",
                    @"neuralNetworkEngine", @"aneEngine", @"_model",
                    @"compiledModel", @"program", @"espressoModel",
                    @"aneProgram", @"backend"]) {
                    @try {
                        id l2 = [l1 valueForKey:key2];
                        if (l2) {
                            printf("  L2: %s.%s -> %s (%s)\n",
                                   [key1 UTF8String], [key2 UTF8String],
                                   [NSStringFromClass([l2 class]) UTF8String],
                                   [[[l2 description] substringToIndex:
                                       MIN(100, [[l2 description] length])]
                                        UTF8String]);
                            if ([l2 isKindOfClass:gANEModel]) {
                                aneModel = l2;
                                printf("  *** Found _ANEModel via %s.%s ***\n",
                                       [key1 UTF8String], [key2 UTF8String]);
                            }

                            for (NSString *key3 in @[@"model", @"aneModel",
                                @"program", @"compiledModel", @"_model"]) {
                                @try {
                                    id l3 = [l2 valueForKey:key3];
                                    if (l3) {
                                        printf("  L3: %s.%s.%s -> %s\n",
                                            [key1 UTF8String],
                                            [key2 UTF8String],
                                            [key3 UTF8String],
                                            [NSStringFromClass([l3 class])
                                                UTF8String]);
                                        if ([l3 isKindOfClass:gANEModel]) {
                                            aneModel = l3;
                                            printf("  *** Found _ANEModel ***\n");
                                        }
                                    }
                                } @catch (NSException *ex) { (void)ex; }
                            }
                        }
                    } @catch (NSException *ex) { (void)ex; }
                }
            }
        }

        if (!aneModel) {
            printf("\n  Trying _ANEClient.loadModel: with .mlmodelc ---\n");
            @try {
                NSURL *espressoURL = compiledURL;
                NSString *espressoKey = [[compiledURL path] lastPathComponent];
                id diskModel = ((id(*)(Class,SEL,id,id))objc_msgSend)(
                    gANEModel, @selector(modelAtURL:key:),
                    espressoURL, espressoKey);
                if (diskModel) {
                    printf("  _ANEModel from mlmodelc: %s\n",
                           [[diskModel description] UTF8String]);
                    dump_props(diskModel);

                    printf("\n  Loading via _ANEClient...\n");
                    SEL loadSel = NSSelectorFromString(
                        @"loadModel:options:qos:error:");
                    NSError *loadErr = nil;
                    BOOL loadOk = ((BOOL(*)(id,SEL,id,id,unsigned int,
                        NSError**))objc_msgSend)(client, loadSel,
                        diskModel, @{}, (unsigned int)21, &loadErr);
                    printf("  loadModel: %s\n", loadOk ? "YES" : "NO");
                    if (loadErr)
                        printf("  Error: %s\n",
                               [[loadErr description] UTF8String]);

                    if (loadOk) {
                        aneModel = diskModel;
                        printf("  *** _ANEModel LOADED via client! ***\n");
                    } else {
                        SEL compileSel = NSSelectorFromString(
                            @"compileModel:options:qos:error:");
                        NSError *compErr = nil;
                        BOOL compOk = ((BOOL(*)(id,SEL,id,id,unsigned int,
                            NSError**))objc_msgSend)(client, compileSel,
                            diskModel, @{}, (unsigned int)21, &compErr);
                        printf("  compileModel: %s\n", compOk ? "YES" : "NO");
                        if (compErr)
                            printf("  Error: %s\n",
                                   [[compErr description] UTF8String]);
                        if (compOk) {
                            aneModel = diskModel;
                            printf("  *** _ANEModel COMPILED via client! ***\n");
                        }
                    }

                    dump_props(diskModel);
                }
            } @catch (NSException *ex) {
                printf("  EXCEPTION: %s\n", [[ex reason] UTF8String]);
            }
        }

        printf("\n  --- Q.5: Inspect extracted _ANEModel ---\n");
        if (aneModel) {
            printf("  _ANEModel class: %s\n",
                   [NSStringFromClass([aneModel class]) UTF8String]);
            dump_props(aneModel);

            printf("\n  Symbol indices:\n");
            @try {
                SEL inSel = NSSelectorFromString(
                    @"inputSymbolIndicesForProcedureIndex:");
                id inSyms = ((id(*)(id,SEL,unsigned int))objc_msgSend)(
                    aneModel, inSel, (unsigned int)0);
                printf("    inputSymbols(0): %s\n",
                       inSyms ? [[inSyms description] UTF8String] : "nil");
                SEL outSel = NSSelectorFromString(
                    @"outputSymbolIndicesForProcedureIndex:");
                id outSyms = ((id(*)(id,SEL,unsigned int))objc_msgSend)(
                    aneModel, outSel, (unsigned int)0);
                printf("    outputSymbols(0): %s\n",
                       outSyms ? [[outSyms description] UTF8String] : "nil");
            } @catch (NSException *ex) {
                printf("    Symbol EXCEPTION: %s\n",
                       [[ex reason] UTF8String]);
            }

            @try {
                SEL piSel = NSSelectorFromString(
                    @"procedureInfoForProcedureIndex:");
                id pInfo = ((id(*)(id,SEL,unsigned int))objc_msgSend)(
                    aneModel, piSel, (unsigned int)0);
                printf("    procedureInfo(0): %s\n",
                       pInfo ? [[pInfo description] UTF8String] : "nil");
                if (pInfo) {
                    printf("    procedureInfo class: %s\n",
                           [NSStringFromClass([pInfo class]) UTF8String]);
                    dump_props(pInfo);
                }
            } @catch (NSException *ex) {
                printf("    procedureInfo EXCEPTION: %s\n",
                       [[ex reason] UTF8String]);
            }

            @try {
                id mapper = [aneModel valueForKey:@"mapper"];
                printf("    mapper: %s\n",
                       mapper ? [[mapper description] UTF8String] : "nil");
            } @catch (NSException *ex) {
                printf("    mapper EXCEPTION: %s\n",
                       [[ex reason] UTF8String]);
            }

            aneProgram = nil;
            @try {
                aneProgram = [aneModel valueForKey:@"program"];
                printf("    program: %s\n",
                       aneProgram ? [[aneProgram description] UTF8String]
                                  : "nil");
            } @catch (NSException *ex) {
                printf("    program EXCEPTION: %s\n",
                       [[ex reason] UTF8String]);
            }

            @try {
                id ph = [aneModel valueForKey:@"programHandle"];
                printf("    programHandle: %s\n",
                       ph ? [[ph description] UTF8String] : "nil");
            } @catch (NSException *ex) { (void)ex; }

            @try {
                id st = [aneModel valueForKey:@"state"];
                printf("    state: %s\n",
                       st ? [[st description] UTF8String] : "nil");
            } @catch (NSException *ex) { (void)ex; }

            @try {
                id uuid = ((id(*)(id,SEL))objc_msgSend)(
                    aneModel, @selector(getUUID));
                printf("    getUUID: %s\n",
                       uuid ? [[uuid description] UTF8String] : "nil");
            } @catch (NSException *ex) {
                printf("    getUUID EXCEPTION: %s\n",
                       [[ex reason] UTF8String]);
            }
        } else {
            printf("  No _ANEModel extracted. Trying alternative approaches...\n");

            printf("\n  --- Q.5b: Deep ivar traversal ---\n");
            fflush(stdout);

            id e5Engine = nil;
            @try {
                e5Engine = [mlModel valueForKey:@"_internalEngine"];
            } @catch (NSException *ex) { (void)ex; }
            if (!e5Engine) {
                @try {
                    Ivar ivar = class_getInstanceVariable(
                        [mlModel class], "_internalEngine");
                    if (ivar) e5Engine = object_getIvar(mlModel, ivar);
                } @catch (NSException *ex) { (void)ex; }
            }

            id savedOpPool = nil;
            if (e5Engine) {
                printf("  MLE5Engine: %s\n",
                       [NSStringFromClass([e5Engine class]) UTF8String]);
                fflush(stdout);

                Class e5Cls = [e5Engine class];
                while (e5Cls && e5Cls != [NSObject class]) {
                    printf("\n  --- Ivars of %s ---\n",
                           [NSStringFromClass(e5Cls) UTF8String]);
                    fflush(stdout);
                    unsigned int ic;
                    Ivar *ivars = class_copyIvarList(e5Cls, &ic);
                    for (unsigned int i = 0; i < ic; i++) {
                        const char *name = ivar_getName(ivars[i]);
                        const char *type = ivar_getTypeEncoding(ivars[i]);
                        printf("    ivar: %s  type: %s\n", name,
                               type ? type : "?");
                        fflush(stdout);
                        if (type && type[0] == '@') {
                            @try {
                                id val = object_getIvar(e5Engine, ivars[i]);
                                if (val) {
                                    printf("      -> %s\n",
                                           [NSStringFromClass([val class])
                                               UTF8String]);
                                    fflush(stdout);
                                    if ([val isKindOfClass:gANEModel]) {
                                        aneModel = val;
                                        printf("      *** FOUND _ANEModel"
                                               " in MLE5Engine ***\n");
                                    }
                                }
                            } @catch (NSException *ex) { (void)ex; }
                        }
                    }
                    free(ivars);
                    e5Cls = class_getSuperclass(e5Cls);
                }

                if (!aneModel) {
                    printf("\n  --- Deep traversal: _programLibrary"
                           " and _operationPool ---\n");
                    fflush(stdout);

                    id targets[] = {nil, nil};
                    const char *tNames[] = {"programLibrary", "operationPool"};
                    @try {
                        targets[0] = [e5Engine valueForKey:@"programLibrary"];
                    } @catch (NSException *ex) { (void)ex; }
                    @try {
                        targets[1] = [e5Engine valueForKey:@"operationPool"];
                    } @catch (NSException *ex) { (void)ex; }

                    for (int ti = 0; ti < 2; ti++) {
                        if (!targets[ti]) continue;
                        printf("\n  [%s] %s\n", tNames[ti],
                               [NSStringFromClass([targets[ti] class])
                                   UTF8String]);
                        fflush(stdout);

                        Class tCls = [targets[ti] class];
                        while (tCls && tCls != [NSObject class]) {
                            unsigned int tic;
                            Ivar *tivars = class_copyIvarList(tCls, &tic);
                            for (unsigned int j = 0; j < tic; j++) {
                                const char *tn = ivar_getName(tivars[j]);
                                const char *tt = ivar_getTypeEncoding(tivars[j]);
                                printf("    ivar: %s  type: %s\n", tn,
                                       tt ? tt : "?");
                                fflush(stdout);
                                if (tt && tt[0] == '@') {
                                    @try {
                                        id tv = object_getIvar(targets[ti],
                                            tivars[j]);
                                        if (tv) {
                                            NSString *cls = NSStringFromClass(
                                                [tv class]);
                                            printf("      -> %s\n",
                                                   [cls UTF8String]);
                                            fflush(stdout);
                                            if ([tv isKindOfClass:gANEModel]) {
                                                aneModel = tv;
                                                printf("      *** FOUND"
                                                    " _ANEModel ***\n");
                                            }

                                            if ([cls containsString:@"ANE"]
                                                || [cls containsString:@"ane"]
                                                || [cls containsString:@"Plan"]
                                                || [cls containsString:@"Program"]
                                                || [cls containsString:@"Segment"]) {
                                                printf("      Digging into"
                                                    " %s...\n",
                                                    [cls UTF8String]);
                                                unsigned int sc;
                                                Ivar *sivars = class_copyIvarList(
                                                    [tv class], &sc);
                                                for (unsigned int si = 0;
                                                     si < sc && si < 30; si++) {
                                                    const char *sn = ivar_getName(sivars[si]);
                                                    const char *st = ivar_getTypeEncoding(sivars[si]);
                                                    printf("        .%s type=%s\n",
                                                           sn, st ? st : "?");
                                                    if (st && st[0] == '@') {
                                                        @try {
                                                            id sv = object_getIvar(tv, sivars[si]);
                                                            if (sv) {
                                                                printf("          -> %s\n",
                                                                    [NSStringFromClass([sv class]) UTF8String]);
                                                                if ([sv isKindOfClass:gANEModel]) {
                                                                    aneModel = sv;
                                                                    printf("          *** FOUND _ANEModel ***\n");
                                                                }
                                                            }
                                                        } @catch (NSException *ex) { (void)ex; }
                                                    }
                                                }
                                                free(sivars);
                                            }

                                            if ([tv isKindOfClass:[NSDictionary class]]) {
                                                NSDictionary *d = (NSDictionary *)tv;
                                                printf("      dict keys: %s\n",
                                                    [[[d allKeys] description] UTF8String]);
                                                for (id key in d) {
                                                    id dv = d[key];
                                                    printf("        [%s] -> %s\n",
                                                        [[key description] UTF8String],
                                                        [NSStringFromClass([dv class]) UTF8String]);
                                                    fflush(stdout);
                                                    if ([dv isKindOfClass:gANEModel]) {
                                                        aneModel = dv;
                                                        printf("        *** FOUND _ANEModel ***\n");
                                                    }
                                                    unsigned int dc;
                                                    Ivar *divars = class_copyIvarList([dv class], &dc);
                                                    for (unsigned int di = 0; di < dc && di < 20; di++) {
                                                        const char *dn = ivar_getName(divars[di]);
                                                        const char *dt = ivar_getTypeEncoding(divars[di]);
                                                        if (dt && dt[0] == '@') {
                                                            @try {
                                                                id ddv = object_getIvar(dv, divars[di]);
                                                                if (ddv && [ddv isKindOfClass:gANEModel]) {
                                                                    aneModel = ddv;
                                                                    printf("          *** FOUND _ANEModel in dict val ivar %s ***\n", dn);
                                                                } else if (ddv) {
                                                                    NSString *dcls = NSStringFromClass([ddv class]);
                                                                    if ([dcls containsString:@"ANE"])
                                                                        printf("          .%s -> %s\n", dn, [dcls UTF8String]);
                                                                }
                                                            } @catch (NSException *ex) { (void)ex; }
                                                        }
                                                    }
                                                    free(divars);
                                                }
                                            }

                                            if ([tv isKindOfClass:[NSArray class]]) {
                                                NSArray *arr = (NSArray *)tv;
                                                printf("      array count: %lu\n",
                                                    (unsigned long)[arr count]);
                                                for (NSUInteger ai = 0;
                                                     ai < [arr count] && ai < 5; ai++) {
                                                    id av = arr[ai];
                                                    printf("        [%lu] -> %s\n",
                                                        (unsigned long)ai,
                                                        [NSStringFromClass([av class]) UTF8String]);
                                                    if ([av isKindOfClass:gANEModel]) {
                                                        aneModel = av;
                                                        printf("        *** FOUND _ANEModel ***\n");
                                                    }
                                                }
                                            }
                                        }
                                    } @catch (NSException *ex) { (void)ex; }
                                }
                            }
                            free(tivars);
                            tCls = class_getSuperclass(tCls);
                        }
                    }
                }
                @try {
                    savedOpPool = [e5Engine valueForKey:@"operationPool"];
                } @catch (NSException *ex) { (void)ex; }
                if (!aneModel) {
                    printf("\n  --- Traversal: _pool in operationPool ---\n");
                    fflush(stdout);
                    @try {
                        id pool = [savedOpPool valueForKey:@"pool"];
                        if (pool && [pool isKindOfClass:[NSSet class]]) {
                            NSSet *s = (NSSet *)pool;
                            printf("  pool count: %lu\n",
                                   (unsigned long)[s count]);
                            for (id item in s) {
                                printf("    item: %s\n",
                                       [NSStringFromClass([item class])
                                           UTF8String]);
                                fflush(stdout);
                                if ([item isKindOfClass:gANEModel]) {
                                    aneModel = item;
                                    printf("    *** FOUND _ANEModel ***\n");
                                }
                                unsigned int pic;
                                Ivar *pivars = class_copyIvarList(
                                    [item class], &pic);
                                for (unsigned int pi = 0;
                                     pi < pic && pi < 30; pi++) {
                                    const char *pn = ivar_getName(pivars[pi]);
                                    const char *pt = ivar_getTypeEncoding(pivars[pi]);
                                    printf("      .%s type=%s\n", pn,
                                           pt ? pt : "?");
                                    if (pt && pt[0] == '@') {
                                        @try {
                                            id pv = object_getIvar(
                                                item, pivars[pi]);
                                            if (pv) {
                                                NSString *pcls =
                                                    NSStringFromClass([pv class]);
                                                printf("        -> %s\n",
                                                       [pcls UTF8String]);
                                                if ([pv isKindOfClass:gANEModel]) {
                                                    aneModel = pv;
                                                    printf("        *** FOUND"
                                                        " _ANEModel ***\n");
                                                }
                                                if ([pcls containsString:@"ANE"]
                                                    || [pcls containsString:@"Plan"]
                                                    || [pcls containsString:@"Program"]
                                                    || [pcls containsString:@"Stream"]) {
                                                    unsigned int sic;
                                                    Ivar *sivar = class_copyIvarList(
                                                        [pv class], &sic);
                                                    for (unsigned int si = 0;
                                                         si < sic && si < 20; si++) {
                                                        const char *sn = ivar_getName(sivar[si]);
                                                        const char *st2 = ivar_getTypeEncoding(sivar[si]);
                                                        printf("          .%s type=%s\n",
                                                               sn, st2 ? st2 : "?");
                                                        if (st2 && st2[0] == '@') {
                                                            @try {
                                                                id sv = object_getIvar(pv, sivar[si]);
                                                                if (sv) {
                                                                    printf("            -> %s\n",
                                                                        [NSStringFromClass([sv class]) UTF8String]);
                                                                    if ([sv isKindOfClass:gANEModel]) {
                                                                        aneModel = sv;
                                                                        printf("            *** FOUND _ANEModel ***\n");
                                                                    }
                                                                }
                                                            } @catch (NSException *ex) { (void)ex; }
                                                        }
                                                    }
                                                    free(sivar);
                                                }
                                            }
                                        } @catch (NSException *ex) { (void)ex; }
                                    }
                                }
                                free(pivars);
                            }
                        }
                    } @catch (NSException *ex) {
                        printf("  Pool EXCEPTION: %s\n",
                               [[ex reason] UTF8String]);
                    }
                }

                if (!aneModel) {
                    printf("\n  --- Force prediction to trigger ANE load ---\n");
                    fflush(stdout);
                    @try {
                        MLModelDescription *desc = [mlModel modelDescription];
                        NSDictionary *inputs = [desc inputDescriptionsByName];
                        printf("  Input features:\n");
                        for (NSString *name in inputs) {
                            MLFeatureDescription *fd = inputs[name];
                            printf("    %s: type=%ld\n", [name UTF8String],
                                   (long)fd.type);
                        }

                        NSString *inputName = [[inputs allKeys] firstObject];
                        if (inputName) {
                            MLMultiArray *arr = [[MLMultiArray alloc]
                                initWithShape:@[@1, @256, @1, @64]
                                dataType:MLMultiArrayDataTypeFloat32
                                error:nil];
                            MLDictionaryFeatureProvider *fp =
                                [[MLDictionaryFeatureProvider alloc]
                                    initWithDictionary:@{inputName: arr}
                                    error:nil];
                            NSError *predErr = nil;
                            id result = [mlModel predictionFromFeatures:fp
                                error:&predErr];
                            printf("  Prediction: %s\n",
                                   result ? "SUCCESS" : "FAILED");
                            if (predErr)
                                printf("  Error: %s\n",
                                       [[predErr description] UTF8String]);
                            fflush(stdout);
                        }

                        @try {
                            id pool2 = [savedOpPool valueForKey:@"pool"];
                            if (pool2 && [pool2 isKindOfClass:[NSSet class]]) {
                                printf("\n  Pool after prediction (count=%lu):\n",
                                       (unsigned long)[(NSSet *)pool2 count]);
                                for (id item in (NSSet *)pool2) {
                                    printf("    %s\n",
                                           [NSStringFromClass([item class])
                                               UTF8String]);
                                    fflush(stdout);
                                    unsigned int pic;
                                    Ivar *pivars = class_copyIvarList(
                                        [item class], &pic);
                                    for (unsigned int pi = 0;
                                         pi < pic && pi < 30; pi++) {
                                        const char *pn = ivar_getName(pivars[pi]);
                                        const char *pt = ivar_getTypeEncoding(pivars[pi]);
                                        if (pt && pt[0] == '@') {
                                            @try {
                                                id pv = object_getIvar(item, pivars[pi]);
                                                if (pv) {
                                                    printf("      .%s -> %s\n", pn,
                                                        [NSStringFromClass([pv class]) UTF8String]);
                                                    if ([pv isKindOfClass:gANEModel]) {
                                                        aneModel = pv;
                                                        printf("      *** FOUND _ANEModel ***\n");
                                                    }
                                                }
                                            } @catch (NSException *ex) { (void)ex; }
                                        }
                                    }
                                    free(pivars);
                                }
                            }
                        } @catch (NSException *ex) { (void)ex; }
                    } @catch (NSException *ex) {
                        printf("  Prediction EXCEPTION: %s\n",
                               [[ex reason] UTF8String]);
                    }
                }

            } else {
                printf("  MLE5Engine: NOT FOUND\n");
            }
        }

        // =================================================================
        // Experiment R: Chaining with CoreML-loaded model
        // =================================================================
        printf("\n==============================================================\n");
        printf("  Experiment R: Chaining with CoreML-loaded _ANEModel\n");
        printf("==============================================================\n\n");

        BOOL chainingSuccess = NO;

        if (!aneModel) {
            printf("  SKIPPED: no _ANEModel available\n");
        } else {
            int ch = 256, sp = 64;
            size_t bufSize = (size_t)ch * sp * 4;
            IOSurfaceRef ioIn = make_surface(bufSize);
            IOSurfaceRef ioOut = make_surface(bufSize);

            printf("  --- R.1: Baseline eval via _ANERequest ---\n");
            @try {
                id ioObjIn = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
                    gAIO, @selector(objectWithIOSurface:), ioIn);
                id ioObjOut = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
                    gAIO, @selector(objectWithIOSurface:), ioOut);
                id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))
                    objc_msgSend)(gAR,
                    @selector(requestWithInputs:inputIndices:outputs:
                        outputIndices:weightsBuffer:perfStats:procedureIndex:),
                    @[ioObjIn], @[@0], @[ioObjOut], @[@0], nil, nil, @0);

                if (req) {
                    NSError *evalErr = nil;
                    BOOL evalOk = ((BOOL(*)(id,SEL,id,id,id,unsigned int,
                        NSError**))objc_msgSend)(client,
                        @selector(evaluateWithModel:options:request:qos:error:),
                        aneModel, @{}, req, (unsigned int)21, &evalErr);
                    printf("  Single eval: %s\n", evalOk ? "YES" : "NO");
                    if (evalErr)
                        printf("  Error: %s\n",
                               [[evalErr description] UTF8String]);

                    if (evalOk) {
                        int niters = 100;
                        uint64_t t0 = mach_absolute_time();
                        for (int i = 0; i < niters; i++) {
                            ((BOOL(*)(id,SEL,id,id,id,unsigned int,
                                NSError**))objc_msgSend)(client,
                                @selector(evaluateWithModel:options:request:
                                    qos:error:),
                                aneModel, @{}, req, (unsigned int)21, nil);
                        }
                        double elapsed = tb_ms(mach_absolute_time() - t0);
                        printf("  Baseline: %d iters in %.3f ms (%.4f ms/eval)\n",
                               niters, elapsed, elapsed / niters);
                    }
                }
            } @catch (NSException *ex) {
                printf("  Baseline EXCEPTION: %s\n",
                       [[ex reason] UTF8String]);
            }

            printf("\n  --- R.2: ChainingRequest with nil params ---\n");
            @try {
                id ioObjIn = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
                    gAIO, @selector(objectWithIOSurface:), ioIn);
                id inBuf = ((id(*)(Class,SEL,id,id,long long))objc_msgSend)(
                    gBuf,
                    @selector(bufferWithIOSurfaceObject:symbolIndex:source:),
                    ioObjIn, @0, (long long)0);

                id ioObjOut = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
                    gAIO, @selector(objectWithIOSurface:), ioOut);
                id outBuf = ((id(*)(Class,SEL,id,id,long long))objc_msgSend)(
                    gBuf,
                    @selector(bufferWithIOSurfaceObject:symbolIndex:source:),
                    ioObjOut, @0, (long long)1);

                IOSurfaceRef statsSurf = make_surface(64);
                id outSet = ((id(*)(Class,SEL,IOSurfaceRef,id))objc_msgSend)(
                    gOutSets, @selector(objectWithstatsSurRef:outputBuffer:),
                    statsSurf, @[outBuf]);

                id cr = ((id(*)(Class,SEL,id,id,id,id,id,id,id,id,id))
                    objc_msgSend)(gChain,
                    @selector(chainingRequestWithInputs:outputSets:
                        lbInputSymbolId:lbOutputSymbolId:procedureIndex:
                        signalEvents:transactionHandle:fwEnqueueDelay:
                        memoryPoolId:),
                    @[inBuf], @[outSet], nil, nil, nil,
                    @[], @0, @0, @0);

                if (!cr) {
                    printf("  ChainingRequest: nil\n");
                } else {
                    BOOL valid = ((BOOL(*)(id,SEL))objc_msgSend)(
                        cr, @selector(validate));
                    printf("  validate: %s\n", valid ? "YES" : "NO");
                    printf("  desc: %.200s\n",
                           [[cr description] UTF8String]);

                    NSError *prepErr = nil;
                    BOOL prepOk = ((BOOL(*)(id,SEL,id,id,id,unsigned int,
                        NSError**))objc_msgSend)(client,
                        @selector(prepareChainingWithModel:options:
                            chainingReq:qos:error:),
                        aneModel, @{}, cr, (unsigned int)21, &prepErr);
                    printf("  prepareChainingWithModel: %s\n",
                           prepOk ? "YES" : "NO");
                    if (prepErr)
                        printf("  Error: %s\n",
                               [[prepErr description] UTF8String]);

                    if (prepOk) {
                        chainingSuccess = YES;
                        printf("  *** PREPARE SUCCEEDED! ***\n");

                        printf("\n  --- R.3: enqueueSetsWithModel ---\n");
                        @try {
                            SEL eqSel = NSSelectorFromString(
                                @"enqueueSetsWithModel:outputSet:"
                                "options:qos:error:");
                            NSError *eqErr = nil;
                            BOOL eqOk = ((BOOL(*)(id,SEL,id,id,id,
                                unsigned int,NSError**))objc_msgSend)(
                                client, eqSel, aneModel, outSet, @{},
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

                        printf("\n  --- R.4: buffersReadyWithModel ---\n");
                        @try {
                            SEL brSel = NSSelectorFromString(
                                @"buffersReadyWithModel:inputBuffers:"
                                "options:qos:error:");
                            NSError *brErr = nil;
                            BOOL brOk = ((BOOL(*)(id,SEL,id,id,id,
                                unsigned int,NSError**))objc_msgSend)(
                                client, brSel, aneModel, @[inBuf], @{},
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
                    }
                }
                CFRelease(statsSurf);
            } @catch (NSException *ex) {
                printf("  EXCEPTION: %s\n", [[ex reason] UTF8String]);
            }

            if (!chainingSuccess) {
                printf("\n  --- R.2b: Try with symbol indices ---\n");
                @try {
                    SEL inSel = NSSelectorFromString(
                        @"inputSymbolIndicesForProcedureIndex:");
                    id inSyms = ((id(*)(id,SEL,unsigned int))objc_msgSend)(
                        aneModel, inSel, (unsigned int)0);
                    SEL outSel = NSSelectorFromString(
                        @"outputSymbolIndicesForProcedureIndex:");
                    id outSyms = ((id(*)(id,SEL,unsigned int))objc_msgSend)(
                        aneModel, outSel, (unsigned int)0);
                    printf("  inputSymbols: %s\n",
                           inSyms ? [[inSyms description] UTF8String] : "nil");
                    printf("  outputSymbols: %s\n",
                           outSyms ? [[outSyms description] UTF8String] : "nil");

                    NSUInteger firstIn = 0, firstOut = 0;
                    if (inSyms && [inSyms isKindOfClass:[NSIndexSet class]]
                        && [(NSIndexSet *)inSyms count] > 0)
                        firstIn = [(NSIndexSet *)inSyms firstIndex];
                    if (outSyms && [outSyms isKindOfClass:[NSIndexSet class]]
                        && [(NSIndexSet *)outSyms count] > 0)
                        firstOut = [(NSIndexSet *)outSyms firstIndex];

                    printf("  Using symbolIndex: in=%lu out=%lu\n",
                           (unsigned long)firstIn, (unsigned long)firstOut);

                    id ioObjIn = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
                        gAIO, @selector(objectWithIOSurface:), ioIn);
                    id inBuf = ((id(*)(Class,SEL,id,id,long long))
                        objc_msgSend)(gBuf,
                        @selector(bufferWithIOSurfaceObject:symbolIndex:
                            source:),
                        ioObjIn, @(firstIn), (long long)0);

                    id ioObjOut = ((id(*)(Class,SEL,IOSurfaceRef))
                        objc_msgSend)(
                        gAIO, @selector(objectWithIOSurface:), ioOut);
                    id outBuf = ((id(*)(Class,SEL,id,id,long long))
                        objc_msgSend)(gBuf,
                        @selector(bufferWithIOSurfaceObject:symbolIndex:
                            source:),
                        ioObjOut, @(firstOut), (long long)1);

                    IOSurfaceRef statsSurf = make_surface(64);
                    id outSet = ((id(*)(Class,SEL,IOSurfaceRef,id))
                        objc_msgSend)(gOutSets,
                        @selector(objectWithstatsSurRef:outputBuffer:),
                        statsSurf, @[outBuf]);

                    id cr = ((id(*)(Class,SEL,id,id,id,id,id,id,id,id,id))
                        objc_msgSend)(gChain,
                        @selector(chainingRequestWithInputs:outputSets:
                            lbInputSymbolId:lbOutputSymbolId:procedureIndex:
                            signalEvents:transactionHandle:fwEnqueueDelay:
                            memoryPoolId:),
                        @[inBuf], @[outSet], nil, nil, nil,
                        @[], @0, @0, @0);

                    if (cr) {
                        BOOL valid = ((BOOL(*)(id,SEL))objc_msgSend)(
                            cr, @selector(validate));
                        printf("  validate: %s\n", valid ? "YES" : "NO");

                        NSError *prepErr = nil;
                        BOOL prepOk = ((BOOL(*)(id,SEL,id,id,id,unsigned int,
                            NSError**))objc_msgSend)(client,
                            @selector(prepareChainingWithModel:options:
                                chainingReq:qos:error:),
                            aneModel, @{}, cr, (unsigned int)21, &prepErr);
                        printf("  prepare (with symbols): %s\n",
                               prepOk ? "YES" : "NO");
                        if (prepErr)
                            printf("  Error: %s\n",
                                   [[prepErr description] UTF8String]);
                        if (prepOk) {
                            chainingSuccess = YES;
                            printf("  *** PREPARE SUCCEEDED WITH SYMBOLS! ***\n");
                        }
                    }
                    CFRelease(statsSurf);
                } @catch (NSException *ex) {
                    printf("  EXCEPTION: %s\n", [[ex reason] UTF8String]);
                }
            }

            CFRelease(ioIn);
            CFRelease(ioOut);
        }

        // =================================================================
        // Experiment S: Two-kernel chaining (if R succeeded)
        // =================================================================
        printf("\n==============================================================\n");
        printf("  Experiment S: Two-Kernel Chaining\n");
        printf("==============================================================\n\n");
        if (!chainingSuccess) {
            printf("  SKIPPED: prepareChainingWithModel not yet working\n");
            printf("  (Requires success in Experiment R first)\n");
        } else {
            printf("  TODO: implement two-kernel chaining pipeline\n");
        }

        // =================================================================
        // Summary
        // =================================================================
        printf("\n============================================================\n");
        printf("  RESULTS SUMMARY\n");
        printf("============================================================\n");
        printf("  Exp Q: CoreML pipeline:       %s\n",
               aneModel ? "MODEL EXTRACTED" : "NO MODEL");
        printf("  Exp R: Chaining:              %s\n",
               chainingSuccess ? "SUCCESS" : "not yet");
        printf("  Exp S: Multi-kernel:          %s\n",
               chainingSuccess ? "TODO" : "BLOCKED");
        printf("============================================================\n");
        printf("\nDone.\n");
    }
    return 0;
}
