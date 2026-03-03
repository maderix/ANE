// layers/anesdk.h — High-level ANE SDK API
#pragma once
#import "types.h"
#import "core.h"
#import "cnn.h"

/**
 * Initialize a Linear (Dense) layer
 */
static ANESDKLayer anesdk_linear_create(const char *name, int in_dim, int out_dim, int seq) {
    ANESDKLayer l = {0};
    strncpy(l.name, name, 63);
    l.type = ANESDK_LAYER_LINEAR;
    l.in_ch = in_dim; l.in_w = seq; l.in_h = 1;
    l.out_ch = out_dim; l.out_w = seq; l.out_h = 1;
    
    NSString *mil = anesdk_gen_linear_fwd(in_dim, out_dim, seq);
    int in_sizes[] = { in_dim * seq * 2, out_dim * in_dim * 2 }; // input x, weight W
    l.kern = compile_kern_mil_w(mil, @{}, in_sizes, 2, out_dim * seq * 2);
    
    return l;
}

/**
 * Initialize a Conv2D layer
 */
static ANESDKLayer anesdk_conv2d_create(const char *name, int in_ch, int out_ch, int in_h, int in_w, 
                                        int k_h, int k_w, int stride_h, int stride_w, int pad) {
    ANESDKLayer l = {0};
    strncpy(l.name, name, 63);
    l.type = ANESDK_LAYER_CONV2D;
    l.in_ch = in_ch; l.in_h = in_h; l.in_w = in_w;
    
    int out_h = (in_h + 2*pad - k_h) / stride_h + 1;
    int out_w = (in_w + 2*pad - k_w) / stride_w + 1;
    l.out_ch = out_ch; l.out_h = out_h; l.out_w = out_w;
    
    NSString *mil = anesdk_gen_conv2d_fwd(in_ch, out_ch, in_h, in_w, k_h, k_w, stride_h, stride_w, pad, pad, pad, pad, 1, 1);
    int in_sizes[] = { in_ch * in_h * in_w * 2, out_ch * in_ch * k_h * k_w * 2 };
    l.kern = compile_kern_mil_w(mil, @{}, in_sizes, 2, out_ch * out_h * out_w * 2);
    
    return l;
}

/**
 * Initialize a ReLU layer
 */
static ANESDKLayer anesdk_relu_create(const char *name, int ch, int h, int w) {
    ANESDKLayer l = {0};
    strncpy(l.name, name, 63);
    l.type = ANESDK_LAYER_RELU;
    l.in_ch = ch; l.in_h = h; l.in_w = w;
    l.out_ch = ch; l.out_h = h; l.out_w = w;
    
    NSString *mil = anesdk_gen_relu_fwd(ch, h * w);
    int in_sizes[] = { ch * h * w * 2 };
    l.kern = compile_kern_mil_w(mil, @{}, in_sizes, 1, ch * h * w * 2);
    
    return l;
}

/**
 * Initialize a Softmax activation
 */
static ANESDKLayer anesdk_softmax_create(const char *name, int ch, int h, int w) {
    ANESDKLayer l = {0};
    strncpy(l.name, name, 63);
    l.type = ANESDK_LAYER_SOFTMAX;
    l.in_ch = ch; l.in_h = h; l.in_w = w;
    l.out_ch = ch; l.out_h = h; l.out_w = w;
    
    NSString *mil = anesdk_gen_softmax_fwd(ch, h * w);
    int in_sizes[] = { ch * h * w * 2 };
    l.kern = compile_kern_mil_w(mil, @{}, in_sizes, 1, ch * h * w * 2);
    
    return l;
}

/**
 * Initialize a LayerNorm layer
 * weight: [dim], bias: [dim]
 */
static ANESDKLayer anesdk_layernorm_create(const char *name, int dim, int seq) {
    ANESDKLayer l = {0};
    strncpy(l.name, name, 63);
    l.type = ANESDK_LAYER_LAYERNORM;
    l.in_ch = dim; l.in_w = seq; l.in_h = 1;
    l.out_ch = dim; l.out_w = seq; l.out_h = 1;
    
    NSString *mil = anesdk_gen_layernorm_fwd(dim, seq);
    int in_sizes[] = { dim * seq * 2, dim * 2, dim * 2 }; // x, weight, bias
    l.kern = compile_kern_mil_w(mil, @{}, in_sizes, 3, dim * seq * 2);
    
    return l;
}

/**
 * Execute a layer
 */
static void anesdk_layer_forward(ANESDKLayer *l) {
    ane_eval(l->kern);
}

/**
 * Initialize a Sequential model from an array of layers
 */
static ANESDKModel anesdk_model_sequential_create(ANESDKLayer *layers, int n_layers) {
    ANESDKModel m = {0};
    m.n_layers = n_layers;
    m.layers = (ANESDKLayer*)malloc(n_layers * sizeof(ANESDKLayer));
    memcpy(m.layers, layers, n_layers * sizeof(ANESDKLayer));
    
    // We can optimize activation memory by ping-ponging two surfaces
    // Layer 1: ioIn -> ioOut(A)
    // Layer 2: ioOut(A) -> ioOut(B)
    // Layer 3: ioOut(B) -> ioOut(A)
    // To do this, we must replace the input IOSurfaceRef in the Kern for each layer
    for (int i=1; i<n_layers; i++) {
        // Replace input surface of layer i with output of layer i-1
        CFRelease(m.layers[i].kern->inputs[0]);
        m.layers[i].kern->inputs[0] = (IOSurfaceRef)CFRetain(m.layers[i-1].kern->ioOut);
        
        // Update the ANE request to use the new surface
        id wI = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), m.layers[i].kern->inputs[0]);
        id wO = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), m.layers[i].kern->ioOut);
        
        // This is a simplified recreate of the request
        // In a real SDK, we'd need a more robust way to manage input indices
        // For Sequential, we assume inputs[0] is the activation input
        NSMutableArray *inObs = [NSMutableArray arrayWithObject:wI];
        NSMutableArray *inIdx = [NSMutableArray arrayWithObject:@0];
        
        // If the layer has additional weights (like Linear's inputs[1]), we keep them
        for (int j=1; j<m.layers[i].kern->n_inputs; j++) {
            [inObs addObject:((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), m.layers[i].kern->inputs[j])];
            [inIdx addObject:@(j)];
        }
        
        CFRelease(m.layers[i].kern->request);
        m.layers[i].kern->request = (void*)CFBridgingRetain(((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_AR,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            inObs, inIdx, @[wO], @[@0], nil, nil, @0));
    }
    
    return m;
}

/**
 * Forward pass for the entire model
 */
static void anesdk_model_forward(ANESDKModel *m) {
    for (int i=0; i<m->n_layers; i++) {
        ane_eval(m->layers[i].kern);
    }
}
