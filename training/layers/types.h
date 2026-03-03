// layers/types.h — ANE SDK Type Definitions
#pragma once
#import "../stories_io.h"

typedef enum {
    ANESDK_LAYER_LINEAR,
    ANESDK_LAYER_CONV2D,
    ANESDK_LAYER_RELU,
    ANESDK_LAYER_GELU,
    ANESDK_LAYER_SIGMOID,
    ANESDK_LAYER_RMSNORM,
    ANESDK_LAYER_LAYERNORM,
    ANESDK_LAYER_SOFTMAX,
    ANESDK_LAYER_ADD,
    ANESDK_LAYER_MUL
} ANESDKLayerType;

typedef struct {
    char name[64];
    ANESDKLayerType type;
    Kern *kern;
    
    // Weight surfaces (if any)
    int n_weights;
    IOSurfaceRef *weights;
    
    // Dimension metadata
    int in_ch, in_h, in_w;
    int out_ch, out_h, out_w;
} ANESDKLayer;

typedef struct {
    int n_layers;
    ANESDKLayer *layers;
    
    // Global activation surfaces
    // In a Sequential model, these can be ping-ponged
    IOSurfaceRef act_a;
    IOSurfaceRef act_b;
} ANESDKModel;
