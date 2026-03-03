// layers/core.h — Modular ANE SDK layer builders
#pragma once
#import <Foundation/Foundation.h>

#define ANESDK_MIL_HDR \
    @"program(1.3)\n[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, " \
    "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, " \
    "{\"coremltools-version\", \"9.0\"}})]\n{\n"

#define ANESDK_CONV_CONST \
    @"        string pt = const()[name=string(\"pt\"), val=string(\"valid\")];\n" \
    "        tensor<int32, [2]> st = const()[name=string(\"st\"), val=tensor<int32, [2]>([1,1])];\n" \
    "        tensor<int32, [4]> pd = const()[name=string(\"pd\"), val=tensor<int32, [4]>([0,0,0,0])];\n" \
    "        tensor<int32, [2]> dl = const()[name=string(\"dl\"), val=tensor<int32, [2]>([1,1])];\n" \
    "        int32 gr = const()[name=string(\"gr\"), val=int32(1)];\n"

/**
 * Linear Layer (Matmul)
 * y = x @ W^T
 * MIL Implementation: conv(x, W) where W is [out_ch, in_ch, 1, 1]
 */
static NSString *anesdk_gen_linear_fwd(int in_dim, int out_dim, int seq) {
    NSMutableString *m = [NSMutableString string];
    [m appendString:ANESDK_MIL_HDR];
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x, "
                      "tensor<fp16, [%d, %d, 1, 1]> W) {\n", 
                      in_dim, seq, out_dim, in_dim];
    [m appendString:ANESDK_CONV_CONST];
    [m appendFormat:@"        tensor<fp16, [1, %d, 1, %d]> y = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W,x=x);\n", out_dim, seq];
    [m appendString:@"    } -> (y);\n}\n"];
    return m;
}

/**
 * ReLU Activation
 */
static NSString *anesdk_gen_relu_fwd(int dim, int seq) {
    NSMutableString *m = [NSMutableString string];
    [m appendString:ANESDK_MIL_HDR];
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n", dim, seq];
    [m appendFormat:@"        tensor<fp16, [1, %d, 1, %d]> y = relu(x=x);\n", dim, seq];
    [m appendString:@"    } -> (y);\n}\n"];
    return m;
}

/**
 * GELU Activation
 */
static NSString *anesdk_gen_gelu_fwd(int dim, int seq) {
    NSMutableString *m = [NSMutableString string];
    [m appendString:ANESDK_MIL_HDR];
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n", dim, seq];
    [m appendFormat:@"        tensor<fp16, [1, %d, 1, %d]> y = gelu(x=x);\n", dim, seq];
    [m appendString:@"    } -> (y);\n}\n"];
    return m;
}

/**
 * Sigmoid Activation
 */
static NSString *anesdk_gen_sigmoid_fwd(int dim, int seq) {
    NSMutableString *m = [NSMutableString string];
    [m appendString:ANESDK_MIL_HDR];
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n", dim, seq];
    [m appendFormat:@"        tensor<fp16, [1, %d, 1, %d]> y = sigmoid(x=x);\n", dim, seq];
    [m appendString:@"    } -> (y);\n}\n"];
    return m;
}

/**
 * RMSNorm Layer
 * y = x * rsqrt(mean(x^2) + eps) * weight
 */
static NSString *anesdk_gen_rmsnorm_fwd(int dim, int seq) {
    float invd = 1.0f/(float)dim;
    NSMutableString *m = [NSMutableString string];
    [m appendString:ANESDK_MIL_HDR];
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x, "
                      "tensor<fp16, [1, %d, 1, 1]> weight) {\n", 
                      dim, seq, dim];
    [m appendFormat:@"        tensor<fp16, [1, %d, 1, %d]> sq = mul(x=x, y=x);\n", dim, seq];
    [m appendString:@"        tensor<int32, [1]> rax = const()[name=string(\"rax\"), val=tensor<int32, [1]>([1])];\n"];
    [m appendString:@"        bool kd = const()[name=string(\"kd\"), val=bool(true)];\n"];
    [m appendFormat:@"        tensor<fp16, [1, 1, 1, %d]> ss = reduce_sum(x=sq, axes=rax, keep_dims=kd);\n", seq];
    [m appendFormat:@"        fp16 invd = const()[name=string(\"invd\"), val=fp16(%f)];\n", invd];
    [m appendFormat:@"        tensor<fp16, [1, 1, 1, %d]> ss2 = mul(x=ss, y=invd);\n", seq];
    [m appendFormat:@"        fp16 eps = const()[name=string(\"eps\"), val=fp16(0.00001)];\n"];
    [m appendFormat:@"        tensor<fp16, [1, 1, 1, %d]> ss3 = add(x=ss2, y=eps);\n", seq];
    [m appendFormat:@"        fp16 nhalf = const()[name=string(\"nhalf\"), val=fp16(-0.5)];\n"];
    [m appendFormat:@"        tensor<fp16, [1, 1, 1, %d]> rrms = pow(x=ss3, y=nhalf);\n", seq];
    [m appendFormat:@"        tensor<fp16, [1, %d, 1, %d]> xr = mul(x=x, y=rrms);\n", dim, seq];
    [m appendFormat:@"        tensor<fp16, [1, %d, 1, %d]> out = mul(x=xr, y=weight);\n", dim, seq];
    [m appendString:@"    } -> (out);\n}\n"];
    return m;
}

/**
 * Element-wise Addition (Residual connection)
 */
static NSString *anesdk_gen_add_fwd(int dim, int seq) {
    NSMutableString *m = [NSMutableString string];
    [m appendString:ANESDK_MIL_HDR];
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x, tensor<fp16, [1, %d, 1, %d]> y) {\n", dim, seq, dim, seq];
    [m appendFormat:@"        tensor<fp16, [1, %d, 1, %d]> out = add(x=x, y=y);\n", dim, seq];
    [m appendString:@"    } -> (out);\n}\n"];
    return m;
}

/**
 * Softmax Activation
 */
static NSString *anesdk_gen_softmax_fwd(int dim, int seq) {
    NSMutableString *m = [NSMutableString string];
    [m appendString:ANESDK_MIL_HDR];
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n", dim, seq];
    [m appendString:@"        int32 axis = const()[name=string(\"axis\"), val=int32(1)];\n"]; // Softmax over dim
    [m appendFormat:@"        tensor<fp16, [1, %d, 1, %d]> y = softmax(x=x, axis=axis);\n", dim, seq];
    [m appendString:@"    } -> (y);\n}\n"];
    return m;
}

/**
 * LayerNorm Layer
 * y = (x - mean) / sqrt(var + eps) * weight + bias
 */
static NSString *anesdk_gen_layernorm_fwd(int dim, int seq) {
    NSMutableString *m = [NSMutableString string];
    [m appendString:ANESDK_MIL_HDR];
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x, "
                      "tensor<fp16, [1, %d, 1, 1]> weight, "
                      "tensor<fp16, [1, %d, 1, 1]> bias) {\n", 
                      dim, seq, dim, dim];
    [m appendString:@"        tensor<int32, [1]> rax = const()[name=string(\"rax\"), val=tensor<int32, [1]>([1])];\n"];
    [m appendString:@"        bool kd = const()[name=string(\"kd\"), val=bool(true)];\n"];
    [m appendFormat:@"        tensor<fp16, [1, 1, 1, %d]> mean = reduce_mean(x=x, axes=rax, keep_dims=kd);\n", seq];
    [m appendFormat:@"        tensor<fp16, [1, %d, 1, %d]> x_sub = sub(x=x, y=mean);\n", dim, seq];
    [m appendFormat:@"        tensor<fp16, [1, %d, 1, %d]> sq = mul(x=x_sub, y=x_sub);\n", dim, seq];
    [m appendFormat:@"        tensor<fp16, [1, 1, 1, %d]> var = reduce_mean(x=sq, axes=rax, keep_dims=kd);\n", seq];
    [m appendFormat:@"        fp16 eps = const()[name=string(\"eps\"), val=fp16(0.00001)];\n"];
    [m appendFormat:@"        tensor<fp16, [1, 1, 1, %d]> var_eps = add(x=var, y=eps);\n", seq];
    [m appendFormat:@"        fp16 nhalf = const()[name=string(\"nhalf\"), val=fp16(-0.5)];\n"];
    [m appendFormat:@"        tensor<fp16, [1, 1, 1, %d]> inv_std = pow(x=var_eps, y=nhalf);\n", seq];
    [m appendFormat:@"        tensor<fp16, [1, %d, 1, %d]> x_norm = mul(x=x_sub, y=inv_std);\n", dim, seq];
    [m appendFormat:@"        tensor<fp16, [1, %d, 1, %d]> x_scale = mul(x=x_norm, y=weight);\n", dim, seq];
    [m appendFormat:@"        tensor<fp16, [1, %d, 1, %d]> out = add(x=x_scale, y=bias);\n", dim, seq];
    [m appendString:@"    } -> (out);\n}\n"];
    return m;
}
