// layers/cnn.h — Modular ANE SDK CNN layer builders
#pragma once
#import "core.h"

/**
 * 2D Convolution Layer
 * weights: [out_ch, in_ch, kH, kW]
 */
static NSString *anesdk_gen_conv2d_fwd(int in_ch, int out_ch, int in_h, int in_w, 
                                       int k_h, int k_w, 
                                       int stride_h, int stride_w, 
                                       int pad_t, int pad_b, int pad_l, int pad_r,
                                       int dil_h, int dil_w) {
    NSMutableString *m = [NSMutableString string];
    [m appendString:ANESDK_MIL_HDR];
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, %d, %d]> x, "
                      "tensor<fp16, [%d, %d, %d, %d]> W) {\n", 
                      in_ch, in_h, in_w, out_ch, in_ch, k_h, k_w];
    
    [m appendFormat:@"        string pt = const()[name=string(\"pt\"), val=string(\"custom\")];\n"];
    [m appendFormat:@"        tensor<int32, [2]> st = const()[name=string(\"st\"), val=tensor<int32, [2]>([%d,%d])];\n", stride_h, stride_w];
    [m appendFormat:@"        tensor<int32, [4]> pd = const()[name=string(\"pd\"), val=tensor<int32, [4]>([%d,%d,%d,%d])];\n", pad_t, pad_b, pad_l, pad_r];
    [m appendFormat:@"        tensor<int32, [2]> dl = const()[name=string(\"dl\"), val=tensor<int32, [2]>([%d,%d])];\n", dil_h, dil_w];
    [m appendFormat:@"        int32 gr = const()[name=string(\"gr\"), val=int32(1)];\n"];
    
    [m appendFormat:@"        tensor<fp16, [1, %d, %d, %d]> y = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W,x=x);\n", 
                    out_ch, (in_h + pad_t + pad_b - k_h) / stride_h + 1, (in_w + pad_l + pad_r - k_w) / stride_w + 1];
    [m appendString:@"    } -> (y);\n}\n"];
    return m;
}

/**
 * 2D Max Pooling
 */
static NSString *anesdk_gen_maxpool2d_fwd(int ch, int in_h, int in_w, int k_h, int k_w, int stride_h, int stride_w) {
    NSMutableString *m = [NSMutableString string];
    [m appendString:ANESDK_MIL_HDR];
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, %d, %d]> x) {\n", ch, in_h, in_w];
    [m appendFormat:@"        tensor<int32, [2]> ks = const()[name=string(\"ks\"), val=tensor<int32, [2]>([%d,%d])];\n", k_h, k_w];
    [m appendFormat:@"        tensor<int32, [2]> st = const()[name=string(\"st\"), val=tensor<int32, [2]>([%d,%d])];\n", stride_h, stride_w];
    [m appendFormat:@"        tensor<int32, [4]> pd = const()[name=string(\"pd\"), val=tensor<int32, [4]>([0,0,0,0])];\n"];
    [m appendString:@"        string pt = const()[name=string(\"pt\"), val=string(\"valid\")];\n"];
    [m appendFormat:@"        tensor<fp16, [1, %d, %d, %d]> y = max_pool(kernel_sizes=ks, pad=pd, pad_type=pt, strides=st, x=x);\n",
                    ch, (in_h - k_h) / stride_h + 1, (in_w - k_w) / stride_w + 1];
    [m appendString:@"    } -> (y);\n}\n"];
    return m;
}
