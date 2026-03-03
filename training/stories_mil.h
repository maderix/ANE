// stories_mil.h — MIL program generators for ANE kernels (Weights-as-Tensors version)
#pragma once
#include "stories_io.h"

#define MIL_HDR \
    @"program(1.3)\n[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, " \
    "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, " \
    "{\"coremltools-version\", \"9.0\"}})]\n{\n"
#define CONV_CONST \
    "        string pt = const()[name=string(\"pt\"), val=string(\"valid\")];\n" \
    "        tensor<int32, [2]> st = const()[name=string(\"st\"), val=tensor<int32, [2]>([1,1])];\n" \
    "        tensor<int32, [4]> pd = const()[name=string(\"pd\"), val=tensor<int32, [4]>([0,0,0,0])];\n" \
    "        tensor<int32, [2]> dl = const()[name=string(\"dl\"), val=tensor<int32, [2]>([1,1])];\n" \
    "        int32 gr = const()[name=string(\"gr\"), val=int32(1)];\n"

// SDPA forward flex: x, rw, Wq, Wk, Wv, Wo, cm
static NSString *gen_sdpa_fwd_flex(void) {
    float sc = 1.0f/sqrtf((float)HD);
    float invd = 1.0f/(float)DIM;
    NSMutableString *m = [NSMutableString string];
    [m appendString:MIL_HDR];
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x, "
                     "tensor<fp16, [1,%d,1,1]> rw, "
                     "tensor<fp16, [%d,%d,1,1]> Wq, "
                     "tensor<fp16, [%d,%d,1,1]> Wk, "
                     "tensor<fp16, [%d,%d,1,1]> Wv, "
                     "tensor<fp16, [%d,%d,1,1]> Wo, "
                     "tensor<fp16, [1,1,%d,%d]> cm) {\n", 
                     DIM, SEQ, DIM, DIM, DIM, DIM, DIM, DIM, DIM, DIM, DIM, SEQ, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> sq = mul(x=x,y=x);\n", DIM, SEQ];
    [m appendString:@"        tensor<int32, [1]> rax = const()[name=string(\"rax\"), val=tensor<int32, [1]>([1])];\n"];
    [m appendString:@"        bool kd = const()[name=string(\"kd\"), val=bool(true)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> ss = reduce_sum(x=sq,axes=rax,keep_dims=kd);\n", SEQ];
    [m appendFormat:@"        fp16 invd = const()[name=string(\"invd\"), val=fp16(%f)];\n", invd];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> ss2 = mul(x=ss,y=invd);\n", SEQ];
    [m appendFormat:@"        fp16 eps = const()[name=string(\"eps\"), val=fp16(0.00001)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> ss3 = add(x=ss2,y=eps);\n", SEQ];
    [m appendFormat:@"        fp16 nhalf = const()[name=string(\"nhalf\"), val=fp16(-0.5)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> rrms = pow(x=ss3,y=nhalf);\n", SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> xr = mul(x=x,y=rrms);\n", DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> xn = mul(x=xr,y=rw);\n", DIM, SEQ];
    [m appendString:@CONV_CONST];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> qf = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wq,x=xn);\n", DIM,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> kf = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wk,x=xn);\n", DIM,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> vf = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wv,x=xn);\n", DIM,SEQ];
    [m appendFormat:@"        tensor<int32, [4]> qsh = const()[name=string(\"qsh\"), val=tensor<int32, [4]>([1,%d,%d,%d])];\n", HEADS,HD,SEQ];
    [m appendString:@"        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> q4 = reshape(shape=qsh,x=qf);\n", HEADS,HD,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> q = transpose(perm=pm,x=q4);\n", HEADS,SEQ,HD];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> k4 = reshape(shape=qsh,x=kf);\n", HEADS,HD,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> k = transpose(perm=pm,x=k4);\n", HEADS,SEQ,HD];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> v4 = reshape(shape=qsh,x=vf);\n", HEADS,HD,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> v = transpose(perm=pm,x=v4);\n", HEADS,SEQ,HD];
    [m appendString:@"        bool tx = const()[name=string(\"tx\"), val=bool(false)];\n"];
    [m appendString:@"        bool ty = const()[name=string(\"ty\"), val=bool(true)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> sc1 = matmul(transpose_x=tx,transpose_y=ty,x=q,y=k);\n", HEADS,SEQ,SEQ];
    [m appendFormat:@"        fp16 scv = const()[name=string(\"scv\"), val=fp16(%f)];\n", sc];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> sc2 = mul(x=sc1,y=scv);\n", HEADS,SEQ,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> ms = add(x=sc2,y=cm);\n", HEADS,SEQ,SEQ];
    [m appendString:@"        int32 sax = const()[name=string(\"sax\"), val=int32(-1)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> aw = softmax(axis=sax,x=ms);\n", HEADS,SEQ,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> a4 = matmul(transpose_x=tx,transpose_y=tx,x=aw,y=v);\n", HEADS,SEQ,HD];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> at = transpose(perm=pm,x=a4);\n", HEADS,HD,SEQ];
    [m appendFormat:@"        tensor<int32, [4]> os = const()[name=string(\"os\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", DIM,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> af = reshape(shape=os,x=at);\n", DIM,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> oo = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wo,x=af);\n", DIM,SEQ];
    [m appendString:@"        int32 cax = const()[name=string(\"cax\"), val=int32(1)];\n"];
    [m appendString:@"        bool cid = const()[name=string(\"cid\"), val=bool(false)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> out = concat(axis=cax,interleave=cid,values=(oo,qf,kf,vf,af,xn));\n", 6*DIM,SEQ];
    [m appendString:@"    } -> (out);\n}\n"];
    return m;
}

// FFN forward flex: x, rw, W1, W2, W3
static NSString *gen_ffn_fwd_flex(void) {
    float invd = 1.0f/(float)DIM;
    NSMutableString *m = [NSMutableString string];
    [m appendString:MIL_HDR];
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x, "
                     "tensor<fp16, [1,%d,1,1]> rw, "
                     "tensor<fp16, [%d,%d,1,1]> W1, "
                     "tensor<fp16, [%d,%d,1,1]> W2, "
                     "tensor<fp16, [%d,%d,1,1]> W3) {\n", 
                     DIM, SEQ, DIM, HIDDEN, DIM, DIM, HIDDEN, HIDDEN, DIM];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> sq = mul(x=x,y=x);\n", DIM, SEQ];
    [m appendString:@"        tensor<int32, [1]> rax = const()[name=string(\"rax\"), val=tensor<int32, [1]>([1])];\n"];
    [m appendString:@"        bool kd = const()[name=string(\"kd\"), val=bool(true)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> ss = reduce_sum(x=sq,axes=rax,keep_dims=kd);\n", SEQ];
    [m appendFormat:@"        fp16 invd = const()[name=string(\"invd\"), val=fp16(%f)];\n", invd];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> ss2 = mul(x=ss,y=invd);\n", SEQ];
    [m appendFormat:@"        fp16 eps = const()[name=string(\"eps\"), val=fp16(0.00001)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> ss3 = add(x=ss2,y=eps);\n", SEQ];
    [m appendFormat:@"        fp16 nhalf = const()[name=string(\"nhalf\"), val=fp16(-0.5)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> rrms = pow(x=ss3,y=nhalf);\n", SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> xr = mul(x=x,y=rrms);\n", DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> xn = mul(x=xr,y=rw);\n", DIM, SEQ];
    [m appendString:@CONV_CONST];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> h1 = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W1,x=xn);\n", HIDDEN,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> h3 = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W3,x=xn);\n", HIDDEN,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> sig = sigmoid(x=h1);\n", HIDDEN,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> silu = mul(x=h1,y=sig);\n", HIDDEN,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> gate = mul(x=silu,y=h3);\n", HIDDEN,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> y = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W2,x=gate);\n", DIM,SEQ];
    [m appendString:@"        int32 cax = const()[name=string(\"cax\"), val=int32(1)];\n"];
    [m appendString:@"        bool cid = const()[name=string(\"cid\"), val=bool(false)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> out = concat(axis=cax,interleave=cid,values=(y,h1,h3,gate,xn));\n", 2*DIM+3*HIDDEN,SEQ];
    [m appendString:@"    } -> (out);\n}\n"];
    return m;
}

// FFN backward flex: x, W1t, W2t, W3t
static NSString *gen_ffn_bwd_flex(void) {
    NSMutableString *m = [NSMutableString string];
    [m appendString:MIL_HDR];
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x, "
                     "tensor<fp16, [%d,%d,1,1]> W1t, "
                     "tensor<fp16, [%d,%d,1,1]> W2t, "
                     "tensor<fp16, [%d,%d,1,1]> W3t) {\n", 
                     DIM+2*HIDDEN, SEQ, DIM, HIDDEN, HIDDEN, DIM, DIM, HIDDEN];
    [m appendString:@CONV_CONST];
    [m appendString:@"        tensor<int32, [4]> bd = const()[name=string(\"bd\"), val=tensor<int32, [4]>([0,0,0,0])];\n"];
    [m appendFormat:@"        tensor<int32, [4]> sd = const()[name=string(\"sd\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dffn = slice_by_size(x=x,begin=bd,size=sd);\n", DIM, SEQ];
    [m appendFormat:@"        tensor<int32, [4]> b1 = const()[name=string(\"b1\"), val=tensor<int32, [4]>([0,%d,0,0])];\n", DIM];
    [m appendFormat:@"        tensor<int32, [4]> s1 = const()[name=string(\"s1\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", HIDDEN, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> h1 = slice_by_size(x=x,begin=b1,size=s1);\n", HIDDEN, SEQ];
    [m appendFormat:@"        tensor<int32, [4]> b3 = const()[name=string(\"b3\"), val=tensor<int32, [4]>([0,%d,0,0])];\n", DIM+HIDDEN];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> h3 = slice_by_size(x=x,begin=b3,size=s1);\n", HIDDEN, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dsilu = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W2t,x=dffn);\n", HIDDEN, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> sig = sigmoid(x=h1);\n", HIDDEN, SEQ];
    [m appendString:@"        fp16 one = const()[name=string(\"one\"), val=fp16(1.0)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> oms = sub(x=one,y=sig);\n", HIDDEN, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> homs = mul(x=h1,y=oms);\n", HIDDEN, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> brk = add(x=one,y=homs);\n", HIDDEN, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dsd = mul(x=sig,y=brk);\n", HIDDEN, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> t1 = mul(x=dsilu,y=h3);\n", HIDDEN, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dh1 = mul(x=t1,y=dsd);\n", HIDDEN, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> slh = mul(x=h1,y=sig);\n", HIDDEN, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dh3 = mul(x=dsilu,y=slh);\n", HIDDEN, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dx1 = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W1t,x=dh1);\n", DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dx3 = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W3t,x=dh3);\n", DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dx = add(x=dx1,y=dx3);\n", DIM, SEQ];
    [m appendString:@"        int32 cax = const()[name=string(\"cax\"), val=int32(1)];\n"];
    [m appendString:@"        bool cid = const()[name=string(\"cid\"), val=bool(false)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> out = concat(axis=cax,interleave=cid,values=(dx,dh1,dh3));\n", DIM+2*HIDDEN, SEQ];
    [m appendString:@"    } -> (out);\n}\n"];
    return m;
}

// QKV backward flex: x, Wqt, Wkt, Wvt
static NSString *gen_qkvb_flex(void) {
    NSMutableString *m = [NSMutableString string];
    [m appendString:MIL_HDR];
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x, "
                     "tensor<fp16, [%d,%d,1,1]> Wqt, "
                     "tensor<fp16, [%d,%d,1,1]> Wkt, "
                     "tensor<fp16, [%d,%d,1,1]> Wvt) {\n", 
                     3*DIM, SEQ, DIM, DIM, DIM, DIM, DIM, DIM];
    [m appendString:@CONV_CONST];
    [m appendFormat:@"        tensor<int32, [4]> sz = const()[name=string(\"sz\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", DIM, SEQ];
    [m appendString:@"        tensor<int32, [4]> b0 = const()[name=string(\"b0\"), val=tensor<int32, [4]>([0,0,0,0])];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dq = slice_by_size(x=x,begin=b0,size=sz);\n", DIM,SEQ];
    [m appendFormat:@"        tensor<int32, [4]> b1 = const()[name=string(\"b1\"), val=tensor<int32, [4]>([0,%d,0,0])];\n", DIM];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dk = slice_by_size(x=x,begin=b1,size=sz);\n", DIM,SEQ];
    [m appendFormat:@"        tensor<int32, [4]> b2 = const()[name=string(\"b2\"), val=tensor<int32, [4]>([0,%d,0,0])];\n", 2*DIM];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dv = slice_by_size(x=x,begin=b2,size=sz);\n", DIM,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dxq = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wqt,x=dq);\n", DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dxk = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wkt,x=dk);\n", DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dxv = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wvt,x=dv);\n", DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dxqk = add(x=dxq,y=dxk);\n", DIM,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> out = add(x=dxqk,y=dxv);\n", DIM,SEQ];
    [m appendString:@"    } -> (out);\n}\n"];
    return m;
}

// SDPA backward part 1 flex: x, Wot, cm
static NSString *gen_sdpa_bwd1_flex(void) {
    float sc = 1.0f/sqrtf((float)HD);
    NSMutableString *m = [NSMutableString string];
    [m appendString:MIL_HDR];
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x, "
                     "tensor<fp16, [%d,%d,1,1]> Wot, "
                     "tensor<fp16, [1,1,%d,%d]> cm) {\n", 
                     4*DIM, SEQ, DIM, DIM, SEQ, SEQ];
    [m appendString:@CONV_CONST];
    [m appendFormat:@"        tensor<int32, [4]> sz = const()[name=string(\"sz\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", DIM, SEQ];
    [m appendString:@"        tensor<int32, [4]> b0 = const()[name=string(\"b0\"), val=tensor<int32, [4]>([0,0,0,0])];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> qf = slice_by_size(x=x,begin=b0,size=sz);\n", DIM,SEQ];
    [m appendFormat:@"        tensor<int32, [4]> b1 = const()[name=string(\"b1\"), val=tensor<int32, [4]>([0,%d,0,0])];\n", DIM];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> kf = slice_by_size(x=x,begin=b1,size=sz);\n", DIM,SEQ];
    [m appendFormat:@"        tensor<int32, [4]> b2 = const()[name=string(\"b2\"), val=tensor<int32, [4]>([0,%d,0,0])];\n", 2*DIM];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> vf = slice_by_size(x=x,begin=b2,size=sz);\n", DIM,SEQ];
    [m appendFormat:@"        tensor<int32, [4]> b3 = const()[name=string(\"b3\"), val=tensor<int32, [4]>([0,%d,0,0])];\n", 3*DIM];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dx2f = slice_by_size(x=x,begin=b3,size=sz);\n", DIM,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> df = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wot,x=dx2f);\n", DIM, SEQ];
    [m appendFormat:@"        tensor<int32, [4]> rsh = const()[name=string(\"rsh\"), val=tensor<int32, [4]>([1,%d,%d,%d])];\n", HEADS,HD,SEQ];
    [m appendString:@"        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> qr = reshape(shape=rsh,x=qf);\n", HEADS,HD,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> q = transpose(perm=pm,x=qr);\n", HEADS,SEQ,HD];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> kr = reshape(shape=rsh,x=kf);\n", HEADS,HD,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> k = transpose(perm=pm,x=kr);\n", HEADS,SEQ,HD];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> vr = reshape(shape=rsh,x=vf);\n", HEADS,HD,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> v = transpose(perm=pm,x=vr);\n", HEADS,SEQ,HD];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> dr = reshape(shape=rsh,x=df);\n", HEADS,HD,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> da = transpose(perm=pm,x=dr);\n", HEADS,SEQ,HD];
    [m appendString:@"        bool bF = const()[name=string(\"bF\"), val=bool(false)];\n"];
    [m appendString:@"        bool bT = const()[name=string(\"bT\"), val=bool(true)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> sc1 = matmul(transpose_x=bF,transpose_y=bT,x=q,y=k);\n", HEADS,SEQ,SEQ];
    [m appendFormat:@"        fp16 scv = const()[name=string(\"scv\"), val=fp16(%f)];\n", sc];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> sc2 = mul(x=sc1,y=scv);\n", HEADS,SEQ,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> ms = add(x=sc2,y=cm);\n", HEADS,SEQ,SEQ];
    [m appendString:@"        int32 sax = const()[name=string(\"sax\"), val=int32(-1)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> probs = softmax(axis=sax,x=ms);\n", HEADS,SEQ,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> dv4 = matmul(transpose_x=bT,transpose_y=bF,x=probs,y=da);\n", HEADS,SEQ,HD];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> dp4 = matmul(transpose_x=bF,transpose_y=bT,x=da,y=v);\n", HEADS,SEQ,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> dvt = transpose(perm=pm,x=dv4);\n", HEADS,HD,SEQ];
    [m appendFormat:@"        tensor<int32, [4]> dvs = const()[name=string(\"dvs\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", DIM,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dvf = reshape(shape=dvs,x=dvt);\n", DIM,SEQ];
    [m appendFormat:@"        tensor<int32, [4]> scs = const()[name=string(\"scs\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", SCORE_CH,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> pf = reshape(shape=scs,x=probs);\n", SCORE_CH,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dpf = reshape(shape=scs,x=dp4);\n", SCORE_CH,SEQ];
    [m appendString:@"        int32 cax = const()[name=string(\"cax\"), val=int32(1)];\n"];
    [m appendString:@"        bool cid = const()[name=string(\"cid\"), val=bool(false)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> out = concat(axis=cax,interleave=cid,values=(dvf,pf,dpf));\n", DIM+2*SCORE_CH,SEQ];
    [m appendString:@"    } -> (out);\n}\n"];
    return m;
}

// SDPA backward part 2 (no weights, stays the same but renamed)
static NSString *gen_sdpa_bwd2_flex(void) {
    float sc = 1.0f/sqrtf((float)HD);
    int bwd2_in = 2*SCORE_CH + 2*DIM;
    NSMutableString *m = [NSMutableString string];
    [m appendString:MIL_HDR];
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n", bwd2_in, SEQ];
    [m appendFormat:@"        tensor<int32, [4]> sz_sc = const()[name=string(\"szsc\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", SCORE_CH, SEQ];
    [m appendString:@"        tensor<int32, [4]> b0 = const()[name=string(\"b0\"), val=tensor<int32, [4]>([0,0,0,0])];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> pf = slice_by_size(x=x,begin=b0,size=sz_sc);\n", SCORE_CH,SEQ];
    [m appendFormat:@"        tensor<int32, [4]> b1 = const()[name=string(\"b1\"), val=tensor<int32, [4]>([0,%d,0,0])];\n", SCORE_CH];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dpf = slice_by_size(x=x,begin=b1,size=sz_sc);\n", SCORE_CH,SEQ];
    [m appendFormat:@"        tensor<int32, [4]> sz_d = const()[name=string(\"szd\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<int32, [4]> b2 = const()[name=string(\"b2\"), val=tensor<int32, [4]>([0,%d,0,0])];\n", 2*SCORE_CH];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> qf = slice_by_size(x=x,begin=b2,size=sz_d);\n", DIM,SEQ];
    [m appendFormat:@"        tensor<int32, [4]> b3 = const()[name=string(\"b3\"), val=tensor<int32, [4]>([0,%d,0,0])];\n", 2*SCORE_CH+DIM];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> kf = slice_by_size(x=x,begin=b3,size=sz_d);\n", DIM,SEQ];
    [m appendFormat:@"        tensor<int32, [4]> ssh = const()[name=string(\"ssh\"), val=tensor<int32, [4]>([1,%d,%d,%d])];\n", HEADS,SEQ,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> probs = reshape(shape=ssh,x=pf);\n", HEADS,SEQ,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> dp = reshape(shape=ssh,x=dpf);\n", HEADS,SEQ,SEQ];
    [m appendFormat:@"        tensor<int32, [4]> rsh = const()[name=string(\"rsh\"), val=tensor<int32, [4]>([1,%d,%d,%d])];\n", HEADS,HD,SEQ];
    [m appendString:@"        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> qr = reshape(shape=rsh,x=qf);\n", HEADS,HD,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> q = transpose(perm=pm,x=qr);\n", HEADS,SEQ,HD];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> kr = reshape(shape=rsh,x=kf);\n", HEADS,HD,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> k = transpose(perm=pm,x=kr);\n", HEADS,SEQ,HD];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> pdp = mul(x=probs,y=dp);\n", HEADS,SEQ,SEQ];
    [m appendString:@"        tensor<int32, [1]> rax = const()[name=string(\"rax\"), val=tensor<int32, [1]>([-1])];\n"];
    [m appendString:@"        bool kd = const()[name=string(\"kd\"), val=bool(true)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,1]> spdp = reduce_sum(x=pdp,axes=rax,keep_dims=kd);\n", HEADS,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> dps = sub(x=dp,y=spdp);\n", HEADS,SEQ,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> ds0 = mul(x=probs,y=dps);\n", HEADS,SEQ,SEQ];
    [m appendFormat:@"        fp16 scv = const()[name=string(\"scv\"), val=fp16(%f)];\n", sc];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> ds = mul(x=ds0,y=scv);\n", HEADS,SEQ,SEQ];
    [m appendString:@"        bool bF = const()[name=string(\"bF\"), val=bool(false)];\n"];
    [m appendString:@"        bool bT = const()[name=string(\"bT\"), val=bool(true)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> dq4 = matmul(transpose_x=bF,transpose_y=bF,x=ds,y=k);\n", HEADS,SEQ,HD];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> dk4 = matmul(transpose_x=bT,transpose_y=bF,x=ds,y=q);\n", HEADS,SEQ,HD];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> dqt = transpose(perm=pm,x=dq4);\n", HEADS,HD,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> dkt = transpose(perm=pm,x=dk4);\n", HEADS,HD,SEQ];
    [m appendFormat:@"        tensor<int32, [4]> fs = const()[name=string(\"fs\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", DIM,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dqf = reshape(shape=fs,x=dqt);\n", DIM,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dkf = reshape(shape=fs,x=dkt);\n", DIM,SEQ];
    [m appendString:@"        int32 cax = const()[name=string(\"cax\"), val=int32(1)];\n"];
    [m appendString:@"        bool cid = const()[name=string(\"cid\"), val=bool(false)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> out = concat(axis=cax,interleave=cid,values=(dqf,dkf));\n", 2*DIM,SEQ];
    [m appendString:@"    } -> (out);\n}\n"];
    return m;
}

// Mask blob helper
static NSData *get_mask_blob(void) {
    _Float16 *mask = (_Float16*)calloc(SEQ*SEQ, sizeof(_Float16));
    for(int t=0;t<SEQ;t++) for(int t2=0;t2<SEQ;t2++)
        mask[t*SEQ+t2] = (t2<=t) ? (_Float16)0.0f : (_Float16)(-65504.0f);
    NSData *d = build_blob_fp16(mask, SEQ*SEQ);
    free(mask);
    return d;
}
