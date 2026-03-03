// stories_mil.h — MIL program generators for ANE kernels
// Same architecture as single-layer train_large.m but parameterized
#pragma once
#include "stories_io.h"

#define MIL_HDR \
    @"program(1.0)\n[buildInfo = dict<tensor<string, []>, tensor<string, []>>({{\"coremlc-version\", \"3505.4.1\"}})]\n{\n"
#define CONV_CONST \
    "        tensor<string, []> pt = const()[name=tensor<string, []>(\"pt\"), val=tensor<string, []>(\"valid\")];\n" \
    "        tensor<int32, [2]> st = const()[name=tensor<string, []>(\"st\"), val=tensor<int32, [2]>([1,1])];\n" \
    "        tensor<int32, [4]> pd = const()[name=tensor<string, []>(\"pd\"), val=tensor<int32, [4]>([0,0,0,0])];\n" \
    "        tensor<int32, [2]> dl = const()[name=tensor<string, []>(\"dl\"), val=tensor<int32, [2]>([1,1])];\n" \
    "        tensor<int32, []> gr = const()[name=tensor<string, []>(\"gr\"), val=tensor<int32, []>(1)];\n"

// SDPA forward + taps: x_in → rmsnorm → QKV+SDPA+Wo → concat(o_out, Q, K, V, attn_out, xnorm)
static NSString *gen_sdpa_fwd_taps(void) {
    float sc = 1.0f/sqrtf((float)HD);
    float invd = 1.0f/(float)DIM;
    NSMutableString *m = [NSMutableString string];
    [m appendString:MIL_HDR];
    [m appendFormat:@"    func main<ios16>(tensor<fp16, [1, %d, 1, %d]> x) {\n", DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> sq = mul(x=x,y=x)[name=tensor<string, []>(\"sq\")];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<int32, [1]> rax = const()[name=tensor<string, []>(\"rax\"), val=tensor<int32, [1]>([1])];\n"];
    [m appendFormat:@"        tensor<bool, []> kd = const()[name=tensor<string, []>(\"kd\"), val=tensor<bool, []>(true)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> ss = reduce_sum(x=sq,axes=rax,keep_dims=kd)[name=tensor<string, []>(\"ss\")];\n", SEQ];
    [m appendFormat:@"        tensor<fp16, []> invd = const()[name=tensor<string, []>(\"invd\"), val=tensor<fp16, []>(%f)];\n", invd];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> ss2 = mul(x=ss,y=invd)[name=tensor<string, []>(\"ss2\")];\n", SEQ];
    [m appendFormat:@"        tensor<fp16, []> eps = const()[name=tensor<string, []>(\"eps\"), val=tensor<fp16, []>(0.00001)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> ss3 = add(x=ss2,y=eps)[name=tensor<string, []>(\"ss3\")];\n", SEQ];
    [m appendFormat:@"        tensor<fp16, []> nhalf = const()[name=tensor<string, []>(\"nhalf\"), val=tensor<fp16, []>(-0.5)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> rrms = pow(x=ss3,y=nhalf)[name=tensor<string, []>(\"rrms\")];\n", SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> xr = mul(x=x,y=rrms)[name=tensor<string, []>(\"xr\")];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,1]> rw = const()[name=tensor<string, []>(\"rw\"), val=tensor<fp16, [1,%d,1,1]>(BLOBFILE(path=tensor<string, []>(\"@model_path/weights/rms1.bin\"), offset=tensor<uint64, []>(64)))];\n", DIM, DIM];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> xn = mul(x=xr,y=rw)[name=tensor<string, []>(\"xn\")];\n", DIM, SEQ];
    [m appendString:@CONV_CONST];
    [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> Wq = const()[name=tensor<string, []>(\"Wq\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=tensor<string, []>(\"@model_path/weights/wq.bin\"), offset=tensor<uint64, []>(64)))];\n", DIM,DIM,DIM,DIM];
    [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> Wk = const()[name=tensor<string, []>(\"Wk\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=tensor<string, []>(\"@model_path/weights/wk.bin\"), offset=tensor<uint64, []>(64)))];\n", DIM,DIM,DIM,DIM];
    [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> Wv = const()[name=tensor<string, []>(\"Wv\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=tensor<string, []>(\"@model_path/weights/wv.bin\"), offset=tensor<uint64, []>(64)))];\n", DIM,DIM,DIM,DIM];
    [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> Wo = const()[name=tensor<string, []>(\"Wo\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=tensor<string, []>(\"@model_path/weights/wo.bin\"), offset=tensor<uint64, []>(64)))];\n", DIM,DIM,DIM,DIM];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> qf = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wq,x=xn)[name=tensor<string, []>(\"cq\")];\n", DIM,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> kf = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wk,x=xn)[name=tensor<string, []>(\"ck\")];\n", DIM,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> vf = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wv,x=xn)[name=tensor<string, []>(\"cv\")];\n", DIM,SEQ];
    [m appendFormat:@"        tensor<int32, [4]> qsh = const()[name=tensor<string, []>(\"qsh\"), val=tensor<int32, [4]>([1,%d,%d,%d])];\n", HEADS,HD,SEQ];
    [m appendString:@"        tensor<int32, [4]> pm = const()[name=tensor<string, []>(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> q4 = reshape(shape=qsh,x=qf)[name=tensor<string, []>(\"rq\")];\n", HEADS,HD,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> q = transpose(perm=pm,x=q4)[name=tensor<string, []>(\"tq\")];\n", HEADS,SEQ,HD];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> k4 = reshape(shape=qsh,x=kf)[name=tensor<string, []>(\"rk\")];\n", HEADS,HD,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> k = transpose(perm=pm,x=k4)[name=tensor<string, []>(\"tk\")];\n", HEADS,SEQ,HD];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> v4 = reshape(shape=qsh,x=vf)[name=tensor<string, []>(\"rv\")];\n", HEADS,HD,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> v = transpose(perm=pm,x=v4)[name=tensor<string, []>(\"tv\")];\n", HEADS,SEQ,HD];
    [m appendString:@"        tensor<bool, []> tx = const()[name=tensor<string, []>(\"tx\"), val=tensor<bool, []>(false)];\n"];
    [m appendString:@"        tensor<bool, []> ty = const()[name=tensor<string, []>(\"ty\"), val=tensor<bool, []>(true)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> sc1 = matmul(transpose_x=tx,transpose_y=ty,x=q,y=k)[name=tensor<string, []>(\"mm1\")];\n", HEADS,SEQ,SEQ];
    [m appendFormat:@"        tensor<fp16, []> scv = const()[name=tensor<string, []>(\"scv\"), val=tensor<fp16, []>(%f)];\n", sc];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> sc2 = mul(x=sc1,y=scv)[name=tensor<string, []>(\"scl\")];\n", HEADS,SEQ,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> cm = const()[name=tensor<string, []>(\"cm\"), val=tensor<fp16, [1,1,%d,%d]>(BLOBFILE(path=tensor<string, []>(\"@model_path/weights/mask.bin\"), offset=tensor<uint64, []>(64)))];\n", SEQ,SEQ,SEQ,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> ms = add(x=sc2,y=cm)[name=tensor<string, []>(\"msk\")];\n", HEADS,SEQ,SEQ];
    [m appendString:@"        tensor<int32, []> sax = const()[name=tensor<string, []>(\"sax\"), val=tensor<int32, []>(-1)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> aw = softmax(axis=sax,x=ms)[name=tensor<string, []>(\"sm\")];\n", HEADS,SEQ,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> a4 = matmul(transpose_x=tx,transpose_y=tx,x=aw,y=v)[name=tensor<string, []>(\"mm2\")];\n", HEADS,SEQ,HD];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> at = transpose(perm=pm,x=a4)[name=tensor<string, []>(\"ta\")];\n", HEADS,HD,SEQ];
    [m appendFormat:@"        tensor<int32, [4]> os = const()[name=tensor<string, []>(\"os\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", DIM,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> af = reshape(shape=os,x=at)[name=tensor<string, []>(\"ra\")];\n", DIM,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> oo = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wo,x=af)[name=tensor<string, []>(\"co\")];\n", DIM,SEQ];
    [m appendString:@"        tensor<int32, []> cax = const()[name=tensor<string, []>(\"cax\"), val=tensor<int32, []>(1)];\n"];
    [m appendString:@"        tensor<bool, []> cid = const()[name=tensor<string, []>(\"cid\"), val=tensor<bool, []>(false)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> out = concat(axis=cax,interleave=cid,values=(oo,qf,kf,vf,af,xn))[name=tensor<string, []>(\"cat\")];\n", 6*DIM,SEQ];
    [m appendString:@"    } -> (out);\n}\n"];
    return m;
}

// FFN forward + taps: x2 → rmsnorm → FFN → concat(ffn_out, h1, h3, silu_out, x2norm)
static NSString *gen_ffn_fwd_taps(void) {
    float invd = 1.0f/(float)DIM;
    NSMutableString *m = [NSMutableString string];
    [m appendString:MIL_HDR];
    [m appendFormat:@"    func main<ios16>(tensor<fp16, [1, %d, 1, %d]> x) {\n", DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> sq = mul(x=x,y=x)[name=tensor<string, []>(\"sq\")];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<int32, [1]> rax = const()[name=tensor<string, []>(\"rax\"), val=tensor<int32, [1]>([1])];\n"];
    [m appendFormat:@"        tensor<bool, []> kd = const()[name=tensor<string, []>(\"kd\"), val=tensor<bool, []>(true)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> ss = reduce_sum(x=sq,axes=rax,keep_dims=kd)[name=tensor<string, []>(\"ss\")];\n", SEQ];
    [m appendFormat:@"        tensor<fp16, []> invd = const()[name=tensor<string, []>(\"invd\"), val=tensor<fp16, []>(%f)];\n", invd];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> ss2 = mul(x=ss,y=invd)[name=tensor<string, []>(\"ss2\")];\n", SEQ];
    [m appendFormat:@"        tensor<fp16, []> eps = const()[name=tensor<string, []>(\"eps\"), val=tensor<fp16, []>(0.00001)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> ss3 = add(x=ss2,y=eps)[name=tensor<string, []>(\"ss3\")];\n", SEQ];
    [m appendFormat:@"        tensor<fp16, []> nhalf = const()[name=tensor<string, []>(\"nhalf\"), val=tensor<fp16, []>(-0.5)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> rrms = pow(x=ss3,y=nhalf)[name=tensor<string, []>(\"rrms\")];\n", SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> xr = mul(x=x,y=rrms)[name=tensor<string, []>(\"xr\")];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,1]> rw = const()[name=tensor<string, []>(\"rw\"), val=tensor<fp16, [1,%d,1,1]>(BLOBFILE(path=tensor<string, []>(\"@model_path/weights/rms2.bin\"), offset=tensor<uint64, []>(64)))];\n", DIM, DIM];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> xn = mul(x=xr,y=rw)[name=tensor<string, []>(\"xn\")];\n", DIM, SEQ];
    [m appendString:@CONV_CONST];
    [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> W1 = const()[name=tensor<string, []>(\"W1\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=tensor<string, []>(\"@model_path/weights/w1.bin\"), offset=tensor<uint64, []>(64)))];\n", HIDDEN,DIM,HIDDEN,DIM];
    [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> W3 = const()[name=tensor<string, []>(\"W3\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=tensor<string, []>(\"@model_path/weights/w3.bin\"), offset=tensor<uint64, []>(64)))];\n", HIDDEN,DIM,HIDDEN,DIM];
    [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> W2 = const()[name=tensor<string, []>(\"W2\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=tensor<string, []>(\"@model_path/weights/w2.bin\"), offset=tensor<uint64, []>(64)))];\n", DIM,HIDDEN,DIM,HIDDEN];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> h1 = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W1,x=xn)[name=tensor<string, []>(\"c1\")];\n", HIDDEN,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> h3 = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W3,x=xn)[name=tensor<string, []>(\"c3\")];\n", HIDDEN,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> sig = sigmoid(x=h1)[name=tensor<string, []>(\"sg\")];\n", HIDDEN,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> silu = mul(x=h1,y=sig)[name=tensor<string, []>(\"si\")];\n", HIDDEN,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> gate = mul(x=silu,y=h3)[name=tensor<string, []>(\"gt\")];\n", HIDDEN,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> y = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W2,x=gate)[name=tensor<string, []>(\"c2\")];\n", DIM,SEQ];
    [m appendString:@"        tensor<int32, []> cax = const()[name=tensor<string, []>(\"cax\"), val=tensor<int32, []>(1)];\n"];
    [m appendString:@"        tensor<bool, []> cid = const()[name=tensor<string, []>(\"cid\"), val=tensor<bool, []>(false)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> out = concat(axis=cax,interleave=cid,values=(y,h1,h3,gate,xn))[name=tensor<string, []>(\"cat\")];\n", 2*DIM+3*HIDDEN,SEQ];
    [m appendString:@"    } -> (out);\n}\n"];
    return m;
}

// FFN backward: concat(dffn,h1,h3) → concat(dx,dh1,dh3)
static NSString *gen_ffn_bwd(void) {
    NSMutableString *m = [NSMutableString string];
    [m appendString:MIL_HDR];
    [m appendFormat:@"    func main<ios16>(tensor<fp16, [1, %d, 1, %d]> x) {\n", DIM+2*HIDDEN, SEQ];
    [m appendString:@CONV_CONST];
    [m appendString:@"        tensor<int32, [4]> bd = const()[name=tensor<string, []>(\"bd\"), val=tensor<int32, [4]>([0,0,0,0])];\n"];
    [m appendFormat:@"        tensor<int32, [4]> sd = const()[name=tensor<string, []>(\"sd\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dffn = slice_by_size(x=x,begin=bd,size=sd)[name=tensor<string, []>(\"s0\")];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<int32, [4]> b1 = const()[name=tensor<string, []>(\"b1\"), val=tensor<int32, [4]>([0,%d,0,0])];\n", DIM];
    [m appendFormat:@"        tensor<int32, [4]> s1 = const()[name=tensor<string, []>(\"s1\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", HIDDEN, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> h1 = slice_by_size(x=x,begin=b1,size=s1)[name=tensor<string, []>(\"s1x\")];\n", HIDDEN, SEQ];
    [m appendFormat:@"        tensor<int32, [4]> b3 = const()[name=tensor<string, []>(\"b3\"), val=tensor<int32, [4]>([0,%d,0,0])];\n", DIM+HIDDEN];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> h3 = slice_by_size(x=x,begin=b3,size=s1)[name=tensor<string, []>(\"s3x\")];\n", HIDDEN, SEQ];
    [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> W2t = const()[name=tensor<string, []>(\"W2t\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=tensor<string, []>(\"@model_path/weights/w2t.bin\"), offset=tensor<uint64, []>(64)))];\n", HIDDEN, DIM, HIDDEN, DIM];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dsilu = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W2t,x=dffn)[name=tensor<string, []>(\"cw2\")];\n", HIDDEN, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> sig = sigmoid(x=h1)[name=tensor<string, []>(\"sg\")];\n", HIDDEN, SEQ];
    [m appendString:@"        tensor<fp16, []> one = const()[name=tensor<string, []>(\"one\"), val=tensor<fp16, []>(1.0)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> oms = sub(x=one,y=sig)[name=tensor<string, []>(\"oms\")];\n", HIDDEN, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> homs = mul(x=h1,y=oms)[name=tensor<string, []>(\"homs\")];\n", HIDDEN, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> brk = add(x=one,y=homs)[name=tensor<string, []>(\"brk\")];\n", HIDDEN, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dsd = mul(x=sig,y=brk)[name=tensor<string, []>(\"dsd\")];\n", HIDDEN, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> t1 = mul(x=dsilu,y=h3)[name=tensor<string, []>(\"t1\")];\n", HIDDEN, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dh1 = mul(x=t1,y=dsd)[name=tensor<string, []>(\"dh1\")];\n", HIDDEN, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> slh = mul(x=h1,y=sig)[name=tensor<string, []>(\"slh\")];\n", HIDDEN, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dh3 = mul(x=dsilu,y=slh)[name=tensor<string, []>(\"dh3\")];\n", HIDDEN, SEQ];
    [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> W1t = const()[name=tensor<string, []>(\"W1t\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=tensor<string, []>(\"@model_path/weights/w1t.bin\"), offset=tensor<uint64, []>(64)))];\n", DIM, HIDDEN, DIM, HIDDEN];
    [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> W3t = const()[name=tensor<string, []>(\"W3t\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=tensor<string, []>(\"@model_path/weights/w3t.bin\"), offset=tensor<uint64, []>(64)))];\n", DIM, HIDDEN, DIM, HIDDEN];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dx1 = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W1t,x=dh1)[name=tensor<string, []>(\"cw1\")];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dx3 = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W3t,x=dh3)[name=tensor<string, []>(\"cw3\")];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dx = add(x=dx1,y=dx3)[name=tensor<string, []>(\"adx\")];\n", DIM, SEQ];
    [m appendString:@"        tensor<int32, []> cax = const()[name=tensor<string, []>(\"cax\"), val=tensor<int32, []>(1)];\n"];
    [m appendString:@"        tensor<bool, []> cid = const()[name=tensor<string, []>(\"cid\"), val=tensor<bool, []>(false)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> out = concat(axis=cax,interleave=cid,values=(dx,dh1,dh3))[name=tensor<string, []>(\"cat\")];\n", DIM+2*HIDDEN, SEQ];
    [m appendString:@"    } -> (out);\n}\n"];
    return m;
}

// QKV backward: concat(dq,dk,dv) → dx
static NSString *gen_qkvb(void) {
    NSMutableString *m = [NSMutableString string];
    [m appendString:MIL_HDR];
    [m appendFormat:@"    func main<ios16>(tensor<fp16, [1, %d, 1, %d]> x) {\n", 3*DIM, SEQ];
    [m appendString:@CONV_CONST];
    [m appendFormat:@"        tensor<int32, [4]> sz = const()[name=tensor<string, []>(\"sz\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", DIM, SEQ];
    [m appendString:@"        tensor<int32, [4]> b0 = const()[name=tensor<string, []>(\"b0\"), val=tensor<int32, [4]>([0,0,0,0])];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dq = slice_by_size(x=x,begin=b0,size=sz)[name=tensor<string, []>(\"s0\")];\n", DIM,SEQ];
    [m appendFormat:@"        tensor<int32, [4]> b1 = const()[name=tensor<string, []>(\"b1\"), val=tensor<int32, [4]>([0,%d,0,0])];\n", DIM];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dk = slice_by_size(x=x,begin=b1,size=sz)[name=tensor<string, []>(\"s1\")];\n", DIM,SEQ];
    [m appendFormat:@"        tensor<int32, [4]> b2 = const()[name=tensor<string, []>(\"b2\"), val=tensor<int32, [4]>([0,%d,0,0])];\n", 2*DIM];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dv = slice_by_size(x=x,begin=b2,size=sz)[name=tensor<string, []>(\"s2\")];\n", DIM,SEQ];
    [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> Wqt = const()[name=tensor<string, []>(\"Wqt\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=tensor<string, []>(\"@model_path/weights/wqt.bin\"), offset=tensor<uint64, []>(64)))];\n", DIM,DIM,DIM,DIM];
    [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> Wkt = const()[name=tensor<string, []>(\"Wkt\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=tensor<string, []>(\"@model_path/weights/wkt.bin\"), offset=tensor<uint64, []>(64)))];\n", DIM,DIM,DIM,DIM];
    [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> Wvt = const()[name=tensor<string, []>(\"Wvt\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=tensor<string, []>(\"@model_path/weights/wvt.bin\"), offset=tensor<uint64, []>(64)))];\n", DIM,DIM,DIM,DIM];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dxq = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wqt,x=dq)[name=tensor<string, []>(\"cq\")];\n", DIM,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dxk = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wkt,x=dk)[name=tensor<string, []>(\"ck\")];\n", DIM,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dxv = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wvt,x=dv)[name=tensor<string, []>(\"cv\")];\n", DIM,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dxqk = add(x=dxq,y=dxk)[name=tensor<string, []>(\"aqk\")];\n", DIM,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> out = add(x=dxqk,y=dxv)[name=tensor<string, []>(\"out\")];\n", DIM,SEQ];
    [m appendString:@"    } -> (out);\n}\n"];
    return m;
}

// SDPA backward part 1 + Wo^T
static NSString *gen_sdpa_bwd1(void) {
    float sc = 1.0f/sqrtf((float)HD);
    NSMutableString *m = [NSMutableString string];
    [m appendString:MIL_HDR];
    [m appendFormat:@"    func main<ios16>(tensor<fp16, [1, %d, 1, %d]> x) {\n", 4*DIM, SEQ];
    [m appendString:@CONV_CONST];
    [m appendFormat:@"        tensor<int32, [4]> sz = const()[name=tensor<string, []>(\"sz\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", DIM, SEQ];
    [m appendString:@"        tensor<int32, [4]> b0 = const()[name=tensor<string, []>(\"b0\"), val=tensor<int32, [4]>([0,0,0,0])];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> qf = slice_by_size(x=x,begin=b0,size=sz)[name=tensor<string, []>(\"s0\")];\n", DIM,SEQ];
    [m appendFormat:@"        tensor<int32, [4]> b1 = const()[name=tensor<string, []>(\"b1\"), val=tensor<int32, [4]>([0,%d,0,0])];\n", DIM];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> kf = slice_by_size(x=x,begin=b1,size=sz)[name=tensor<string, []>(\"s1\")];\n", DIM,SEQ];
    [m appendFormat:@"        tensor<int32, [4]> b2 = const()[name=tensor<string, []>(\"b2\"), val=tensor<int32, [4]>([0,%d,0,0])];\n", 2*DIM];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> vf = slice_by_size(x=x,begin=b2,size=sz)[name=tensor<string, []>(\"s2\")];\n", DIM,SEQ];
    [m appendFormat:@"        tensor<int32, [4]> b3 = const()[name=tensor<string, []>(\"b3\"), val=tensor<int32, [4]>([0,%d,0,0])];\n", 3*DIM];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dx2f = slice_by_size(x=x,begin=b3,size=sz)[name=tensor<string, []>(\"s3\")];\n", DIM,SEQ];
    [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> Wot = const()[name=tensor<string, []>(\"Wot\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=tensor<string, []>(\"@model_path/weights/wot.bin\"), offset=tensor<uint64, []>(64)))];\n", DIM,DIM,DIM,DIM];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> df = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wot,x=dx2f)[name=tensor<string, []>(\"cwo\")];\n", DIM,SEQ];
    [m appendFormat:@"        tensor<int32, [4]> rsh = const()[name=tensor<string, []>(\"rsh\"), val=tensor<int32, [4]>([1,%d,%d,%d])];\n", HEADS,HD,SEQ];
    [m appendString:@"        tensor<int32, [4]> pm = const()[name=tensor<string, []>(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> qr = reshape(shape=rsh,x=qf)[name=tensor<string, []>(\"rq\")];\n", HEADS,HD,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> q = transpose(perm=pm,x=qr)[name=tensor<string, []>(\"tq\")];\n", HEADS,SEQ,HD];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> kr = reshape(shape=rsh,x=kf)[name=tensor<string, []>(\"rk\")];\n", HEADS,HD,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> k = transpose(perm=pm,x=kr)[name=tensor<string, []>(\"tk\")];\n", HEADS,SEQ,HD];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> vr = reshape(shape=rsh,x=vf)[name=tensor<string, []>(\"rv\")];\n", HEADS,HD,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> v = transpose(perm=pm,x=vr)[name=tensor<string, []>(\"tv\")];\n", HEADS,SEQ,HD];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> dr = reshape(shape=rsh,x=df)[name=tensor<string, []>(\"rd\")];\n", HEADS,HD,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> da = transpose(perm=pm,x=dr)[name=tensor<string, []>(\"td\")];\n", HEADS,SEQ,HD];
    [m appendString:@"        tensor<bool, []> bF = const()[name=tensor<string, []>(\"bF\"), val=tensor<bool, []>(false)];\n"];
    [m appendString:@"        tensor<bool, []> bT = const()[name=tensor<string, []>(\"bT\"), val=tensor<bool, []>(true)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> sc1 = matmul(transpose_x=bF,transpose_y=bT,x=q,y=k)[name=tensor<string, []>(\"mm1\")];\n", HEADS,SEQ,SEQ];
    [m appendFormat:@"        tensor<fp16, []> scv = const()[name=tensor<string, []>(\"scv\"), val=tensor<fp16, []>(%f)];\n", sc];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> sc2 = mul(x=sc1,y=scv)[name=tensor<string, []>(\"scl\")];\n", HEADS,SEQ,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> cm = const()[name=tensor<string, []>(\"cm\"), val=tensor<fp16, [1,1,%d,%d]>(BLOBFILE(path=tensor<string, []>(\"@model_path/weights/mask.bin\"), offset=tensor<uint64, []>(64)))];\n", SEQ,SEQ,SEQ,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> ms = add(x=sc2,y=cm)[name=tensor<string, []>(\"msk\")];\n", HEADS,SEQ,SEQ];
    [m appendString:@"        tensor<int32, []> sax = const()[name=tensor<string, []>(\"sax\"), val=tensor<int32, []>(-1)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> probs = softmax(axis=sax,x=ms)[name=tensor<string, []>(\"sm\")];\n", HEADS,SEQ,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> dv4 = matmul(transpose_x=bT,transpose_y=bF,x=probs,y=da)[name=tensor<string, []>(\"dv\")];\n", HEADS,SEQ,HD];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> dp4 = matmul(transpose_x=bF,transpose_y=bT,x=da,y=v)[name=tensor<string, []>(\"dp\")];\n", HEADS,SEQ,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> dvt = transpose(perm=pm,x=dv4)[name=tensor<string, []>(\"dvt\")];\n", HEADS,HD,SEQ];
    [m appendFormat:@"        tensor<int32, [4]> dvs = const()[name=tensor<string, []>(\"dvs\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", DIM,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dvf = reshape(shape=dvs,x=dvt)[name=tensor<string, []>(\"dvf\")];\n", DIM,SEQ];
    [m appendFormat:@"        tensor<int32, [4]> scs = const()[name=tensor<string, []>(\"scs\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", SCORE_CH,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> pf = reshape(shape=scs,x=probs)[name=tensor<string, []>(\"pf\")];\n", SCORE_CH,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dpf = reshape(shape=scs,x=dp4)[name=tensor<string, []>(\"dpf\")];\n", SCORE_CH,SEQ];
    [m appendString:@"        tensor<int32, []> cax = const()[name=tensor<string, []>(\"cax\"), val=tensor<int32, []>(1)];\n"];
    [m appendString:@"        tensor<bool, []> cid = const()[name=tensor<string, []>(\"cid\"), val=tensor<bool, []>(false)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> out = concat(axis=cax,interleave=cid,values=(dvf,pf,dpf))[name=tensor<string, []>(\"cat\")];\n", DIM+2*SCORE_CH,SEQ];
    [m appendString:@"    } -> (out);\n}\n"];
    return m;
}

// SDPA backward part 2: concat(probs,dp,Q,K) → concat(dQ,dK)
static NSString *gen_sdpa_bwd2(void) {
    float sc = 1.0f/sqrtf((float)HD);
    int bwd2_in = 2*SCORE_CH + 2*DIM;
    NSMutableString *m = [NSMutableString string];
    [m appendString:MIL_HDR];
    [m appendFormat:@"    func main<ios16>(tensor<fp16, [1, %d, 1, %d]> x) {\n", bwd2_in, SEQ];
    [m appendFormat:@"        tensor<int32, [4]> sz_sc = const()[name=tensor<string, []>(\"szsc\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", SCORE_CH, SEQ];
    [m appendString:@"        tensor<int32, [4]> b0 = const()[name=tensor<string, []>(\"b0\"), val=tensor<int32, [4]>([0,0,0,0])];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> pf = slice_by_size(x=x,begin=b0,size=sz_sc)[name=tensor<string, []>(\"s0\")];\n", SCORE_CH,SEQ];
    [m appendFormat:@"        tensor<int32, [4]> b1 = const()[name=tensor<string, []>(\"b1\"), val=tensor<int32, [4]>([0,%d,0,0])];\n", SCORE_CH];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dpf = slice_by_size(x=x,begin=b1,size=sz_sc)[name=tensor<string, []>(\"s1\")];\n", SCORE_CH,SEQ];
    [m appendFormat:@"        tensor<int32, [4]> sz_d = const()[name=tensor<string, []>(\"szd\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<int32, [4]> b2 = const()[name=tensor<string, []>(\"b2\"), val=tensor<int32, [4]>([0,%d,0,0])];\n", 2*SCORE_CH];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> qf = slice_by_size(x=x,begin=b2,size=sz_d)[name=tensor<string, []>(\"s2\")];\n", DIM,SEQ];
    [m appendFormat:@"        tensor<int32, [4]> b3 = const()[name=tensor<string, []>(\"b3\"), val=tensor<int32, [4]>([0,%d,0,0])];\n", 2*SCORE_CH+DIM];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> kf = slice_by_size(x=x,begin=b3,size=sz_d)[name=tensor<string, []>(\"s3\")];\n", DIM,SEQ];
    [m appendFormat:@"        tensor<int32, [4]> ssh = const()[name=tensor<string, []>(\"ssh\"), val=tensor<int32, [4]>([1,%d,%d,%d])];\n", HEADS,SEQ,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> probs = reshape(shape=ssh,x=pf)[name=tensor<string, []>(\"rp\")];\n", HEADS,SEQ,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> dp = reshape(shape=ssh,x=dpf)[name=tensor<string, []>(\"rdp\")];\n", HEADS,SEQ,SEQ];
    [m appendFormat:@"        tensor<int32, [4]> rsh = const()[name=tensor<string, []>(\"rsh\"), val=tensor<int32, [4]>([1,%d,%d,%d])];\n", HEADS,HD,SEQ];
    [m appendString:@"        tensor<int32, [4]> pm = const()[name=tensor<string, []>(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> qr = reshape(shape=rsh,x=qf)[name=tensor<string, []>(\"rq\")];\n", HEADS,HD,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> q = transpose(perm=pm,x=qr)[name=tensor<string, []>(\"tq\")];\n", HEADS,SEQ,HD];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> kr = reshape(shape=rsh,x=kf)[name=tensor<string, []>(\"rk\")];\n", HEADS,HD,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> k = transpose(perm=pm,x=kr)[name=tensor<string, []>(\"tk\")];\n", HEADS,SEQ,HD];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> pdp = mul(x=probs,y=dp)[name=tensor<string, []>(\"pdp\")];\n", HEADS,SEQ,SEQ];
    [m appendString:@"        tensor<int32, [1]> rax = const()[name=tensor<string, []>(\"rax\"), val=tensor<int32, [1]>([-1])];\n"];
    [m appendString:@"        tensor<bool, []> kd = const()[name=tensor<string, []>(\"kd\"), val=tensor<bool, []>(true)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,1]> spdp = reduce_sum(x=pdp,axes=rax,keep_dims=kd)[name=tensor<string, []>(\"rs\")];\n", HEADS,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> dps = sub(x=dp,y=spdp)[name=tensor<string, []>(\"dps\")];\n", HEADS,SEQ,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> ds0 = mul(x=probs,y=dps)[name=tensor<string, []>(\"ds0\")];\n", HEADS,SEQ,SEQ];
    [m appendFormat:@"        tensor<fp16, []> scv = const()[name=tensor<string, []>(\"scv\"), val=tensor<fp16, []>(%f)];\n", sc];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> ds = mul(x=ds0,y=scv)[name=tensor<string, []>(\"ds\")];\n", HEADS,SEQ,SEQ];
    [m appendString:@"        tensor<bool, []> bF = const()[name=tensor<string, []>(\"bF\"), val=tensor<bool, []>(false)];\n"];
    [m appendString:@"        tensor<bool, []> bT = const()[name=tensor<string, []>(\"bT\"), val=tensor<bool, []>(true)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> dq4 = matmul(transpose_x=bF,transpose_y=bF,x=ds,y=k)[name=tensor<string, []>(\"dq\")];\n", HEADS,SEQ,HD];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> dk4 = matmul(transpose_x=bT,transpose_y=bF,x=ds,y=q)[name=tensor<string, []>(\"dk\")];\n", HEADS,SEQ,HD];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> dqt = transpose(perm=pm,x=dq4)[name=tensor<string, []>(\"dqt\")];\n", HEADS,HD,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> dkt = transpose(perm=pm,x=dk4)[name=tensor<string, []>(\"dkt\")];\n", HEADS,HD,SEQ];
    [m appendFormat:@"        tensor<int32, [4]> fs = const()[name=tensor<string, []>(\"fs\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", DIM,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dqf = reshape(shape=fs,x=dqt)[name=tensor<string, []>(\"dqf\")];\n", DIM,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dkf = reshape(shape=fs,x=dkt)[name=tensor<string, []>(\"dkf\")];\n", DIM,SEQ];
    [m appendString:@"        tensor<int32, []> cax = const()[name=tensor<string, []>(\"cax\"), val=tensor<int32, []>(1)];\n"];
    [m appendString:@"        tensor<bool, []> cid = const()[name=tensor<string, []>(\"cid\"), val=tensor<bool, []>(false)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> out = concat(axis=cax,interleave=cid,values=(dqf,dkf))[name=tensor<string, []>(\"cat\")];\n", 2*DIM,SEQ];
    [m appendString:@"    } -> (out);\n}\n"];
    return m;
}

// Mask blob (causal mask [SEQ,SEQ])
static NSData *g_mask_blob = nil;
static NSData *get_mask_blob(void) {
    if (!g_mask_blob) {
        _Float16 *mask = (_Float16*)calloc(SEQ*SEQ, sizeof(_Float16));
        for(int t=0;t<SEQ;t++) for(int t2=0;t2<SEQ;t2++)
            mask[t*SEQ+t2] = (t2<=t) ? (_Float16)0.0f : (_Float16)(-65504.0f);
        g_mask_blob = build_blob_fp16(mask, SEQ*SEQ);
        free(mask);
    }
    return g_mask_blob;
}
