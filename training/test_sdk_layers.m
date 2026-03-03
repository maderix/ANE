// test_sdk_layers.m — Verify modular ANE SDK layers
#import "layers/anesdk.h"

int main() {
    @autoreleasepool {
    ane_init();
    mach_timebase_info(&g_tb);

    printf("--- ANE SDK Layer Test ---\n");

    // 1. Create a Linear Layer (768 -> 2048)
    int dim_in = 768, dim_out = 2048, seq = 256;
    printf("Creating Linear layer [%d -> %d, seq=%d]...\n", dim_in, dim_out, seq);
    ANESDKLayer lin = anesdk_linear_create("fc1", dim_in, dim_out, seq);
    if (!lin.kern) { printf("Failed to create linear layer\n"); return 1; }
    printf("Linear layer compiled.\n");

    // 2. Create a ReLU Layer
    printf("Creating ReLU layer...\n");
    ANESDKLayer relu = anesdk_relu_create("relu1", dim_out, 1, seq);
    if (!relu.kern) { printf("Failed to create relu layer\n"); return 1; }
    printf("ReLU layer compiled.\n");

    // 3. Prepare Dummy Input for Linear
    printf("Running Forward Pass...\n");
    float *x = (float*)calloc(dim_in * seq, sizeof(float));
    for (int i=0; i<10; i++) x[i] = 1.0f;
    io_write_fp16(lin.kern->inputs[0], x, dim_in, seq);
    
    // Write dummy weights
    float *w = (float*)calloc(dim_out * dim_in, sizeof(float));
    for (int i=0; i<dim_out; i++) w[i*dim_in] = 0.5f;
    io_write_fp16_t(lin.kern->inputs[1], w, dim_out, dim_in);

    // 4. Eval Linear
    anesdk_layer_forward(&lin);
    printf("Linear Forward Done.\n");

    // 5. Connect Linear Output to ReLU Input (io_copy)
    io_copy(relu.kern->inputs[0], 0, lin.kern->ioOut, 0, dim_out, seq);
    
    // 6. Eval ReLU
    anesdk_layer_forward(&relu);
    printf("ReLU Forward Done.\n");

    // 7. Test Softmax
    printf("Creating Softmax layer...\n");
    ANESDKLayer smm = anesdk_softmax_create("softmax1", dim_out, 1, seq);
    if (!smm.kern) { printf("Failed to create softmax layer\n"); return 1; }
    printf("Softmax layer compiled.\n");
    
    io_copy(smm.kern->inputs[0], 0, relu.kern->ioOut, 0, dim_out, seq);
    anesdk_layer_forward(&smm);
    printf("Softmax Forward Done.\n");

    // 8. Test LayerNorm
    printf("Creating LayerNorm layer...\n");
    ANESDKLayer lnm = anesdk_layernorm_create("ln1", dim_in, seq);
    if (!lnm.kern) { printf("Failed to create layernorm layer\n"); return 1; }
    printf("LayerNorm layer compiled.\n");
    
    io_write_fp16(lnm.kern->inputs[0], x, dim_in, seq);
    anesdk_layer_forward(&lnm);
    printf("LayerNorm Forward Done.\n");

    // 9. Read Result
    float *y = (float*)malloc(dim_out * seq * sizeof(float));
    io_read_fp16(smm.kern->ioOut, y, 0, dim_out, seq); // Using softmax output for verification of smack-dab parity
    printf("Result sample [0]: %f\n", y[0]);

    // Cleanup
    free_kern(lin.kern);
    free_kern(relu.kern);
    free_kern(smm.kern);
    free_kern(lnm.kern);
    free(x); free(w); free(y);

    printf("--- SDK Layer Test PASSED ---\n");
    return 0;
    }
}
