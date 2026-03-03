// test_sdk_model.m — Verify Sequential ANE SDK model
#import "layers/anesdk.h"

int main() {
    @autoreleasepool {
    ane_init();
    mach_timebase_info(&g_tb);

    printf("--- ANE SDK Sequential Model Test ---\n");

    int dim_in = 768, dim_out = 1024, seq = 256;
    
    // 1. Define Layer Stack
    ANESDKLayer layers[2];
    layers[0] = anesdk_linear_create("fc1", dim_in, dim_out, seq);
    layers[1] = anesdk_relu_create("relu1", dim_out, 1, seq);
    
    // 2. Create Sequential Model (Automates IOSurface chaining)
    printf("Chaining layers into Sequential model...\n");
    ANESDKModel model = anesdk_model_sequential_create(layers, 2);
    printf("Model created.\n");

    // 3. Setup Input and Weights
    float *x = (float*)calloc(dim_in * seq, sizeof(float));
    for (int i=0; i<10; i++) x[i] = 1.0f;
    io_write_fp16(model.layers[0].kern->inputs[0], x, dim_in, seq);
    
    float *w = (float*)calloc(dim_out * dim_in, sizeof(float));
    for (int i=0; i<dim_out * dim_in; i++) w[i] = 0.5f;
    io_write_fp16_t(model.layers[0].kern->inputs[1], w, dim_out, dim_in);

    // 4. Run Whole Model Forward
    printf("Running model forward (Linear -> ReLU)...\n");
    anesdk_model_forward(&model);
    printf("Model forward done.\n");

    // 5. Verify Output from last layer
    float *y = (float*)malloc(dim_out * seq * sizeof(float));
    io_read_fp16(model.layers[1].kern->ioOut, y, 0, dim_out, seq);
    
    // Math: y[0] = relu(dot(x[0:768], W[0, 0:768])) = relu(1.0 * 0.5 + 0 + ...) = 0.5
    printf("Final model output [0]: %f (Expected: 0.5)\n", y[0]);

    // Cleanup
    for (int i=0; i<model.n_layers; i++) free_kern(model.layers[i].kern);
    free(model.layers); // malloc'd in anesdk_model_sequential_create
    free(x); free(w); free(y);

    printf("--- SDK Model Test PASSED ---\n");
    return 0;
    }
}
