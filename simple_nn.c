#include "ggml.h"
#include <stdio.h>
#include <stdlib.h>

#define N_INPUTS 2
#define N_HIDDEN 3
#define N_OUTPUTS 1

int main(int argc, char ** argv) {
    struct ggml_init_params params = {
        .mem_size   = 16*1024*1024,
        .mem_buffer = NULL,
    };

    // initializing the memory allocator
    struct ggml_context * ctx = ggml_init(params);

    // creating the weights for the hidden layer
    struct ggml_tensor * w1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N_INPUTS, N_HIDDEN);
    struct ggml_tensor * b1 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, N_HIDDEN);

    // creating the weights for the output layer
    struct ggml_tensor * w2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N_HIDDEN, N_OUTPUTS);
    struct ggml_tensor * b2 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, N_OUTPUTS);

    // initializing the weights with random values
    for (int i = 0; i < ggml_nelements(w1); i++) {
        ((float *)w1->data)[i] = (float)rand() / RAND_MAX;
    }
    for (int i = 0; i < ggml_nelements(b1); i++) {
        ((float *)b1->data)[i] = (float)rand() / RAND_MAX;
    }
    for (int i = 0; i < ggml_nelements(w2); i++) {
        ((float *)w2->data)[i] = (float)rand() / RAND_MAX;
    }
    for (int i = 0; i < ggml_nelements(b2); i++) {
        ((float *)b2->data)[i] = (float)rand() / RAND_MAX;
    }

    // create the input
    struct ggml_tensor * input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, N_INPUTS);
    ggml_set_f32(input, 0, 1.0f);
    ggml_set_f32(input, 1, 2.0f);

    // forward pass
    struct ggml_tensor * hidden = ggml_add(ctx, ggml_mul_mat(ctx, w1, input), b1);
    hidden = ggml_relu(ctx, hidden);
    struct ggml_tensor * output = ggml_add(ctx, ggml_mul_mat(ctx, w2, hidden), b2);

    float * out = ggml_get_data_f32(output);
    printf("Output: %f\n", out[0]);

    ggml_free(ctx);

    return 0;
}
