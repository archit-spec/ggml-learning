#include "ggml.h"
#include <stdio.h>
#include <stdlib.h>

#define N_INPUTS 2
#define N_HIDDEN 3
#define N_OUTPUTS 1

void initialize_tensor_with_random_values(struct ggml_tensor *tensor) {
    for (int i = 0; i < ggml_nelements(tensor); i++) {
        ((float *)tensor->data)[i] = (float)rand() / RAND_MAX;
    }
}



int main(int argc, char **argv) {
    struct ggml_init_params params = {
        .mem_size = 16 * 1024 * 1024,
        .mem_buffer = NULL,
    };

    // Initialize the memory allocator
    struct ggml_context *ctx = ggml_init(params);
    if (!ctx) {
        fprintf(stderr, "Failed to initialize ggml context\n");
        return 1;
    }

    // Create the weights and biases for the hidden layer
    struct ggml_tensor *w1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N_INPUTS, N_HIDDEN);
    struct ggml_tensor *b1 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, N_HIDDEN);

    // Create the weights and biases for the output layer
    struct ggml_tensor *w2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N_HIDDEN, N_OUTPUTS);
    struct ggml_tensor *b2 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, N_OUTPUTS);

    // Initialize the weights and biases with random values
    initialize_tensor_with_random_values(w1);
    initialize_tensor_with_random_values(b1);
    initialize_tensor_with_random_values(w2);
    initialize_tensor_with_random_values(b2);

    // Create the input tensor
    struct ggml_tensor *input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, N_INPUTS);
    ggml_set_f32(input, 1.0f); // Set first value
    ((float *)input->data)[1] = 2.0f; // Set second value manually

    // Forward pass
    struct ggml_tensor *hidden = ggml_add(ctx, ggml_mul_mat(ctx, w1, input), b1);
    hidden = ggml_relu(ctx, hidden);
    struct ggml_tensor *output = ggml_add(ctx, ggml_mul_mat(ctx, w2, hidden), b2);

    // Print the output
    float *out = (float *)ggml_get_data(output);
    printf("Output: %f\n", out[0]);

    // Free the memory
    ggml_free(ctx);

    return 0;
}

