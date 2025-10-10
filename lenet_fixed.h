
#ifndef LENET_CNN_FIXED_H
#define LENET_CNN_FIXED_H

#include <stdint.h>

// Fixed-point configuration
#define FIXED_POINT_FRACTIONAL_BITS 12
#define FIXED_POINT_MULTIPLIER (1 << FIXED_POINT_FRACTIONAL_BITS)

// Type définition pour fixed-point (Q4.12 format - 4 bits entier, 12 bits fractionnaire)
typedef int16_t fixed_t;

// Conversion macros
#define FLOAT_TO_FIXED(x) ((fixed_t)((x) * FIXED_POINT_MULTIPLIER))
#define FIXED_TO_FLOAT(x) (((float)(x)) / FIXED_POINT_MULTIPLIER)
#define FIXED_MUL(a, b) (((int32_t)(a) * (int32_t)(b)) >> FIXED_POINT_FRACTIONAL_BITS)
#define FIXED_DIV(a, b) (((int32_t)(a) << FIXED_POINT_FRACTIONAL_BITS) / (b))

// Network dimensions
#define IMG_WIDTH 28
#define IMG_HEIGHT 28
#define IMG_DEPTH 1

#define CONV1_DIM 5
#define CONV1_NBOUTPUT 20
#define CONV1_WIDTH 24
#define CONV1_HEIGHT 24

#define POOL1_DIM 2
#define POOL1_NBOUTPUT 20
#define POOL1_WIDTH 12
#define POOL1_HEIGHT 12

#define CONV2_DIM 5
#define CONV2_NBOUTPUT 40
#define CONV2_WIDTH 8
#define CONV2_HEIGHT 8

#define POOL2_DIM 2
#define POOL2_NBOUTPUT 40
#define POOL2_WIDTH 4
#define POOL2_HEIGHT 4

#define FC1_NBOUTPUT 400
#define FC2_NBOUTPUT 10

// Function prototypes
void Conv1_28x28x1_5x5x20_1_0_fixed(
    fixed_t input[IMG_DEPTH][IMG_HEIGHT][IMG_WIDTH],
    fixed_t kernel[CONV1_NBOUTPUT][IMG_DEPTH][CONV1_DIM][CONV1_DIM],
    fixed_t bias[CONV1_NBOUTPUT],
    fixed_t output[CONV1_NBOUTPUT][CONV1_HEIGHT][CONV1_WIDTH]);

void Pool1_24x24x20_2x2x20_2_0_fixed(
    fixed_t input[CONV1_NBOUTPUT][CONV1_HEIGHT][CONV1_WIDTH],
    fixed_t output[POOL1_NBOUTPUT][POOL1_HEIGHT][POOL1_WIDTH]);

void Conv2_12x12x20_5x5x40_1_0_fixed(
    fixed_t input[POOL1_NBOUTPUT][POOL1_HEIGHT][POOL1_WIDTH],
    fixed_t kernel[CONV2_NBOUTPUT][POOL1_NBOUTPUT][CONV2_DIM][CONV2_DIM],
    fixed_t bias[CONV2_NBOUTPUT],
    fixed_t output[CONV2_NBOUTPUT][CONV2_HEIGHT][CONV2_WIDTH]);

void Pool2_8x8x40_2x2x40_2_0_fixed(
    fixed_t input[CONV2_NBOUTPUT][CONV2_HEIGHT][CONV2_WIDTH],
    fixed_t output[POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH]);

void Fc1_40_400_fixed(
    fixed_t input[POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH],
    fixed_t kernel[FC1_NBOUTPUT][POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH],
    fixed_t bias[FC1_NBOUTPUT],
    fixed_t output[FC1_NBOUTPUT]);

void Fc2_400_10_fixed(
    fixed_t input[FC1_NBOUTPUT],
    fixed_t kernel[FC2_NBOUTPUT][FC1_NBOUTPUT],
    fixed_t bias[FC2_NBOUTPUT],
    fixed_t output[FC2_NBOUTPUT]);

void Softmax_fixed(fixed_t vector_in[FC2_NBOUTPUT], fixed_t vector_out[FC2_NBOUTPUT]);

void lenet_cnn_fixed(
    fixed_t input[IMG_DEPTH][IMG_HEIGHT][IMG_WIDTH],
    fixed_t conv1_kernel[CONV1_NBOUTPUT][IMG_DEPTH][CONV1_DIM][CONV1_DIM],
    fixed_t conv1_bias[CONV1_NBOUTPUT],
    fixed_t conv2_kernel[CONV2_NBOUTPUT][POOL1_NBOUTPUT][CONV2_DIM][CONV2_DIM],
    fixed_t conv2_bias[CONV2_NBOUTPUT],
    fixed_t fc1_kernel[FC1_NBOUTPUT][POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH],
    fixed_t fc1_bias[FC1_NBOUTPUT],
    fixed_t fc2_kernel[FC2_NBOUTPUT][FC1_NBOUTPUT],
    fixed_t fc2_bias[FC2_NBOUTPUT],
    fixed_t output[FC2_NBOUTPUT]);

// Utility functions
void NormalizeImg_fixed(unsigned char *img, fixed_t *norm_img, int width, int height);
void ReadPgmFile(char *filename, unsigned char *data);
void ReadWeights_fixed(char *hdf5_filename, char *dataset_name, fixed_t *data, int size);

#endif // LENET_CNN_FIXED_H
