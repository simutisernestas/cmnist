#include <stdio.h>
#include <stdlib.h>
#include <png.h>
#include <assert.h>
#include <math.h>
#include <omp.h>

#define IMAGE_SIZE 28

#define max(a, b) \
    __extension__({ typeof (a) _a = (a), _b = (b); _a > _b ? _a : _b; })

png_byte **read_png_file(const char *file_name, int *width, int *height)
{
    FILE *fp = fopen(file_name, "rb");
    if (!fp)
    {
        fprintf(stderr, "Error: Couldn't open the PNG file: %s\n", file_name);
        return NULL;
    }

    png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr)
    {
        fprintf(stderr, "Error: png_create_read_struct failed.\n");
        fclose(fp);
        return NULL;
    }

    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr)
    {
        fprintf(stderr, "Error: png_create_info_struct failed.\n");
        png_destroy_read_struct(&png_ptr, NULL, NULL);
        fclose(fp);
        return NULL;
    }

    if (setjmp(png_jmpbuf(png_ptr)))
    {
        fprintf(stderr, "Error: Error during PNG file reading.\n");
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
        fclose(fp);
        return NULL;
    }

    png_init_io(png_ptr, fp);
    png_read_info(png_ptr, info_ptr);

    *width = png_get_image_width(png_ptr, info_ptr);
    *height = png_get_image_height(png_ptr, info_ptr);

    // Allocate memory for the pixel data
    png_byte **image_data = (png_byte **)malloc(sizeof(png_byte *) * (*height));
    for (int y = 0; y < (*height); y++)
    {
        image_data[y] = (png_byte *)malloc(png_get_rowbytes(png_ptr, info_ptr));
    }

    // Read image data
    png_read_image(png_ptr, image_data);

    png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
    fclose(fp);

    return image_data;
}

void free_image_data(png_byte **image_data, int height)
{
    for (int y = 0; y < height; y++)
        free(image_data[y]);
    free(image_data);
}

void read_in_bin_file(const char *file_name, float *data, int size)
{
    FILE *fp = fopen(file_name, "rb");
    if (!fp)
    {
        fprintf(stderr, "Error: Couldn't open the bin file: %s\n", file_name);
        return;
    }

    int err = fread(data, sizeof(float), size, fp);
    if (err != size)
    {
        fprintf(stderr, "Error: Couldn't read the bin file: %s\n", file_name);
        return;
    }
    fclose(fp);
}

void save_buffer_to_bin_file(const char *file_name, float *data, int size)
{
    FILE *fp = fopen(file_name, "wb");
    if (!fp)
    {
        fprintf(stderr, "Error: Couldn't open the bin file: %s\n", file_name);
        return;
    }

    fwrite(data, sizeof(float), size, fp);
    fclose(fp);
}

void relu(float *data, int size)
{
    for (int i = 0; i < size; i++)
    {
        if (data[i] < 0)
            data[i] = 0;
    }
}

void relu_short(short *data, int size)
{
    for (int i = 0; i < size; i++)
    {
        if (data[i] < 0)
            data[i] = 0;
    }
}

#define QUANTIZE_FACTOR 32767.0f
#define MAX_FLOAT .4f

void quantize_weights(float *weights, short *quantized_weights, int size)
{
    // float max_value = 0;
    // for (int i = 0; i < size; i++)
    // {
    //     max_value = max(max_value, fabs(weights[i]));
    // }
    for (int i = 0; i < size; i++)
    {
        quantized_weights[i] = (short)round((weights[i] / MAX_FLOAT * QUANTIZE_FACTOR));
    }
}

void quantize_image(png_byte **image_data, short *quantized_image_data, int width, int height)
{
    float max = 0;
    for (int y = 0; y < height; y++)
    {
        png_byte *row = image_data[y];
        for (int x = 0; x < width; x++)
        {
            float pixel = ((float)row[x] / 255 - 0.1307) / 0.3081;
            max = max(max, fabs(pixel));
            quantized_image_data[y * width + x] = (short)round(pixel / MAX_FLOAT * QUANTIZE_FACTOR);
        }
    }
    printf("max: %f\n", max);
}

int main()
{
    const char *file_name = "data/0.png";
    int width, height;
    png_byte **image_data = read_png_file(file_name, &width, &height);
    if (!image_data)
        return -1;
    // quantize
    short *quantized_image_data = (short *)malloc(sizeof(short) * width * height);
    quantize_image(image_data, quantized_image_data, width, height);

    for (int y = 0; y < height; y++)
    {
        png_byte *row = image_data[y];
        for (int x = 0; x < width; x++)
        {
            png_byte pixel = row[x];
            if (pixel > 0)
                printf("#");
            else
                printf(" ");
        }
        printf("\n");
    }

    // self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1)
    int conv1_wn = 32 * 3 * 3;
    int conv1_bn = 32;
    float *conv1_weights = (float *)malloc(sizeof(float) * conv1_wn);
    float *conv1_bias = (float *)malloc(sizeof(float) * conv1_bn);
    short *quantized_conv1_weights = (short *)malloc(sizeof(short) * conv1_wn);
    short *quantized_conv1_bias = (short *)malloc(sizeof(short) * conv1_bn);
    read_in_bin_file("data/binm/conv1.bin", conv1_weights, conv1_wn);
    read_in_bin_file("data/binm/conv1_bias.bin", conv1_bias, conv1_bn);
    quantize_weights(conv1_weights, quantized_conv1_weights, conv1_wn);
    quantize_weights(conv1_bias, quantized_conv1_bias, conv1_bn);

    // self.conv2 = nn.Conv2d(32, 64, 3, 1)
    int conv2_wn = 32 * 64 * 3 * 3;
    int conv2_bn = 64;
    float *conv2_weights = (float *)malloc(sizeof(float) * conv2_wn);
    float *conv2_bias = (float *)malloc(sizeof(float) * conv2_bn);
    short *quantized_conv2_weights = (short *)malloc(sizeof(short) * conv2_wn);
    short *quantized_conv2_bias = (short *)malloc(sizeof(short) * conv2_bn);
    read_in_bin_file("data/binm/conv2.bin", conv2_weights, conv2_wn);
    read_in_bin_file("data/binm/conv2_bias.bin", conv2_bias, conv2_bn);
    quantize_weights(conv2_weights, quantized_conv2_weights, conv2_wn);
    quantize_weights(conv2_bias, quantized_conv2_bias, conv2_bn);

    // self.fc1 = nn.Linear(9216, 128)
    int fc1_wn = 9216 * 128;
    int fc1_bn = 128;
    float *fc1_weights = (float *)malloc(sizeof(float) * fc1_wn);
    float *fc1_bias = (float *)malloc(sizeof(float) * fc1_bn);
    short *quantized_fc1_weights = (short *)malloc(sizeof(short) * fc1_wn);
    short *quantized_fc1_bias = (short *)malloc(sizeof(short) * fc1_bn);
    read_in_bin_file("data/binm/fc1.bin", fc1_weights, fc1_wn);
    read_in_bin_file("data/binm/fc1_bias.bin", fc1_bias, fc1_bn);
    quantize_weights(fc1_weights, quantized_fc1_weights, fc1_wn);
    quantize_weights(fc1_bias, quantized_fc1_bias, fc1_bn);

    // self.fc2 = nn.Linear(128, 10)
    int fc2_wn = 128 * 10;
    int fc2_bn = 10;
    float *fc2_weights = (float *)malloc(sizeof(float) * fc2_wn);
    float *fc2_bias = (float *)malloc(sizeof(float) * fc2_bn);
    short *quantized_fc2_weights = (short *)malloc(sizeof(short) * fc2_wn);
    short *quantized_fc2_bias = (short *)malloc(sizeof(short) * fc2_bn);
    read_in_bin_file("data/binm/fc2.bin", fc2_weights, fc2_wn);
    read_in_bin_file("data/binm/fc2_bias.bin", fc2_bias, fc2_bn);
    quantize_weights(fc2_weights, quantized_fc2_weights, fc2_wn);
    quantize_weights(fc2_bias, quantized_fc2_bias, fc2_bn);

    int conv1_out_size = (height - 3 + 1) * (width - 3 + 1) * 32;
    short *conv1_out = (short *)malloc(sizeof(short) * conv1_out_size);
    for (int i = 0; i < conv1_out_size; i++)
        conv1_out[i] = 0;
    int conv1_out_height = 26;
    int conv1_out_width = 26;
    int conv2_out_height = (conv1_out_height - 3 + 1);
    int conv2_out_width = (conv1_out_width - 3 + 1);
    int conv2_out_size = conv2_out_height * conv2_out_width * 64;
    short *conv2_out = (short *)malloc(sizeof(short) * conv2_out_size);
    for (int i = 0; i < conv2_out_size; i++)
        conv2_out[i] = 0;
    int pool2_out_height = conv2_out_height / 2;
    int pool2_out_width = conv2_out_width / 2;
    int pool2_out_size = pool2_out_height * pool2_out_width * 64;
    short *pool2_out = (short *)malloc(sizeof(short) * pool2_out_size);
    for (int i = 0; i < pool2_out_size; i++)
        pool2_out[i] = 0;
    // fully connected 1
    int fc1_out_size = 128;
    short *fc1_out = (short *)malloc(sizeof(short) * fc1_out_size);
    for (int i = 0; i < fc1_out_size; i++)
        fc1_out[i] = 0;
    // fully connected 2
    int fc2_out_size = 10;
    short *fc2_out = (short *)malloc(sizeof(short) * fc2_out_size);
    for (int i = 0; i < fc2_out_size; i++)
        fc2_out[i] = 0;

    clock_t begin = clock();

#ifndef TESTOUT
    for (size_t i = 0; i < 1; i++)
    {
#endif
        for (int i = 0; i < 32; i++)
        { // conv1 out channel
            short *filter_weights = quantized_conv1_weights + (i * 3 * 3);
            for (int j = 0; j < height - 3 + 1; j++)
            { // conv1 out height
                for (int k = 0; k < width - 3 + 1; k++)
                { // conv1 out width
                    short *conv1_out_data = &(conv1_out[i * (height - 3 + 1) * (width - 3 + 1) +
                                                        j * (width - 3 + 1) + k]);
                    for (int m = 0; m < 3; m++)
                    { // conv1 in height
                        for (int n = 0; n < 3; n++)
                        { // conv1 in width
                            *conv1_out_data +=
                                quantized_image_data[(j + m) * width + k + n] * filter_weights[m * 3 + n];
                        }
                    }
                    *conv1_out_data += quantized_conv1_bias[i];
                }
            }
        }
#ifdef TESTOUT
        save_buffer_to_bin_file("log/conv1_out.bin", conv1_out, conv1_out_size);
#endif

        relu_short(conv1_out, conv1_out_size);

#ifdef TESTOUT
        save_buffer_to_bin_file("log/conv1_relu_out.bin", conv1_out, conv1_out_size);
#endif

        int idx = 0;
        for (int i = 0; i < 64; i++)
        { // dst channel
            int weight_base = i * 32 * 3 * 3;
            for (int j = 0; j < conv2_out_height; j++)
            { // conv2 out height
                for (int k = 0; k < conv2_out_width; k++)
                { // conv2 out width
                    for (int m = 0; m < 32; m++)
                    { // conv2 in channel
                        int src_base = m * conv1_out_height * conv1_out_width + j * conv1_out_width + k;
                        int weight_base_inner = weight_base + m * 3 * 3;
                        for (int n = 0; n < 3; n++)
                        { // kernel height
                            for (int p = 0; p < 3; p++)
                            { // kernel width
                                conv2_out[idx] += conv1_out[src_base + n * conv1_out_width + p] * quantized_conv2_weights[weight_base_inner + n * 3 + p];
                            }
                        }
                    }
                    conv2_out[idx] += quantized_conv2_bias[i];
                    idx++;
                }
            }
        }
        assert(idx == conv2_out_size);
#ifdef TESTOUT
        save_buffer_to_bin_file("log/conv2_out.bin", conv2_out, conv2_out_size);
#endif

        for (int i = 0; i < 64; i++)
        { // dst channel
            for (int j = 0; j < pool2_out_height; j++)
            { // pool2 out height
                for (int k = 0; k < pool2_out_width; k++)
                { // pool2 out width
                    int src_base = i * conv2_out_height * conv2_out_width + j * 2 * conv2_out_width + k * 2;
                    short max_val = conv2_out[src_base];
                    for (int m = 0; m < 2; m++)
                    { // pool2 in height
                        for (int n = 0; n < 2; n++)
                        { // pool2 in width
                            max_val = max(max_val, conv2_out[src_base + m * conv2_out_width + n]);
                        }
                    }
                    pool2_out[i * pool2_out_height * pool2_out_width + j * pool2_out_width + k] = max_val;
                }
            }
        }

#ifdef TESTOUT
        save_buffer_to_bin_file("log/max_pool2d_out.bin", pool2_out, pool2_out_size);
#endif

        for (int i = 0; i < fc1_out_size; i++)
        { // dst channel
            for (int j = 0; j < pool2_out_size; j++)
            { // src channel
                fc1_out[i] += pool2_out[j] * quantized_fc1_weights[i * pool2_out_size + j];
            }
            fc1_out[i] += quantized_fc1_bias[i];
        }
#ifdef TESTOUT
        save_buffer_to_bin_file("log/fc1_out.bin", fc1_out, fc1_out_size);
#endif
        relu_short(fc1_out, fc1_out_size);

#ifdef TESTOUT
        save_buffer_to_bin_file("log/fc1_relu_out.bin", fc1_out, fc1_out_size);
#endif

        for (int i = 0; i < fc2_out_size; i++)
        { // dst channel
            for (int j = 0; j < fc1_out_size; j++)
            { // src channel
                fc2_out[i] += fc1_out[j] * quantized_fc2_weights[i * fc1_out_size + j];
            }
            fc2_out[i] += quantized_fc2_bias[i];
        }

#ifdef TESTOUT
        save_buffer_to_bin_file("log/fc2_out.bin", fc2_out, fc2_out_size);
#endif

#ifndef TESTOUT
    }
#endif

    clock_t end = clock();
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("Time spent: %f\n", time_spent / 1000);

    // dequantize fc2_out
    float *dequantized_fc2_out = (float *)malloc(sizeof(float) * fc2_out_size);
    short max_abs = fc2_out[0];
    for (int i = 1; i < fc2_out_size; i++)
    {
        max_abs = max(max_abs, abs(fc2_out[i]));
    }
    printf("max_abs is %d\n", max_abs);
    for (int i = 0; i < fc2_out_size; i++)
    {
        dequantized_fc2_out[i] = ((float)max_abs / QUANTIZE_FACTOR * (float)fc2_out[i]);
    }

    int max_idx = 0;
    float max_val = dequantized_fc2_out[0];
    for (int i = 1; i < fc2_out_size; i++)
    {
        if (dequantized_fc2_out[i] > max_val)
        {
            max_val = dequantized_fc2_out[i];
            max_idx = i;
        }
    }
    printf("The result is %d\n", max_idx);

    free_image_data(image_data, height);
    free(conv1_weights);
    free(conv1_bias);
    free(conv2_weights);
    free(conv2_bias);
    free(fc1_weights);
    free(fc1_bias);
    free(fc2_weights);
    free(fc2_bias);
    free(conv1_out);
    free(conv2_out);
    free(pool2_out);
    free(fc1_out);
    free(fc2_out);
    return 0;
}
