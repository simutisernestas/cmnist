#include <stdio.h>
#include <stdlib.h>
#include <png.h>

#define IMAGE_SIZE 28

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
    png_byte color_type = png_get_color_type(png_ptr, info_ptr);
    png_byte bit_depth = png_get_bit_depth(png_ptr, info_ptr);

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

    fread(data, sizeof(int), size, fp);
    fclose(fp);
}

int main()
{
    const char *file_name = "data/0.png";
    int width, height;
    png_byte **image_data = read_png_file(file_name, &width, &height);

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
    int conv1_wn = 32*3*3;
    int conv1_bn = 32;
    float *conv1_weights = (float *)malloc(sizeof(float) * conv1_wn);
    float *conv1_bias = (float *)malloc(sizeof(float) * conv1_bn);
    read_in_bin_file("data/binm/conv1.bin", conv1_weights, conv1_wn);
    // for (size_t i = 0; i < conv1_wn; i++)
    //     printf("%f\n", conv1_weights[i]);

    // self.conv2 = nn.Conv2d(32, 64, 3, 1)
    // self.dropout1 = nn.Dropout2d(0.25)
    // self.dropout2 = nn.Dropout2d(0.5)
    // self.fc1 = nn.Linear(9216, 128)
    // self.fc2 = nn.Linear(128, 10)
    

    free_image_data(image_data, height);
    return 0;
}
