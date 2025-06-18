#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define WIDTH 512
#define HEIGHT 512
#define MAX_ITER 1000

void check(cl_int err, const char* msg) {
    if (err != CL_SUCCESS) {
        fprintf(stderr, "%s failed: %d\n", msg, err);
        exit(1);
    }
}

char* load_source(const char* path, size_t* size) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* src = malloc(len + 1);
    fread(src, 1, len, f);
    src[len] = '\0';
    fclose(f);
    if (size) *size = len;
    return src;
}

int main() {
    cl_int err;
    cl_uint num_platforms = 0;
    err = clGetPlatformIDs(0, NULL, &num_platforms);
    if (err != CL_SUCCESS || num_platforms == 0) {
        fprintf(stderr, "No OpenCL platforms found\n");
        return 1;
    }
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, NULL);

    cl_device_id device;
    cl_uint num_devices = 0;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, &num_devices);
    if (err != CL_SUCCESS || num_devices == 0) {
        fprintf(stderr, "No OpenCL devices found\n");
        return 1;
    }

    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    check(err, "create context");
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, 0, &err);
    check(err, "create queue");

    size_t src_size;
    char* src = load_source("mandelbrot.cl", &src_size);
    if (!src) {
        fprintf(stderr, "failed to load kernel source\n");
        return 1;
    }
    const char* sources[] = { src };
    cl_program program = clCreateProgramWithSource(context, 1, sources, &src_size, &err);
    check(err, "create program");
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char* log = malloc(log_size + 1);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        log[log_size] = '\0';
        fprintf(stderr, "Build log:\n%s\n", log);
        free(log);
        check(err, "build program");
    }

    cl_kernel kernel = clCreateKernel(program, "mandelbrot", &err);
    check(err, "create kernel");

    cl_image_format format;
    format.image_channel_order = CL_RGBA;
    format.image_channel_data_type = CL_UNORM_INT8;
    cl_image_desc desc;
    memset(&desc, 0, sizeof(desc));
    desc.image_type = CL_MEM_OBJECT_IMAGE2D;
    desc.image_width = WIDTH;
    desc.image_height = HEIGHT;

    const int width = WIDTH;
    const int height = HEIGHT;

    cl_mem image = clCreateImage(context, CL_MEM_WRITE_ONLY, &format, &desc, NULL, &err);
    check(err, "create image");

    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &image);
    err |= clSetKernelArg(kernel, 1, sizeof(int), &width);
    err |= clSetKernelArg(kernel, 2, sizeof(int), &height);
    float xmin = -2.0f, xmax = 1.0f, ymin = -1.5f, ymax = 1.5f;
    err |= clSetKernelArg(kernel, 3, sizeof(float), &xmin);
    err |= clSetKernelArg(kernel, 4, sizeof(float), &xmax);
    err |= clSetKernelArg(kernel, 5, sizeof(float), &ymin);
    err |= clSetKernelArg(kernel, 6, sizeof(float), &ymax);
    int max_iter = MAX_ITER;
    err |= clSetKernelArg(kernel, 7, sizeof(int), &max_iter);
    check(err, "set kernel args");

    size_t global_size = WIDTH * HEIGHT;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, NULL, 0, NULL, NULL);
    check(err, "enqueue kernel");
    clFinish(queue);

    size_t origin[3] = {0,0,0};
    size_t region[3] = {WIDTH, HEIGHT, 1};
    unsigned char* data = malloc(WIDTH * HEIGHT * 4);
    err = clEnqueueReadImage(queue, image, CL_TRUE, origin, region, 0, 0, data, 0, NULL, NULL);
    check(err, "read image");

    FILE* out = fopen("mandelbrot.ppm", "wb");
    fprintf(out, "P6\n%d %d\n255\n", WIDTH, HEIGHT);
    for (int i = 0; i < WIDTH * HEIGHT; ++i) {
        fwrite(&data[i*4], 1, 3, out);
    }
    fclose(out);

    free(data);
    clReleaseMemObject(image);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    free(src);
    return 0;
}
