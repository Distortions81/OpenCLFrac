__kernel void mandelbrot(__global uchar4* output, int width, int height, float xmin, float xmax, float ymin, float ymax, int max_iter) {
    int gid = get_global_id(0);
    int x = gid % width;
    int y = gid / width;
    if (x >= width || y >= height) return;
    float real = xmin + (xmax - xmin) * ((float)x / (float)width);
    float imag = ymin + (ymax - ymin) * ((float)y / (float)height);
    float cReal = real;
    float cImag = imag;
    int iter = 0;
    for (; iter < max_iter; ++iter) {
        float real2 = real * real - imag * imag + cReal;
        imag = 2.0f * real * imag + cImag;
        real = real2;
        if (real * real + imag * imag > 4.0f) break;
    }
    uchar color = (uchar)(255 * ((float)iter / (float)max_iter));
    output[gid] = (uchar4)(color, color, color, 255);
}
