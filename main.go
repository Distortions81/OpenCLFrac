package main

import (
	"fmt"
	"image"
	"image/png"
	"os"

	cl "github.com/CyberChainXyz/go-opencl"
)

const mandelbrotKernel = `
__kernel void mandelbrot(__global uchar* img,
                         float xmin, float xmax,
                         float ymin, float ymax,
                         uint width, uint height,
                         uint maxIter) {
    size_t gid = get_global_id(0);
    uint x = gid % width;
    uint y = gid / width;
    if (x >= width || y >= height) return;

    float dx = (xmax - xmin) / (float)(width - 1);
    float dy = (ymax - ymin) / (float)(height - 1);

    float rSum = 0.0f;
    float gSum = 0.0f;
    float bSum = 0.0f;

    for (int sy = 0; sy < 8; sy++) {
        for (int sx = 0; sx < 8; sx++) {
            float cr = xmin + ((float)x + ((float)sx + 0.5f) / 8.0f) * dx;
            float ci = ymin + ((float)y + ((float)sy + 0.5f) / 8.0f) * dy;

            float zr = 0.0f;
            float zi = 0.0f;
            uint iter = 0;
            while (zr * zr + zi * zi <= 4.0f && iter < maxIter) {
                float tmp = zr * zr - zi * zi + cr;
                zi = 2.0f * zr * zi + ci;
                zr = tmp;
                iter++;
            }

            float t = (float)iter / (float)maxIter;
            float r = 9.0f * (1.0f - t) * t * t * t;
            float g = 15.0f * (1.0f - t) * (1.0f - t) * t * t;
            float b = 8.5f * (1.0f - t) * (1.0f - t) * (1.0f - t) * t;

            rSum += r;
            gSum += g;
            bSum += b;
        }
    }

    ushort r = (ushort)(rSum / 64.0f * 65535.0f);
    ushort g = (ushort)(gSum / 64.0f * 65535.0f);
    ushort b = (ushort)(bSum / 64.0f * 65535.0f);

    uint idx = gid * 8;
    img[idx + 0] = (uchar)(r >> 8);
    img[idx + 1] = (uchar)(r & 0xFF);
    img[idx + 2] = (uchar)(g >> 8);
    img[idx + 3] = (uchar)(g & 0xFF);
    img[idx + 4] = (uchar)(b >> 8);
    img[idx + 5] = (uchar)(b & 0xFF);
    img[idx + 6] = 0xFF;
    img[idx + 7] = 0xFF;
}`

func main() {
	// Select the device with the most compute units to maximize performance.
	device := getBestDevice()
	if device == nil {
		fmt.Println("No OpenCL device found, skipping GPU run.")
		return
	}

	runner, err := device.InitRunner()
	if err != nil {
		panic(err)
	}
	defer runner.Free()

	// Compile kernel with relaxed math optimizations for better throughput.
	compileOpts := "-cl-fast-relaxed-math"
	if err := runner.CompileKernels([]string{mandelbrotKernel}, []string{"mandelbrot"}, compileOpts); err != nil {
		panic(err)
	}

	width := uint32(8192)
	height := uint32(8192)
	maxIter := uint32(10000)
	xmin := float32(-2.0)
	xmax := float32(1.0)
	ymin := float32(-1.2)
	ymax := float32(1.2)

	bufSize := int(width * height * 8)
	imgBuf, err := runner.CreateEmptyBuffer(cl.WRITE_ONLY, bufSize)
	if err != nil {
		panic(err)
	}

	args := []cl.KernelParam{
		cl.BufferParam(imgBuf),
		cl.Param(&xmin),
		cl.Param(&xmax),
		cl.Param(&ymin),
		cl.Param(&ymax),
		cl.Param(&width),
		cl.Param(&height),
		cl.Param(&maxIter),
	}

	// Use the device's maximum work-group size to better saturate the GPU.
	global := uint64(width) * uint64(height)
	local := []uint64{uint64(device.Max_work_group_size)}
	if err := runner.RunKernel("mandelbrot", 1, nil, []uint64{global}, local, args, true); err != nil {
		panic(err)
	}

	data := make([]byte, bufSize)
	if err := cl.ReadBuffer(runner, 0, imgBuf, data); err != nil {
		panic(err)
	}

	img := &image.RGBA64{
		Pix:    data,
		Stride: int(width) * 8,
		Rect:   image.Rect(0, 0, int(width), int(height)),
	}

	out, err := os.Create("mandelbrot.png")
	if err != nil {
		panic(err)
	}
	defer out.Close()

	if err := png.Encode(out, img); err != nil {
		panic(err)
	}

	fmt.Println("mandelbrot.png written")
}
