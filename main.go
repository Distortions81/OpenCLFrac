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

    float cr = xmin + ((float)x / (float)(width - 1)) * (xmax - xmin);
    float ci = ymin + ((float)y / (float)(height - 1)) * (ymax - ymin);
    float zr = 0.0f;
    float zi = 0.0f;
    uint iter = 0;
    while (zr * zr + zi * zi <= 4.0f && iter < maxIter) {
        float tmp = zr * zr - zi * zi + cr;
        zi = 2.0f * zr * zi + ci;
        zr = tmp;
        iter++;
    }
    uchar color = (uchar)(255 - (iter * 255) / maxIter);
    uint idx = gid * 4;
    img[idx + 0] = color;
    img[idx + 1] = color;
    img[idx + 2] = color;
    img[idx + 3] = 255;
}`

func main() {
	device := getFirstDevice()
	if device == nil {
		fmt.Println("No OpenCL device found, skipping GPU run.")
		return
	}

	runner, err := device.InitRunner()
	if err != nil {
		panic(err)
	}
	defer runner.Free()

	if err := runner.CompileKernels([]string{mandelbrotKernel}, []string{"mandelbrot"}, ""); err != nil {
		panic(err)
	}

	width := uint32(800)
	height := uint32(600)
	maxIter := uint32(1000)
	xmin := float32(-2.0)
	xmax := float32(1.0)
	ymin := float32(-1.2)
	ymax := float32(1.2)

	bufSize := int(width * height * 4)
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

	global := uint64(width) * uint64(height)
	if err := runner.RunKernel("mandelbrot", 1, nil, []uint64{global}, nil, args, true); err != nil {
		panic(err)
	}

	data := make([]byte, bufSize)
	if err := cl.ReadBuffer(runner, 0, imgBuf, data); err != nil {
		panic(err)
	}

	img := &image.RGBA{
		Pix:    data,
		Stride: int(width) * 4,
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
