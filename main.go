package main

import (
	"crypto/rand"
	"fmt"
	"math/big"
	mathrand "math/rand"
	"time"

	cl "github.com/CyberChainXyz/go-opencl"
)

const pollardKernel = `
ulong gcd(ulong a, ulong b) {
    while (b != 0) {
        ulong t = b;
        b = a % b;
        a = t;
    }
    return a;
}

__kernel void pollard_kernel(__global ulong* results, ulong n, ulong c, uint iterations) {
    size_t id = get_global_id(0);
    ulong x = id + 2;
    ulong y = x;
    ulong d = 1;

    for (uint i = 0; i < iterations && d == 1; i++) {
        x = (x * x + c) % n;
        y = (y * y + c) % n;
        y = (y * y + c) % n;
        ulong diff = x > y ? x - y : y - x;
        d = gcd(diff, n);
    }

    if (d != 1 && d != n) {
        results[id] = d;
    } else {
        results[id] = 0;
    }
}
`

func main() {
	mathrand.Seed(time.Now().UnixNano())

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

	if err := runner.CompileKernels([]string{pollardKernel}, []string{"pollard_kernel"}, ""); err != nil {
		panic(err)
	}

	threads := uint64(1024)
	iterations := uint32(1000)

	for bits := 16; bits <= 32; bits += 4 {
		p, _ := rand.Prime(rand.Reader, bits)
		q, _ := rand.Prime(rand.Reader, bits)
		n := new(big.Int).Mul(p, q).Uint64()

		fmt.Printf("Factoring %d-bit n = %d (p=%s, q=%s)...\n", bits*2, n, p.String(), q.String())

		found := false
		attempts := 0

		for !found {
			attempts++

			resultBuf, err := runner.CreateEmptyBuffer(cl.WRITE_ONLY, int(threads*8))
			if err != nil {
				panic(err)
			}

			c := uint64(mathrand.Intn(1000) + 1)
			args := []cl.KernelParam{
				cl.BufferParam(resultBuf),
				cl.Param(&n),
				cl.Param(&c),
				cl.Param(&iterations),
			}

			if err := runner.RunKernel("pollard_kernel", 1, nil, []uint64{threads}, nil, args, true); err != nil {
				panic(err)
			}

			results := make([]uint64, threads)
			if err := cl.ReadBuffer(runner, 0, resultBuf, results); err != nil {
				panic(err)
			}

			for _, d := range results {
				if d > 1 && d < n {
					fmt.Printf("Found factor: %d × %d = %d after %d attempt(s)\n", d, n/d, n, attempts)
					found = true
					break
				}
			}

			if !found {
				//fmt.Println("No factor found, retrying...")
			}
		}

	}
}
