package main

import (
	cl "github.com/CyberChainXyz/go-opencl"
)

func getFirstDevice() *cl.OpenCLDevice {
	info, err := cl.Info()
	if err != nil || info.Platform_count == 0 {
		return nil
	}
	for _, p := range info.Platforms {
		if len(p.Devices) > 0 {
			return p.Devices[0]
		}
	}
	return nil
}
