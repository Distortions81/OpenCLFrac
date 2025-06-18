package main

import (
	cl "github.com/CyberChainXyz/go-opencl"
)

// getBestDevice scans all available OpenCL devices and returns the one with the
// highest compute unit count. If no devices are found, nil is returned.
func getBestDevice() *cl.OpenCLDevice {
	info, err := cl.Info()
	if err != nil || info.Platform_count == 0 {
		return nil
	}

	var best *cl.OpenCLDevice
	var maxUnits uint32
	for _, p := range info.Platforms {
		for _, d := range p.Devices {
			if d.Max_compute_units > maxUnits {
				best = d
				maxUnits = d.Max_compute_units
			}
		}
	}
	return best
}
