//go:build metal

// cgo compiles C++ (.cpp) files but not Objective-C++ (.mm), and only sources in
// the package directory (not subdirectories). This shim is therefore a .cpp that
// is forced to Objective-C++ via `-xobjective-c++` (see gpu_miner_metal.go) and
// pulls in the real implementation under kernel/. The build constraint keeps it
// out of non-metal builds.
#include "kernel/metal_launcher.mm"
