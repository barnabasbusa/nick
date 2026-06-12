//go:build (linux || darwin) && !nocl
// +build linux darwin
// +build !nocl

/*
 * OpenCL GPU miner (Linux + macOS) for Nick's-method vanity addresses.
 * Runs the secp256k1-ecrecover + keccak kernel (kernel/nick.cl).
 */

package miner

/*
#cgo linux LDFLAGS: -lOpenCL
#cgo darwin LDFLAGS: -framework OpenCL

#define CL_TARGET_OPENCL_VERSION 120
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif
#include <stdlib.h>
#include <string.h>

const char* cl_error_string(cl_int error) {
    switch(error) {
        case CL_SUCCESS: return "CL_SUCCESS";
        case CL_DEVICE_NOT_FOUND: return "CL_DEVICE_NOT_FOUND";
        case CL_DEVICE_NOT_AVAILABLE: return "CL_DEVICE_NOT_AVAILABLE";
        case CL_COMPILER_NOT_AVAILABLE: return "CL_COMPILER_NOT_AVAILABLE";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case CL_OUT_OF_RESOURCES: return "CL_OUT_OF_RESOURCES";
        case CL_OUT_OF_HOST_MEMORY: return "CL_OUT_OF_HOST_MEMORY";
        case CL_BUILD_PROGRAM_FAILURE: return "CL_BUILD_PROGRAM_FAILURE";
        case CL_INVALID_VALUE: return "CL_INVALID_VALUE";
        case CL_INVALID_PLATFORM: return "CL_INVALID_PLATFORM";
        case CL_INVALID_DEVICE: return "CL_INVALID_DEVICE";
        case CL_INVALID_CONTEXT: return "CL_INVALID_CONTEXT";
        case CL_INVALID_COMMAND_QUEUE: return "CL_INVALID_COMMAND_QUEUE";
        case CL_INVALID_MEM_OBJECT: return "CL_INVALID_MEM_OBJECT";
        case CL_INVALID_PROGRAM: return "CL_INVALID_PROGRAM";
        case CL_INVALID_PROGRAM_EXECUTABLE: return "CL_INVALID_PROGRAM_EXECUTABLE";
        case CL_INVALID_KERNEL_NAME: return "CL_INVALID_KERNEL_NAME";
        case CL_INVALID_KERNEL: return "CL_INVALID_KERNEL";
        case CL_INVALID_ARG_INDEX: return "CL_INVALID_ARG_INDEX";
        case CL_INVALID_ARG_VALUE: return "CL_INVALID_ARG_VALUE";
        case CL_INVALID_ARG_SIZE: return "CL_INVALID_ARG_SIZE";
        case CL_INVALID_KERNEL_ARGS: return "CL_INVALID_KERNEL_ARGS";
        case CL_INVALID_WORK_GROUP_SIZE: return "CL_INVALID_WORK_GROUP_SIZE";
        default: return "Unknown error";
    }
}
*/
import "C"

import (
	"embed"
	"fmt"
	"time"
	"unsafe"
)

//go:embed kernel/nick_lib.cl kernel/nick.cl
var kernelFS embed.FS

// GPUMiner is an OpenCL miner instance bound to a single device.
type GPUMiner struct {
	platform   C.cl_platform_id
	device     C.cl_device_id
	context    C.cl_context
	queue      C.cl_command_queue
	program    C.cl_program
	kernel     C.cl_kernel
	deviceName string
	batchSize  int

	tableBuf  C.cl_mem
	qbaseBuf  C.cl_mem
	loadedFor unsafe.Pointer // identifies the Precompute whose table is uploaded
}

// ListGPUs returns the available OpenCL GPU devices.
func ListGPUs() ([]GPUInfo, error) {
	var numPlatforms C.cl_uint
	if ret := C.clGetPlatformIDs(0, nil, &numPlatforms); ret != C.CL_SUCCESS {
		return nil, fmt.Errorf("failed to get platform count: %s", C.GoString(C.cl_error_string(ret)))
	}
	if numPlatforms == 0 {
		return nil, fmt.Errorf("no OpenCL platforms found")
	}
	platforms := make([]C.cl_platform_id, numPlatforms)
	if ret := C.clGetPlatformIDs(numPlatforms, &platforms[0], nil); ret != C.CL_SUCCESS {
		return nil, fmt.Errorf("failed to get platforms: %s", C.GoString(C.cl_error_string(ret)))
	}

	var gpus []GPUInfo
	gpuIndex := 0
	for _, platform := range platforms {
		var numDevices C.cl_uint
		ret := C.clGetDeviceIDs(platform, C.CL_DEVICE_TYPE_GPU, 0, nil, &numDevices)
		if ret != C.CL_SUCCESS || numDevices == 0 {
			continue
		}
		devices := make([]C.cl_device_id, numDevices)
		if C.clGetDeviceIDs(platform, C.CL_DEVICE_TYPE_GPU, numDevices, &devices[0], nil) != C.CL_SUCCESS {
			continue
		}
		for _, device := range devices {
			info := getDeviceInfo(device)
			info.Index = gpuIndex
			gpus = append(gpus, info)
			gpuIndex++
		}
	}
	return gpus, nil
}

func getDeviceInfo(device C.cl_device_id) GPUInfo {
	var info GPUInfo
	var nameSize C.size_t
	C.clGetDeviceInfo(device, C.CL_DEVICE_NAME, 0, nil, &nameSize)
	nameBuf := make([]byte, nameSize)
	C.clGetDeviceInfo(device, C.CL_DEVICE_NAME, nameSize, unsafe.Pointer(&nameBuf[0]), nil)
	if nameSize > 0 {
		info.Name = string(nameBuf[:nameSize-1])
	}

	var vendorSize C.size_t
	C.clGetDeviceInfo(device, C.CL_DEVICE_VENDOR, 0, nil, &vendorSize)
	vendorBuf := make([]byte, vendorSize)
	C.clGetDeviceInfo(device, C.CL_DEVICE_VENDOR, vendorSize, unsafe.Pointer(&vendorBuf[0]), nil)
	if vendorSize > 0 {
		info.Vendor = string(vendorBuf[:vendorSize-1])
	}

	var computeUnits C.cl_uint
	C.clGetDeviceInfo(device, C.CL_DEVICE_MAX_COMPUTE_UNITS, C.size_t(unsafe.Sizeof(computeUnits)),
		unsafe.Pointer(&computeUnits), nil)
	info.ComputeUnits = int(computeUnits)

	var maxWorkSize C.size_t
	C.clGetDeviceInfo(device, C.CL_DEVICE_MAX_WORK_GROUP_SIZE, C.size_t(unsafe.Sizeof(maxWorkSize)),
		unsafe.Pointer(&maxWorkSize), nil)
	info.MaxWorkSize = int(maxWorkSize)
	return info
}

// NewGPUMiner creates an OpenCL miner for the GPU at deviceIndex.
func NewGPUMiner(deviceIndex int, batchSize int) (*GPUMiner, error) {
	m := &GPUMiner{batchSize: batchSize}

	var numPlatforms C.cl_uint
	if ret := C.clGetPlatformIDs(0, nil, &numPlatforms); ret != C.CL_SUCCESS {
		return nil, fmt.Errorf("failed to get platform count: %s", C.GoString(C.cl_error_string(ret)))
	}
	platforms := make([]C.cl_platform_id, numPlatforms)
	if ret := C.clGetPlatformIDs(numPlatforms, &platforms[0], nil); ret != C.CL_SUCCESS {
		return nil, fmt.Errorf("failed to get platforms: %s", C.GoString(C.cl_error_string(ret)))
	}

	gpuIndex := 0
	found := false
	for _, platform := range platforms {
		var numDevices C.cl_uint
		ret := C.clGetDeviceIDs(platform, C.CL_DEVICE_TYPE_GPU, 0, nil, &numDevices)
		if ret != C.CL_SUCCESS || numDevices == 0 {
			continue
		}
		devices := make([]C.cl_device_id, numDevices)
		if C.clGetDeviceIDs(platform, C.CL_DEVICE_TYPE_GPU, numDevices, &devices[0], nil) != C.CL_SUCCESS {
			continue
		}
		for _, device := range devices {
			if gpuIndex == deviceIndex {
				m.platform = platform
				m.device = device
				found = true
				break
			}
			gpuIndex++
		}
		if found {
			break
		}
	}
	if !found {
		return nil, fmt.Errorf("GPU device %d not found", deviceIndex)
	}

	info := getDeviceInfo(m.device)
	m.deviceName = info.Name

	var errCode C.cl_int
	m.context = C.clCreateContext(nil, 1, &m.device, nil, nil, &errCode)
	if errCode != C.CL_SUCCESS {
		return nil, fmt.Errorf("failed to create context: %s", C.GoString(C.cl_error_string(errCode)))
	}
	m.queue = C.clCreateCommandQueue(m.context, m.device, 0, &errCode)
	if errCode != C.CL_SUCCESS {
		C.clReleaseContext(m.context)
		return nil, fmt.Errorf("failed to create command queue: %s", C.GoString(C.cl_error_string(errCode)))
	}

	// Build the program from the shared lib + kernel entry (two source strings).
	libSrc, err := kernelFS.ReadFile("kernel/nick_lib.cl")
	if err != nil {
		m.Close()
		return nil, fmt.Errorf("failed to read kernel lib: %v", err)
	}
	kernSrc, err := kernelFS.ReadFile("kernel/nick.cl")
	if err != nil {
		m.Close()
		return nil, fmt.Errorf("failed to read kernel: %v", err)
	}

	libC := C.CString(string(libSrc))
	defer C.free(unsafe.Pointer(libC))
	kernC := C.CString(string(kernSrc))
	defer C.free(unsafe.Pointer(kernC))
	srcs := [2]*C.char{libC, kernC}
	lens := [2]C.size_t{C.size_t(len(libSrc)), C.size_t(len(kernSrc))}

	m.program = C.clCreateProgramWithSource(m.context, 2, &srcs[0], &lens[0], &errCode)
	if errCode != C.CL_SUCCESS {
		m.Close()
		return nil, fmt.Errorf("failed to create program: %s", C.GoString(C.cl_error_string(errCode)))
	}
	if ret := C.clBuildProgram(m.program, 1, &m.device, nil, nil, nil); ret != C.CL_SUCCESS {
		var logSize C.size_t
		C.clGetProgramBuildInfo(m.program, m.device, C.CL_PROGRAM_BUILD_LOG, 0, nil, &logSize)
		logBuf := make([]byte, logSize)
		C.clGetProgramBuildInfo(m.program, m.device, C.CL_PROGRAM_BUILD_LOG, logSize,
			unsafe.Pointer(&logBuf[0]), nil)
		m.Close()
		return nil, fmt.Errorf("failed to build program: %s\nBuild log:\n%s",
			C.GoString(C.cl_error_string(ret)), string(logBuf))
	}

	kernelName := C.CString("mine_nick")
	defer C.free(unsafe.Pointer(kernelName))
	m.kernel = C.clCreateKernel(m.program, kernelName, &errCode)
	if errCode != C.CL_SUCCESS {
		m.Close()
		return nil, fmt.Errorf("failed to create kernel: %s", C.GoString(C.cl_error_string(errCode)))
	}
	return m, nil
}

// Close releases all OpenCL resources.
func (m *GPUMiner) Close() {
	if m.tableBuf != nil {
		C.clReleaseMemObject(m.tableBuf)
		m.tableBuf = nil
	}
	if m.qbaseBuf != nil {
		C.clReleaseMemObject(m.qbaseBuf)
		m.qbaseBuf = nil
	}
	if m.kernel != nil {
		C.clReleaseKernel(m.kernel)
	}
	if m.program != nil {
		C.clReleaseProgram(m.program)
	}
	if m.queue != nil {
		C.clReleaseCommandQueue(m.queue)
	}
	if m.context != nil {
		C.clReleaseContext(m.context)
	}
}

// DeviceName returns the device name.
func (m *GPUMiner) DeviceName() string { return m.deviceName }

// BatchSize returns the per-batch candidate count.
func (m *GPUMiner) BatchSize() int { return m.batchSize }

// ensureTable uploads the (constant) comb table and Q_base once per Precompute.
func (m *GPUMiner) ensureTable(p *Precompute) error {
	if m.loadedFor == unsafe.Pointer(p) && m.tableBuf != nil {
		return nil
	}
	if m.tableBuf != nil {
		C.clReleaseMemObject(m.tableBuf)
		m.tableBuf = nil
	}
	if m.qbaseBuf != nil {
		C.clReleaseMemObject(m.qbaseBuf)
		m.qbaseBuf = nil
	}
	var errCode C.cl_int
	m.tableBuf = C.clCreateBuffer(m.context, C.CL_MEM_READ_ONLY|C.CL_MEM_COPY_HOST_PTR,
		C.size_t(len(p.DTable)), unsafe.Pointer(&p.DTable[0]), &errCode)
	if errCode != C.CL_SUCCESS {
		return fmt.Errorf("failed to create table buffer: %s", C.GoString(C.cl_error_string(errCode)))
	}
	m.qbaseBuf = C.clCreateBuffer(m.context, C.CL_MEM_READ_ONLY|C.CL_MEM_COPY_HOST_PTR,
		C.size_t(len(p.QBaseBytes)), unsafe.Pointer(&p.QBaseBytes[0]), &errCode)
	if errCode != C.CL_SUCCESS {
		return fmt.Errorf("failed to create qbase buffer: %s", C.GoString(C.cl_error_string(errCode)))
	}
	m.loadedFor = unsafe.Pointer(p)
	return nil
}

// smallBuf creates a read-only buffer from b (uses a 1-byte dummy if empty).
func (m *GPUMiner) smallBuf(b []byte) (C.cl_mem, error) {
	data := b
	if len(data) == 0 {
		data = []byte{0}
	}
	var errCode C.cl_int
	buf := C.clCreateBuffer(m.context, C.CL_MEM_READ_ONLY|C.CL_MEM_COPY_HOST_PTR,
		C.size_t(len(data)), unsafe.Pointer(&data[0]), &errCode)
	if errCode != C.CL_SUCCESS {
		return nil, fmt.Errorf("failed to create buffer: %s", C.GoString(C.cl_error_string(errCode)))
	}
	return buf, nil
}

// Mine runs one batch of the nick kernel.
func (m *GPUMiner) Mine(p *Precompute, prefix, suffix []byte, startNonce uint64) (*GPUResult, time.Duration, error) {
	startTime := time.Now()
	if err := m.ensureTable(p); err != nil {
		return nil, 0, err
	}

	prefixBuf, err := m.smallBuf(prefix)
	if err != nil {
		return nil, 0, err
	}
	defer C.clReleaseMemObject(prefixBuf)
	suffixBuf, err := m.smallBuf(suffix)
	if err != nil {
		return nil, 0, err
	}
	defer C.clReleaseMemObject(suffixBuf)

	var errCode C.cl_int
	resultAddrBuf := C.clCreateBuffer(m.context, C.CL_MEM_WRITE_ONLY, 20, nil, &errCode)
	if errCode != C.CL_SUCCESS {
		return nil, 0, fmt.Errorf("failed to create result addr buffer: %s", C.GoString(C.cl_error_string(errCode)))
	}
	defer C.clReleaseMemObject(resultAddrBuf)
	resultNonceBuf := C.clCreateBuffer(m.context, C.CL_MEM_WRITE_ONLY, 8, nil, &errCode)
	if errCode != C.CL_SUCCESS {
		return nil, 0, fmt.Errorf("failed to create result nonce buffer: %s", C.GoString(C.cl_error_string(errCode)))
	}
	defer C.clReleaseMemObject(resultNonceBuf)

	found := int32(0)
	foundBuf := C.clCreateBuffer(m.context, C.CL_MEM_READ_WRITE|C.CL_MEM_COPY_HOST_PTR,
		C.size_t(unsafe.Sizeof(found)), unsafe.Pointer(&found), &errCode)
	if errCode != C.CL_SUCCESS {
		return nil, 0, fmt.Errorf("failed to create found buffer: %s", C.GoString(C.cl_error_string(errCode)))
	}
	defer C.clReleaseMemObject(foundBuf)

	prefixLen := C.int(len(prefix))
	suffixLen := C.int(len(suffix))
	startNonceC := C.ulong(startNonce)

	C.clSetKernelArg(m.kernel, 0, C.size_t(unsafe.Sizeof(m.tableBuf)), unsafe.Pointer(&m.tableBuf))
	C.clSetKernelArg(m.kernel, 1, C.size_t(unsafe.Sizeof(m.qbaseBuf)), unsafe.Pointer(&m.qbaseBuf))
	C.clSetKernelArg(m.kernel, 2, C.size_t(unsafe.Sizeof(prefixBuf)), unsafe.Pointer(&prefixBuf))
	C.clSetKernelArg(m.kernel, 3, C.size_t(unsafe.Sizeof(prefixLen)), unsafe.Pointer(&prefixLen))
	C.clSetKernelArg(m.kernel, 4, C.size_t(unsafe.Sizeof(suffixBuf)), unsafe.Pointer(&suffixBuf))
	C.clSetKernelArg(m.kernel, 5, C.size_t(unsafe.Sizeof(suffixLen)), unsafe.Pointer(&suffixLen))
	C.clSetKernelArg(m.kernel, 6, C.size_t(unsafe.Sizeof(startNonceC)), unsafe.Pointer(&startNonceC))
	C.clSetKernelArg(m.kernel, 7, C.size_t(unsafe.Sizeof(resultAddrBuf)), unsafe.Pointer(&resultAddrBuf))
	C.clSetKernelArg(m.kernel, 8, C.size_t(unsafe.Sizeof(resultNonceBuf)), unsafe.Pointer(&resultNonceBuf))
	C.clSetKernelArg(m.kernel, 9, C.size_t(unsafe.Sizeof(foundBuf)), unsafe.Pointer(&foundBuf))

	localSize := C.size_t(64)
	globalSize := C.size_t(m.batchSize)
	if globalSize%localSize != 0 {
		globalSize = ((globalSize / localSize) + 1) * localSize
	}

	if ret := C.clEnqueueNDRangeKernel(m.queue, m.kernel, 1, nil, &globalSize, &localSize, 0, nil, nil); ret != C.CL_SUCCESS {
		return nil, 0, fmt.Errorf("failed to execute kernel: %s", C.GoString(C.cl_error_string(ret)))
	}
	C.clFinish(m.queue)

	C.clEnqueueReadBuffer(m.queue, foundBuf, C.CL_TRUE, 0, C.size_t(unsafe.Sizeof(found)),
		unsafe.Pointer(&found), 0, nil, nil)
	elapsed := time.Since(startTime)

	if found != 0 {
		var addr [20]byte
		var nonceBytes [8]byte
		C.clEnqueueReadBuffer(m.queue, resultAddrBuf, C.CL_TRUE, 0, 20, unsafe.Pointer(&addr[0]), 0, nil, nil)
		C.clEnqueueReadBuffer(m.queue, resultNonceBuf, C.CL_TRUE, 0, 8, unsafe.Pointer(&nonceBytes[0]), 0, nil, nil)
		res := &GPUResult{}
		res.Address = addr
		res.Nonce = uint64(nonceBytes[0]) | uint64(nonceBytes[1])<<8 | uint64(nonceBytes[2])<<16 |
			uint64(nonceBytes[3])<<24 | uint64(nonceBytes[4])<<32 | uint64(nonceBytes[5])<<40 |
			uint64(nonceBytes[6])<<48 | uint64(nonceBytes[7])<<56
		return res, elapsed, nil
	}
	return nil, elapsed, nil
}
