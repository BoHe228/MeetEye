// Fused OpenCL direct-slice remap benchmark backend.
//
// It maps one BGR fisheye frame plus slice_map_x/slice_map_y directly into
// RKNN-ready RGB NHWC inputs: [num_slices, imgsz, imgsz, 3].
// This is intentionally a benchmark-only C API and is not wired into main.py.

#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

// Minimal OpenCL C API declarations keep this benchmark buildable on boards
// that ship OpenCL runtime libraries but not OpenCL development headers.
using cl_bool = uint32_t;
using cl_uint = uint32_t;
using cl_int = int32_t;
using cl_ulong = uint64_t;
using cl_device_type = cl_ulong;
using cl_mem_flags = cl_ulong;
using cl_command_queue_properties = cl_ulong;
using cl_context_properties = intptr_t;
using cl_program_build_info = cl_uint;
using cl_platform_info = cl_uint;
using cl_device_info = cl_uint;
using cl_import_properties_arm = intptr_t;

struct _cl_platform_id;
struct _cl_device_id;
struct _cl_context;
struct _cl_command_queue;
struct _cl_program;
struct _cl_kernel;
struct _cl_mem;
struct _cl_event;

using cl_platform_id = _cl_platform_id*;
using cl_device_id = _cl_device_id*;
using cl_context = _cl_context*;
using cl_command_queue = _cl_command_queue*;
using cl_program = _cl_program*;
using cl_kernel = _cl_kernel*;
using cl_mem = _cl_mem*;
using cl_event = _cl_event*;

constexpr cl_int CL_SUCCESS = 0;
constexpr cl_bool CL_TRUE = 1;
constexpr cl_device_type CL_DEVICE_TYPE_GPU = 1ULL << 2;
constexpr cl_device_type CL_DEVICE_TYPE_ALL = 0xFFFFFFFFULL;
constexpr cl_mem_flags CL_MEM_READ_ONLY = 1ULL << 2;
constexpr cl_mem_flags CL_MEM_WRITE_ONLY = 1ULL << 1;
constexpr cl_mem_flags CL_MEM_READ_WRITE = 1ULL << 0;
constexpr cl_mem_flags CL_MEM_COPY_HOST_PTR = 1ULL << 5;
constexpr cl_platform_info CL_PLATFORM_EXTENSIONS = 0x0904;
constexpr cl_device_info CL_DEVICE_EXTENSIONS = 0x1030;
constexpr cl_program_build_info CL_PROGRAM_BUILD_LOG = 0x1183;
constexpr intptr_t CL_IMPORT_TYPE_ARM = 0x40B2;
constexpr intptr_t CL_IMPORT_TYPE_DMA_BUF_ARM = 0x40B4;

extern "C" {
cl_int clGetPlatformIDs(cl_uint, cl_platform_id*, cl_uint*);
cl_int clGetDeviceIDs(cl_platform_id, cl_device_type, cl_uint, cl_device_id*, cl_uint*);
cl_int clGetPlatformInfo(cl_platform_id, cl_platform_info, size_t, void*, size_t*);
cl_int clGetDeviceInfo(cl_device_id, cl_device_info, size_t, void*, size_t*);
cl_context clCreateContext(const cl_context_properties*, cl_uint, const cl_device_id*,
                           void (*)(const char*, const void*, size_t, void*),
                           void*, cl_int*);
cl_command_queue clCreateCommandQueue(cl_context, cl_device_id, cl_command_queue_properties, cl_int*);
cl_program clCreateProgramWithSource(cl_context, cl_uint, const char**, const size_t*, cl_int*);
cl_int clBuildProgram(cl_program, cl_uint, const cl_device_id*, const char*,
                      void (*)(cl_program, void*), void*);
cl_int clGetProgramBuildInfo(cl_program, cl_device_id, cl_program_build_info, size_t, void*, size_t*);
cl_kernel clCreateKernel(cl_program, const char*, cl_int*);
cl_mem clCreateBuffer(cl_context, cl_mem_flags, size_t, void*, cl_int*);
cl_int clEnqueueWriteBuffer(cl_command_queue, cl_mem, cl_bool, size_t, size_t,
                            const void*, cl_uint, const cl_event*, cl_event*);
cl_int clSetKernelArg(cl_kernel, cl_uint, size_t, const void*);
cl_int clEnqueueNDRangeKernel(cl_command_queue, cl_kernel, cl_uint, const size_t*,
                              const size_t*, const size_t*, cl_uint, const cl_event*, cl_event*);
cl_int clFinish(cl_command_queue);
cl_int clEnqueueReadBuffer(cl_command_queue, cl_mem, cl_bool, size_t, size_t,
                           void*, cl_uint, const cl_event*, cl_event*);
void* clGetExtensionFunctionAddressForPlatform(cl_platform_id, const char*);
cl_int clReleaseMemObject(cl_mem);
cl_int clReleaseKernel(cl_kernel);
cl_int clReleaseProgram(cl_program);
cl_int clReleaseCommandQueue(cl_command_queue);
cl_int clReleaseContext(cl_context);
}

using clImportMemoryARMFn = cl_mem (*)(cl_context,
                                      cl_mem_flags,
                                      const cl_import_properties_arm*,
                                      void*,
                                      size_t,
                                      cl_int*);

namespace {

using Clock = std::chrono::steady_clock;

double elapsed_ms(const Clock::time_point& start, const Clock::time_point& end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

void set_error(char* err, int err_len, const std::string& msg) {
  if (err == nullptr || err_len <= 0) {
    return;
  }
  std::snprintf(err, static_cast<size_t>(err_len), "%s", msg.c_str());
}

std::string cl_error(const char* what, cl_int code) {
  std::ostringstream oss;
  oss << what << " failed: " << code;
  return oss.str();
}

const char* kKernelSource = R"CLC(
__kernel void direct_slice_fused(
    __global const uchar* frame,
    __global const float* map_x,
    __global const float* map_y,
    __global const int4* roi_rects,
    __global uchar* output,
    const int src_w,
    const int src_h,
    const int frame_stride,
    const int roi_h,
    const int roi_w,
    const int imgsz,
    const int out_slice_offset,
    const int slice_override)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int s = slice_override >= 0 ? slice_override : get_global_id(2);
    if (x >= imgsz || y >= imgsz) {
        return;
    }

    const int out_idx = (((s - out_slice_offset) * imgsz + y) * imgsz + x) * 3;
    const int4 roi = roi_rects[s]; // x=left, y=top, z=new_w, w=new_h

    uchar rgb0 = (uchar)114;
    uchar rgb1 = (uchar)114;
    uchar rgb2 = (uchar)114;

    if (x >= roi.x && x < roi.x + roi.z && y >= roi.y && y < roi.y + roi.w) {
        const int rx = x - roi.x;
        const int ry = y - roi.y;
        if (rx >= 0 && rx < roi_w && ry >= 0 && ry < roi_h) {
            const int map_idx = (s * roi_h + ry) * roi_w + rx;
            const float sx = map_x[map_idx];
            const float sy = map_y[map_idx];
            const int x0 = (int)floor(sx);
            const int y0 = (int)floor(sy);
            const float fx = sx - (float)x0;
            const float fy = sy - (float)y0;
            const int x1 = x0 + 1;
            const int y1 = y0 + 1;

            float b = 114.0f;
            float g = 114.0f;
            float r = 114.0f;
            float wsum = 0.0f;

            const float w00 = (1.0f - fx) * (1.0f - fy);
            const float w01 = fx * (1.0f - fy);
            const float w10 = (1.0f - fx) * fy;
            const float w11 = fx * fy;

            b = 114.0f * (w00 + w01 + w10 + w11);
            g = b;
            r = b;

            if (x0 >= 0 && x0 < src_w && y0 >= 0 && y0 < src_h) {
                const int p = y0 * frame_stride + x0 * 3;
                b += ((float)frame[p + 0] - 114.0f) * w00;
                g += ((float)frame[p + 1] - 114.0f) * w00;
                r += ((float)frame[p + 2] - 114.0f) * w00;
                wsum += w00;
            }
            if (x1 >= 0 && x1 < src_w && y0 >= 0 && y0 < src_h) {
                const int p = y0 * frame_stride + x1 * 3;
                b += ((float)frame[p + 0] - 114.0f) * w01;
                g += ((float)frame[p + 1] - 114.0f) * w01;
                r += ((float)frame[p + 2] - 114.0f) * w01;
                wsum += w01;
            }
            if (x0 >= 0 && x0 < src_w && y1 >= 0 && y1 < src_h) {
                const int p = y1 * frame_stride + x0 * 3;
                b += ((float)frame[p + 0] - 114.0f) * w10;
                g += ((float)frame[p + 1] - 114.0f) * w10;
                r += ((float)frame[p + 2] - 114.0f) * w10;
                wsum += w10;
            }
            if (x1 >= 0 && x1 < src_w && y1 >= 0 && y1 < src_h) {
                const int p = y1 * frame_stride + x1 * 3;
                b += ((float)frame[p + 0] - 114.0f) * w11;
                g += ((float)frame[p + 1] - 114.0f) * w11;
                r += ((float)frame[p + 2] - 114.0f) * w11;
                wsum += w11;
            }

            // OpenCV remap with BORDER_CONSTANT uses constant samples for any
            // out-of-frame neighbors. The formula above starts from 114 and
            // adds only the delta of valid neighbors, matching that behavior.
            rgb0 = convert_uchar_sat_rte(r);
            rgb1 = convert_uchar_sat_rte(g);
            rgb2 = convert_uchar_sat_rte(b);
        }
    }

    output[out_idx + 0] = rgb0;
    output[out_idx + 1] = rgb1;
    output[out_idx + 2] = rgb2;
}
)CLC";

struct DirectSliceOpenCLFused {
  int src_w = 0;
  int src_h = 0;
  int frame_stride = 0;
  int num_slices = 0;
  int roi_h = 0;
  int roi_w = 0;
  int imgsz = 0;
  size_t frame_bytes = 0;
  size_t output_bytes = 0;

  cl_platform_id platform = nullptr;
  cl_device_id device = nullptr;
  cl_context context = nullptr;
  cl_command_queue queue = nullptr;
  cl_program program = nullptr;
  cl_kernel kernel = nullptr;
  cl_mem frame_buf = nullptr;
  cl_mem map_x_buf = nullptr;
  cl_mem map_y_buf = nullptr;
  cl_mem roi_buf = nullptr;
  cl_mem output_buf = nullptr;
  clImportMemoryARMFn import_memory_arm = nullptr;
  std::vector<cl_mem> imported_outputs;
  std::vector<int> imported_fds;
  std::vector<uint64_t> imported_sizes;

  ~DirectSliceOpenCLFused() {
    for (cl_mem mem : imported_outputs) {
      if (mem) clReleaseMemObject(mem);
    }
    if (output_buf) clReleaseMemObject(output_buf);
    if (roi_buf) clReleaseMemObject(roi_buf);
    if (map_y_buf) clReleaseMemObject(map_y_buf);
    if (map_x_buf) clReleaseMemObject(map_x_buf);
    if (frame_buf) clReleaseMemObject(frame_buf);
    if (kernel) clReleaseKernel(kernel);
    if (program) clReleaseProgram(program);
    if (queue) clReleaseCommandQueue(queue);
    if (context) clReleaseContext(context);
  }
};

std::string get_platform_info_string(cl_platform_id platform, cl_platform_info name) {
  size_t size = 0;
  cl_int ret = clGetPlatformInfo(platform, name, 0, nullptr, &size);
  if (ret != CL_SUCCESS || size == 0) {
    return std::string();
  }
  std::vector<char> buf(size + 1, '\0');
  ret = clGetPlatformInfo(platform, name, size, buf.data(), nullptr);
  if (ret != CL_SUCCESS) {
    return std::string();
  }
  return std::string(buf.data());
}

std::string get_device_info_string(cl_device_id device, cl_device_info name) {
  size_t size = 0;
  cl_int ret = clGetDeviceInfo(device, name, 0, nullptr, &size);
  if (ret != CL_SUCCESS || size == 0) {
    return std::string();
  }
  std::vector<char> buf(size + 1, '\0');
  ret = clGetDeviceInfo(device, name, size, buf.data(), nullptr);
  if (ret != CL_SUCCESS) {
    return std::string();
  }
  return std::string(buf.data());
}

bool pick_device(cl_platform_id* out_platform, cl_device_id* out_device, std::string* error) {
  cl_uint num_platforms = 0;
  cl_int ret = clGetPlatformIDs(0, nullptr, &num_platforms);
  if (ret != CL_SUCCESS || num_platforms == 0) {
    *error = cl_error("clGetPlatformIDs", ret);
    return false;
  }

  std::vector<cl_platform_id> platforms(num_platforms);
  ret = clGetPlatformIDs(num_platforms, platforms.data(), nullptr);
  if (ret != CL_SUCCESS) {
    *error = cl_error("clGetPlatformIDs(list)", ret);
    return false;
  }

  for (cl_platform_id platform : platforms) {
    cl_uint num_devices = 0;
    ret = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices);
    if (ret == CL_SUCCESS && num_devices > 0) {
      std::vector<cl_device_id> devices(num_devices);
      ret = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, num_devices, devices.data(), nullptr);
      if (ret == CL_SUCCESS) {
        *out_platform = platform;
        *out_device = devices[0];
        return true;
      }
    }
  }

  for (cl_platform_id platform : platforms) {
    cl_uint num_devices = 0;
    ret = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &num_devices);
    if (ret == CL_SUCCESS && num_devices > 0) {
      std::vector<cl_device_id> devices(num_devices);
      ret = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, num_devices, devices.data(), nullptr);
      if (ret == CL_SUCCESS) {
        *out_platform = platform;
        *out_device = devices[0];
        return true;
      }
    }
  }

  *error = "no OpenCL device found";
  return false;
}

bool build_program(DirectSliceOpenCLFused* state, std::string* error) {
  cl_int ret = CL_SUCCESS;
  const char* sources[] = {kKernelSource};
  const size_t lengths[] = {std::strlen(kKernelSource)};
  state->program = clCreateProgramWithSource(state->context, 1, sources, lengths, &ret);
  if (ret != CL_SUCCESS) {
    *error = cl_error("clCreateProgramWithSource", ret);
    return false;
  }

  ret = clBuildProgram(state->program, 1, &state->device, "", nullptr, nullptr);
  if (ret != CL_SUCCESS) {
    size_t log_size = 0;
    clGetProgramBuildInfo(state->program, state->device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
    std::string log(log_size, '\0');
    if (log_size > 0) {
      clGetProgramBuildInfo(state->program, state->device, CL_PROGRAM_BUILD_LOG, log_size, &log[0], nullptr);
    }
    std::ostringstream oss;
    oss << "clBuildProgram failed: " << ret << "\n" << log;
    *error = oss.str();
    return false;
  }

  state->kernel = clCreateKernel(state->program, "direct_slice_fused", &ret);
  if (ret != CL_SUCCESS) {
    *error = cl_error("clCreateKernel", ret);
    return false;
  }
  return true;
}

}  // namespace

extern "C" {

int ds_opencl_fused_create(const float* map_x,
                           const float* map_y,
                           const int* roi_rects,
                           int src_w,
                           int src_h,
                           int frame_stride,
                           int num_slices,
                           int roi_h,
                           int roi_w,
                           int imgsz,
                           void** out_handle,
                           char* err,
                           int err_len) {
  if (!map_x || !map_y || !roi_rects || !out_handle) {
    set_error(err, err_len, "null argument");
    return -1;
  }
  if (src_w <= 0 || src_h <= 0 || frame_stride < src_w * 3 ||
      num_slices <= 0 || roi_h <= 0 || roi_w <= 0 || imgsz <= 0) {
    set_error(err, err_len, "invalid shape argument");
    return -2;
  }

  std::string error;
  DirectSliceOpenCLFused* state = new DirectSliceOpenCLFused();
  state->src_w = src_w;
  state->src_h = src_h;
  state->frame_stride = frame_stride;
  state->num_slices = num_slices;
  state->roi_h = roi_h;
  state->roi_w = roi_w;
  state->imgsz = imgsz;
  state->frame_bytes = static_cast<size_t>(src_h) * static_cast<size_t>(frame_stride);
  state->output_bytes = static_cast<size_t>(num_slices) * imgsz * imgsz * 3;

  if (!pick_device(&state->platform, &state->device, &error)) {
    delete state;
    set_error(err, err_len, error);
    return -3;
  }

  cl_int ret = CL_SUCCESS;
  state->context = clCreateContext(nullptr, 1, &state->device, nullptr, nullptr, &ret);
  if (ret != CL_SUCCESS) {
    delete state;
    set_error(err, err_len, cl_error("clCreateContext", ret));
    return -4;
  }

  state->queue = clCreateCommandQueue(state->context, state->device, 0, &ret);
  if (ret != CL_SUCCESS) {
    delete state;
    set_error(err, err_len, cl_error("clCreateCommandQueue", ret));
    return -5;
  }
  const std::string platform_ext = get_platform_info_string(state->platform, CL_PLATFORM_EXTENSIONS);
  const std::string device_ext = get_device_info_string(state->device, CL_DEVICE_EXTENSIONS);
  const bool has_import_memory =
      platform_ext.find("cl_arm_import_memory") != std::string::npos ||
      device_ext.find("cl_arm_import_memory") != std::string::npos;
  const bool has_import_dmabuf =
      platform_ext.find("cl_arm_import_memory_dma_buf") != std::string::npos ||
      device_ext.find("cl_arm_import_memory_dma_buf") != std::string::npos;
  if (has_import_memory && has_import_dmabuf) {
    state->import_memory_arm = reinterpret_cast<clImportMemoryARMFn>(
        clGetExtensionFunctionAddressForPlatform(state->platform, "clImportMemoryARM"));
  }

  if (!build_program(state, &error)) {
    delete state;
    set_error(err, err_len, error);
    return -6;
  }

  const size_t map_bytes = static_cast<size_t>(num_slices) * roi_h * roi_w * sizeof(float);
  const size_t roi_bytes = static_cast<size_t>(num_slices) * 4 * sizeof(int);
  state->frame_buf = clCreateBuffer(state->context, CL_MEM_READ_ONLY, state->frame_bytes, nullptr, &ret);
  if (ret != CL_SUCCESS) {
    delete state;
    set_error(err, err_len, cl_error("clCreateBuffer(frame)", ret));
    return -7;
  }
  state->map_x_buf = clCreateBuffer(state->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    map_bytes, const_cast<float*>(map_x), &ret);
  if (ret != CL_SUCCESS) {
    delete state;
    set_error(err, err_len, cl_error("clCreateBuffer(map_x)", ret));
    return -8;
  }
  state->map_y_buf = clCreateBuffer(state->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    map_bytes, const_cast<float*>(map_y), &ret);
  if (ret != CL_SUCCESS) {
    delete state;
    set_error(err, err_len, cl_error("clCreateBuffer(map_y)", ret));
    return -9;
  }
  state->roi_buf = clCreateBuffer(state->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  roi_bytes, const_cast<int*>(roi_rects), &ret);
  if (ret != CL_SUCCESS) {
    delete state;
    set_error(err, err_len, cl_error("clCreateBuffer(roi)", ret));
    return -10;
  }
  state->output_buf = clCreateBuffer(state->context, CL_MEM_WRITE_ONLY, state->output_bytes, nullptr, &ret);
  if (ret != CL_SUCCESS) {
    delete state;
    set_error(err, err_len, cl_error("clCreateBuffer(output)", ret));
    return -11;
  }

  *out_handle = state;
  return 0;
}

int ds_opencl_fused_run(void* handle,
                        const uint8_t* frame,
                        uint8_t* output,
                        double* timings,
                        char* err,
                        int err_len) {
  if (!handle || !frame || !output) {
    set_error(err, err_len, "null argument");
    return -1;
  }
  DirectSliceOpenCLFused* state = static_cast<DirectSliceOpenCLFused*>(handle);
  cl_int ret = CL_SUCCESS;
  auto total_start = Clock::now();

  auto upload_start = Clock::now();
  ret = clEnqueueWriteBuffer(
      state->queue,
      state->frame_buf,
      CL_TRUE,
      0,
      state->frame_bytes,
      frame,
      0,
      nullptr,
      nullptr);
  auto upload_end = Clock::now();
  if (ret != CL_SUCCESS) {
    set_error(err, err_len, cl_error("clEnqueueWriteBuffer(frame)", ret));
    return -2;
  }

  int arg = 0;
  ret  = clSetKernelArg(state->kernel, arg++, sizeof(cl_mem), &state->frame_buf);
  ret |= clSetKernelArg(state->kernel, arg++, sizeof(cl_mem), &state->map_x_buf);
  ret |= clSetKernelArg(state->kernel, arg++, sizeof(cl_mem), &state->map_y_buf);
  ret |= clSetKernelArg(state->kernel, arg++, sizeof(cl_mem), &state->roi_buf);
  ret |= clSetKernelArg(state->kernel, arg++, sizeof(cl_mem), &state->output_buf);
  ret |= clSetKernelArg(state->kernel, arg++, sizeof(int), &state->src_w);
  ret |= clSetKernelArg(state->kernel, arg++, sizeof(int), &state->src_h);
  ret |= clSetKernelArg(state->kernel, arg++, sizeof(int), &state->frame_stride);
  ret |= clSetKernelArg(state->kernel, arg++, sizeof(int), &state->roi_h);
  ret |= clSetKernelArg(state->kernel, arg++, sizeof(int), &state->roi_w);
  ret |= clSetKernelArg(state->kernel, arg++, sizeof(int), &state->imgsz);
  int out_slice_offset = 0;
  ret |= clSetKernelArg(state->kernel, arg++, sizeof(int), &out_slice_offset);
  int slice_override = -1;
  ret |= clSetKernelArg(state->kernel, arg++, sizeof(int), &slice_override);
  if (ret != CL_SUCCESS) {
    set_error(err, err_len, cl_error("clSetKernelArg", ret));
    return -3;
  }

  const size_t global[3] = {
      static_cast<size_t>(state->imgsz),
      static_cast<size_t>(state->imgsz),
      static_cast<size_t>(state->num_slices),
  };
  auto kernel_start = Clock::now();
  ret = clEnqueueNDRangeKernel(state->queue, state->kernel, 3, nullptr, global, nullptr, 0, nullptr, nullptr);
  if (ret != CL_SUCCESS) {
    set_error(err, err_len, cl_error("clEnqueueNDRangeKernel", ret));
    return -4;
  }
  ret = clFinish(state->queue);
  auto kernel_end = Clock::now();
  if (ret != CL_SUCCESS) {
    set_error(err, err_len, cl_error("clFinish(kernel)", ret));
    return -5;
  }

  auto read_start = Clock::now();
  ret = clEnqueueReadBuffer(
      state->queue,
      state->output_buf,
      CL_TRUE,
      0,
      state->output_bytes,
      output,
      0,
      nullptr,
      nullptr);
  auto read_end = Clock::now();
  if (ret != CL_SUCCESS) {
    set_error(err, err_len, cl_error("clEnqueueReadBuffer(output)", ret));
    return -6;
  }

  auto total_end = Clock::now();
  if (timings != nullptr) {
    timings[0] = elapsed_ms(upload_start, upload_end);
    timings[1] = elapsed_ms(kernel_start, kernel_end);
    timings[2] = elapsed_ms(read_start, read_end);
    timings[3] = elapsed_ms(total_start, total_end);
  }
  return 0;
}

int ds_opencl_fused_run_split(void* handle,
                              const uint8_t* frame,
                              uint8_t** outputs,
                              int num_outputs,
                              double* timings,
                              char* err,
                              int err_len) {
  if (!handle || !frame || !outputs) {
    set_error(err, err_len, "null argument");
    return -1;
  }
  DirectSliceOpenCLFused* state = static_cast<DirectSliceOpenCLFused*>(handle);
  if (num_outputs != state->num_slices) {
    set_error(err, err_len, "num_outputs must match num_slices");
    return -1;
  }
  const size_t slice_bytes = static_cast<size_t>(state->imgsz) * state->imgsz * 3;
  for (int i = 0; i < num_outputs; ++i) {
    if (outputs[i] == nullptr) {
      set_error(err, err_len, "null output slice");
      return -1;
    }
  }

  cl_int ret = CL_SUCCESS;
  auto total_start = Clock::now();

  auto upload_start = Clock::now();
  ret = clEnqueueWriteBuffer(
      state->queue,
      state->frame_buf,
      CL_TRUE,
      0,
      state->frame_bytes,
      frame,
      0,
      nullptr,
      nullptr);
  auto upload_end = Clock::now();
  if (ret != CL_SUCCESS) {
    set_error(err, err_len, cl_error("clEnqueueWriteBuffer(frame)", ret));
    return -2;
  }

  int arg = 0;
  ret  = clSetKernelArg(state->kernel, arg++, sizeof(cl_mem), &state->frame_buf);
  ret |= clSetKernelArg(state->kernel, arg++, sizeof(cl_mem), &state->map_x_buf);
  ret |= clSetKernelArg(state->kernel, arg++, sizeof(cl_mem), &state->map_y_buf);
  ret |= clSetKernelArg(state->kernel, arg++, sizeof(cl_mem), &state->roi_buf);
  ret |= clSetKernelArg(state->kernel, arg++, sizeof(cl_mem), &state->output_buf);
  ret |= clSetKernelArg(state->kernel, arg++, sizeof(int), &state->src_w);
  ret |= clSetKernelArg(state->kernel, arg++, sizeof(int), &state->src_h);
  ret |= clSetKernelArg(state->kernel, arg++, sizeof(int), &state->frame_stride);
  ret |= clSetKernelArg(state->kernel, arg++, sizeof(int), &state->roi_h);
  ret |= clSetKernelArg(state->kernel, arg++, sizeof(int), &state->roi_w);
  ret |= clSetKernelArg(state->kernel, arg++, sizeof(int), &state->imgsz);
  int out_slice_offset = 0;
  ret |= clSetKernelArg(state->kernel, arg++, sizeof(int), &out_slice_offset);
  int slice_override = -1;
  ret |= clSetKernelArg(state->kernel, arg++, sizeof(int), &slice_override);
  if (ret != CL_SUCCESS) {
    set_error(err, err_len, cl_error("clSetKernelArg", ret));
    return -3;
  }

  const size_t global[3] = {
      static_cast<size_t>(state->imgsz),
      static_cast<size_t>(state->imgsz),
      static_cast<size_t>(state->num_slices),
  };
  auto kernel_start = Clock::now();
  ret = clEnqueueNDRangeKernel(state->queue, state->kernel, 3, nullptr, global, nullptr, 0, nullptr, nullptr);
  if (ret != CL_SUCCESS) {
    set_error(err, err_len, cl_error("clEnqueueNDRangeKernel", ret));
    return -4;
  }
  ret = clFinish(state->queue);
  auto kernel_end = Clock::now();
  if (ret != CL_SUCCESS) {
    set_error(err, err_len, cl_error("clFinish(kernel)", ret));
    return -5;
  }

  auto read_start = Clock::now();
  for (int i = 0; i < num_outputs; ++i) {
    ret = clEnqueueReadBuffer(
        state->queue,
        state->output_buf,
        CL_TRUE,
        static_cast<size_t>(i) * slice_bytes,
        slice_bytes,
        outputs[i],
        0,
        nullptr,
        nullptr);
    if (ret != CL_SUCCESS) {
      set_error(err, err_len, cl_error("clEnqueueReadBuffer(output slice)", ret));
      return -6;
    }
  }
  auto read_end = Clock::now();

  auto total_end = Clock::now();
  if (timings != nullptr) {
    timings[0] = elapsed_ms(upload_start, upload_end);
    timings[1] = elapsed_ms(kernel_start, kernel_end);
    timings[2] = elapsed_ms(read_start, read_end);
    timings[3] = elapsed_ms(total_start, total_end);
  }
  return 0;
}

int ds_opencl_fused_import_output_fds(void* handle,
                                      const int* fds,
                                      const uint64_t* sizes,
                                      int num_outputs,
                                      char* err,
                                      int err_len) {
  if (!handle || !fds || !sizes) {
    set_error(err, err_len, "null argument");
    return -1;
  }
  DirectSliceOpenCLFused* state = static_cast<DirectSliceOpenCLFused*>(handle);
  if (num_outputs != state->num_slices) {
    set_error(err, err_len, "num_outputs must match num_slices");
    return -2;
  }
  if (state->import_memory_arm == nullptr) {
    set_error(err, err_len, "OpenCL cl_arm_import_memory_dma_buf is unavailable");
    return -3;
  }
  const uint64_t slice_bytes = static_cast<uint64_t>(state->imgsz) * state->imgsz * 3;
  for (int i = 0; i < num_outputs; ++i) {
    if (fds[i] < 0) {
      set_error(err, err_len, "invalid fd");
      return -4;
    }
    if (sizes[i] < slice_bytes) {
      set_error(err, err_len, "import fd size is smaller than one slice");
      return -5;
    }
  }

  if (state->imported_outputs.size() == static_cast<size_t>(num_outputs) &&
      state->imported_fds.size() == static_cast<size_t>(num_outputs) &&
      state->imported_sizes.size() == static_cast<size_t>(num_outputs)) {
    bool same = true;
    for (int i = 0; i < num_outputs; ++i) {
      same = same && state->imported_fds[static_cast<size_t>(i)] == fds[i] &&
             state->imported_sizes[static_cast<size_t>(i)] == sizes[i];
    }
    if (same) {
      return 0;
    }
  }

  for (cl_mem mem : state->imported_outputs) {
    if (mem) clReleaseMemObject(mem);
  }
  state->imported_outputs.clear();
  state->imported_fds.clear();
  state->imported_sizes.clear();

  const cl_import_properties_arm props[] = {
      CL_IMPORT_TYPE_ARM,
      CL_IMPORT_TYPE_DMA_BUF_ARM,
      0,
  };
  for (int i = 0; i < num_outputs; ++i) {
    int fd_value = fds[i];
    cl_int ret = CL_SUCCESS;
    cl_mem imported = state->import_memory_arm(
        state->context,
        CL_MEM_READ_WRITE,
        props,
        &fd_value,
        static_cast<size_t>(sizes[i]),
        &ret);
    if (ret != CL_SUCCESS || imported == nullptr) {
      std::ostringstream oss;
      oss << "clImportMemoryARM(output fd " << i << ") failed: " << ret;
      set_error(err, err_len, oss.str());
      for (cl_mem mem : state->imported_outputs) {
        if (mem) clReleaseMemObject(mem);
      }
      state->imported_outputs.clear();
      state->imported_fds.clear();
      state->imported_sizes.clear();
      return -6;
    }
    state->imported_outputs.push_back(imported);
    state->imported_fds.push_back(fds[i]);
    state->imported_sizes.push_back(sizes[i]);
  }
  return 0;
}

int ds_opencl_fused_run_imported(void* handle,
                                 const uint8_t* frame,
                                 double* timings,
                                 char* err,
                                 int err_len) {
  if (!handle || !frame) {
    set_error(err, err_len, "null argument");
    return -1;
  }
  DirectSliceOpenCLFused* state = static_cast<DirectSliceOpenCLFused*>(handle);
  if (state->imported_outputs.size() != static_cast<size_t>(state->num_slices)) {
    set_error(err, err_len, "imported outputs are not prepared");
    return -2;
  }

  cl_int ret = CL_SUCCESS;
  auto total_start = Clock::now();

  auto upload_start = Clock::now();
  ret = clEnqueueWriteBuffer(
      state->queue,
      state->frame_buf,
      CL_TRUE,
      0,
      state->frame_bytes,
      frame,
      0,
      nullptr,
      nullptr);
  auto upload_end = Clock::now();
  if (ret != CL_SUCCESS) {
    set_error(err, err_len, cl_error("clEnqueueWriteBuffer(frame)", ret));
    return -3;
  }

  auto kernel_start = Clock::now();
  for (int slice = 0; slice < state->num_slices; ++slice) {
    cl_mem out_mem = state->imported_outputs[static_cast<size_t>(slice)];
    int arg = 0;
    ret  = clSetKernelArg(state->kernel, arg++, sizeof(cl_mem), &state->frame_buf);
    ret |= clSetKernelArg(state->kernel, arg++, sizeof(cl_mem), &state->map_x_buf);
    ret |= clSetKernelArg(state->kernel, arg++, sizeof(cl_mem), &state->map_y_buf);
    ret |= clSetKernelArg(state->kernel, arg++, sizeof(cl_mem), &state->roi_buf);
    ret |= clSetKernelArg(state->kernel, arg++, sizeof(cl_mem), &out_mem);
    ret |= clSetKernelArg(state->kernel, arg++, sizeof(int), &state->src_w);
    ret |= clSetKernelArg(state->kernel, arg++, sizeof(int), &state->src_h);
    ret |= clSetKernelArg(state->kernel, arg++, sizeof(int), &state->frame_stride);
    ret |= clSetKernelArg(state->kernel, arg++, sizeof(int), &state->roi_h);
    ret |= clSetKernelArg(state->kernel, arg++, sizeof(int), &state->roi_w);
    ret |= clSetKernelArg(state->kernel, arg++, sizeof(int), &state->imgsz);
    int out_slice_offset = slice;
    ret |= clSetKernelArg(state->kernel, arg++, sizeof(int), &out_slice_offset);
    int slice_override = slice;
    ret |= clSetKernelArg(state->kernel, arg++, sizeof(int), &slice_override);
    if (ret != CL_SUCCESS) {
      set_error(err, err_len, cl_error("clSetKernelArg(imported)", ret));
      return -4;
    }

    const size_t global[3] = {
        static_cast<size_t>(state->imgsz),
        static_cast<size_t>(state->imgsz),
        static_cast<size_t>(1),
    };
    ret = clEnqueueNDRangeKernel(state->queue, state->kernel, 3, nullptr, global, nullptr, 0, nullptr, nullptr);
    if (ret != CL_SUCCESS) {
      set_error(err, err_len, cl_error("clEnqueueNDRangeKernel(imported)", ret));
      return -5;
    }
  }
  ret = clFinish(state->queue);
  auto kernel_end = Clock::now();
  if (ret != CL_SUCCESS) {
    set_error(err, err_len, cl_error("clFinish(imported kernel)", ret));
    return -6;
  }

  auto total_end = Clock::now();
  if (timings != nullptr) {
    timings[0] = elapsed_ms(upload_start, upload_end);
    timings[1] = elapsed_ms(kernel_start, kernel_end);
    timings[2] = 0.0;
    timings[3] = elapsed_ms(total_start, total_end);
  }
  return 0;
}

void ds_opencl_fused_destroy(void* handle) {
  DirectSliceOpenCLFused* state = static_cast<DirectSliceOpenCLFused*>(handle);
  delete state;
}

}  // extern "C"
