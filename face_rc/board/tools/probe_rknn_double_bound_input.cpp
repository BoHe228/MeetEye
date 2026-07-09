// Probe whether one RKNN context can use two input memories as a safe double
// buffer, and whether OpenCL can write the next buffer while RKNN is running
// the current buffer.

#include "rknn_api.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using cl_uint = uint32_t;
using cl_int = int32_t;
using cl_ulong = uint64_t;
using cl_device_type = cl_ulong;
using cl_mem_flags = cl_ulong;
using cl_command_queue_properties = cl_ulong;
using cl_context_properties = intptr_t;
using cl_program_build_info = cl_uint;
using cl_mem_properties_arm = intptr_t;
using cl_import_properties_arm = intptr_t;

struct _cl_platform_id;
struct _cl_device_id;
struct _cl_context;
struct _cl_command_queue;
struct _cl_program;
struct _cl_kernel;
struct _cl_mem;

using cl_platform_id = _cl_platform_id*;
using cl_device_id = _cl_device_id*;
using cl_context = _cl_context*;
using cl_command_queue = _cl_command_queue*;
using cl_program = _cl_program*;
using cl_kernel = _cl_kernel*;
using cl_mem = _cl_mem*;

constexpr cl_int CL_SUCCESS = 0;
constexpr cl_device_type CL_DEVICE_TYPE_GPU = 1ULL << 2;
constexpr cl_device_type CL_DEVICE_TYPE_ALL = 0xFFFFFFFFULL;
constexpr cl_mem_flags CL_MEM_READ_WRITE = 1ULL << 0;
constexpr cl_uint CL_PLATFORM_EXTENSIONS = 0x0904;
constexpr cl_uint CL_DEVICE_EXTENSIONS = 0x1030;
constexpr cl_uint CL_PROGRAM_BUILD_LOG = 0x1183;
constexpr cl_mem_properties_arm CL_IMPORT_TYPE_ARM = 0x40B2;
constexpr cl_mem_properties_arm CL_IMPORT_TYPE_DMA_BUF_ARM = 0x40B4;

extern "C" {
cl_int clGetPlatformIDs(cl_uint, cl_platform_id*, cl_uint*);
cl_int clGetDeviceIDs(cl_platform_id, cl_device_type, cl_uint, cl_device_id*, cl_uint*);
cl_int clGetPlatformInfo(cl_platform_id, cl_uint, size_t, void*, size_t*);
cl_int clGetDeviceInfo(cl_device_id, cl_uint, size_t, void*, size_t*);
cl_context clCreateContext(const cl_context_properties*, cl_uint, const cl_device_id*,
                           void (*)(const char*, const void*, size_t, void*),
                           void*, cl_int*);
cl_command_queue clCreateCommandQueue(cl_context, cl_device_id, cl_command_queue_properties, cl_int*);
void* clGetExtensionFunctionAddressForPlatform(cl_platform_id, const char*);
cl_program clCreateProgramWithSource(cl_context, cl_uint, const char**, const size_t*, cl_int*);
cl_int clBuildProgram(cl_program, cl_uint, const cl_device_id*, const char*,
                      void (*)(cl_program, void*), void*);
cl_int clGetProgramBuildInfo(cl_program, cl_device_id, cl_program_build_info, size_t, void*, size_t*);
cl_kernel clCreateKernel(cl_program, const char*, cl_int*);
cl_int clSetKernelArg(cl_kernel, cl_uint, size_t, const void*);
cl_int clEnqueueNDRangeKernel(cl_command_queue, cl_kernel, cl_uint, const size_t*,
                              const size_t*, const size_t*, cl_uint, const void*, void*);
cl_int clFinish(cl_command_queue);
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

const char* kFillKernel = R"CLC(
__kernel void fill_imported(__global uchar* dst, const uint n, const uchar value)
{
    const uint i = get_global_id(0);
    if (i < n) {
        dst[i] = (uchar)(value + (uchar)(i & 15));
    }
}
)CLC";

static int64_t now_us() {
  using clock = std::chrono::steady_clock;
  return std::chrono::duration_cast<std::chrono::microseconds>(
             clock::now().time_since_epoch())
      .count();
}

static std::string cl_err(const char* what, cl_int code) {
  char buf[128];
  const char* name = "UNKNOWN";
  switch (code) {
    case 0: name = "CL_SUCCESS"; break;
    case -1: name = "CL_DEVICE_NOT_FOUND"; break;
    case -2: name = "CL_DEVICE_NOT_AVAILABLE"; break;
    case -4: name = "CL_MEM_OBJECT_ALLOCATION_FAILURE"; break;
    case -5: name = "CL_OUT_OF_RESOURCES"; break;
    case -6: name = "CL_OUT_OF_HOST_MEMORY"; break;
    case -30: name = "CL_INVALID_VALUE"; break;
    case -34: name = "CL_INVALID_CONTEXT"; break;
    case -61: name = "CL_INVALID_BUFFER_SIZE"; break;
    default: break;
  }
  std::snprintf(buf, sizeof(buf), "%s failed: %d (%s)", what, code, name);
  return std::string(buf);
}

static bool get_info_string_platform(cl_platform_id platform, cl_uint name, std::string* out) {
  size_t size = 0;
  cl_int ret = clGetPlatformInfo(platform, name, 0, nullptr, &size);
  if (ret != CL_SUCCESS || size == 0) return false;
  std::vector<char> buf(size + 1, '\0');
  ret = clGetPlatformInfo(platform, name, size, buf.data(), nullptr);
  if (ret != CL_SUCCESS) return false;
  *out = buf.data();
  return true;
}

static bool get_info_string_device(cl_device_id device, cl_uint name, std::string* out) {
  size_t size = 0;
  cl_int ret = clGetDeviceInfo(device, name, 0, nullptr, &size);
  if (ret != CL_SUCCESS || size == 0) return false;
  std::vector<char> buf(size + 1, '\0');
  ret = clGetDeviceInfo(device, name, size, buf.data(), nullptr);
  if (ret != CL_SUCCESS) return false;
  *out = buf.data();
  return true;
}

static bool pick_opencl_device(cl_platform_id* out_platform,
                               cl_device_id* out_device,
                               std::string* error) {
  cl_uint num_platforms = 0;
  cl_int ret = clGetPlatformIDs(0, nullptr, &num_platforms);
  if (ret != CL_SUCCESS || num_platforms == 0) {
    *error = cl_err("clGetPlatformIDs", ret);
    return false;
  }
  std::vector<cl_platform_id> platforms(num_platforms);
  ret = clGetPlatformIDs(num_platforms, platforms.data(), nullptr);
  if (ret != CL_SUCCESS) {
    *error = cl_err("clGetPlatformIDs(list)", ret);
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

static bool query_native_input_attr(rknn_context ctx, rknn_tensor_attr* attr, std::string* error) {
  std::memset(attr, 0, sizeof(*attr));
  attr->index = 0;
  int ret = rknn_query(ctx, RKNN_QUERY_NATIVE_NHWC_INPUT_ATTR, attr, sizeof(*attr));
  if (ret != RKNN_SUCC) {
    std::memset(attr, 0, sizeof(*attr));
    attr->index = 0;
    ret = rknn_query(ctx, RKNN_QUERY_NATIVE_INPUT_ATTR, attr, sizeof(*attr));
  }
  if (ret != RKNN_SUCC) {
    char buf[128];
    std::snprintf(buf, sizeof(buf), "RKNN_QUERY_NATIVE_INPUT_ATTR failed: %d", ret);
    *error = buf;
    return false;
  }
  attr->index = 0;
  attr->type = RKNN_TENSOR_UINT8;
  attr->fmt = RKNN_TENSOR_NHWC;
  attr->pass_through = 0;
  attr->h_stride = 0;
  return true;
}

static bool build_fill_kernel(cl_context context,
                              cl_device_id device,
                              cl_program* out_program,
                              cl_kernel* out_kernel,
                              std::string* error) {
  const char* src = kFillKernel;
  const size_t len = std::strlen(kFillKernel);
  cl_int ret = CL_SUCCESS;
  cl_program program = clCreateProgramWithSource(context, 1, &src, &len, &ret);
  if (ret != CL_SUCCESS || program == nullptr) {
    *error = cl_err("clCreateProgramWithSource", ret);
    return false;
  }
  ret = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
  if (ret != CL_SUCCESS) {
    size_t log_size = 0;
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
    std::vector<char> log(log_size + 1, '\0');
    if (log_size > 0) {
      clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
    }
    *error = cl_err("clBuildProgram", ret) + "\n" + std::string(log.data());
    clReleaseProgram(program);
    return false;
  }
  cl_kernel kernel = clCreateKernel(program, "fill_imported", &ret);
  if (ret != CL_SUCCESS || kernel == nullptr) {
    *error = cl_err("clCreateKernel", ret);
    clReleaseProgram(program);
    return false;
  }
  *out_program = program;
  *out_kernel = kernel;
  return true;
}

static rknn_tensor_mem* create_input_mem(rknn_context ctx,
                                         uint32_t size,
                                         const char* name) {
  rknn_tensor_mem* mem = rknn_create_mem(ctx, size);
  if (mem == nullptr || mem->virt_addr == nullptr || mem->fd < 0) {
    std::fprintf(stderr,
                 "[fail] rknn_create_mem(%s) failed or has no fd: mem=%p virt=%p fd=%d\n",
                 name,
                 static_cast<void*>(mem),
                 mem ? mem->virt_addr : nullptr,
                 mem ? mem->fd : -1);
    return nullptr;
  }
  std::printf("[ok] %s: size=%u virt=%p fd=%d phys=0x%llx\n",
              name,
              mem->size,
              mem->virt_addr,
              mem->fd,
              static_cast<unsigned long long>(mem->phys_addr));
  return mem;
}

static bool import_mem(cl_context clctx,
                       clImportMemoryARMFn import_fn,
                       rknn_tensor_mem* mem,
                       cl_mem* out,
                       const char* name,
                       std::string* error) {
  const cl_import_properties_arm props[] = {
      CL_IMPORT_TYPE_ARM,
      CL_IMPORT_TYPE_DMA_BUF_ARM,
      0,
  };
  int fd_value = mem->fd;
  cl_int ret = CL_SUCCESS;
  cl_mem imported = import_fn(clctx, CL_MEM_READ_WRITE, props, &fd_value, mem->size, &ret);
  if (ret != CL_SUCCESS || imported == nullptr) {
    *error = cl_err("clImportMemoryARM", ret);
    return false;
  }
  std::printf("[ok] imported %s fd=%d as OpenCL cl_mem\n", name, mem->fd);
  *out = imported;
  return true;
}

static bool fill_imported(cl_command_queue queue,
                          cl_kernel kernel,
                          cl_mem imported,
                          uint32_t n,
                          uint8_t value,
                          const char* name,
                          double* elapsed_ms,
                          std::string* error) {
  cl_int ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), &imported);
  ret |= clSetKernelArg(kernel, 1, sizeof(cl_uint), &n);
  ret |= clSetKernelArg(kernel, 2, sizeof(uint8_t), &value);
  if (ret != CL_SUCCESS) {
    *error = cl_err("clSetKernelArg", ret);
    return false;
  }
  const size_t global = ((static_cast<size_t>(n) + 255) / 256) * 256;
  const int64_t t0 = now_us();
  ret = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global, nullptr, 0, nullptr, nullptr);
  if (ret != CL_SUCCESS) {
    *error = cl_err("clEnqueueNDRangeKernel", ret);
    return false;
  }
  ret = clFinish(queue);
  const int64_t t1 = now_us();
  if (ret != CL_SUCCESS) {
    *error = cl_err("clFinish", ret);
    return false;
  }
  *elapsed_ms = (t1 - t0) / 1000.0;
  std::printf("[ok] OpenCL filled %s value=0x%02x bytes=%u time=%.3f ms\n",
              name, static_cast<unsigned>(value), n, *elapsed_ms);
  return true;
}

static bool sync_from_device(rknn_context ctx, rknn_tensor_mem* mem, const char* name) {
  const int ret = rknn_mem_sync(ctx, mem, RKNN_MEMORY_SYNC_FROM_DEVICE);
  if (ret != RKNN_SUCC) {
    std::fprintf(stderr, "[fail] rknn_mem_sync(FROM_DEVICE %s) failed: %d\n", name, ret);
    return false;
  }
  return true;
}

static bool sync_to_device(rknn_context ctx, rknn_tensor_mem* mem, const char* name) {
  const int ret = rknn_mem_sync(ctx, mem, RKNN_MEMORY_SYNC_TO_DEVICE);
  if (ret != RKNN_SUCC) {
    std::fprintf(stderr, "[fail] rknn_mem_sync(TO_DEVICE %s) failed: %d\n", name, ret);
    return false;
  }
  return true;
}

static bool verify_pattern(rknn_tensor_mem* mem,
                           uint32_t n,
                           uint8_t value,
                           const char* name) {
  const uint8_t* ptr = static_cast<const uint8_t*>(mem->virt_addr);
  if (ptr == nullptr) {
    std::fprintf(stderr, "[fail] %s has null virt_addr\n", name);
    return false;
  }
  const uint32_t check_n = std::min<uint32_t>(n, 4096);
  for (uint32_t i = 0; i < check_n; ++i) {
    const uint8_t expected = static_cast<uint8_t>(value + (i & 15));
    if (ptr[i] != expected) {
      std::fprintf(stderr,
                   "[fail] %s mismatch at %u: got=%u expected=%u\n",
                   name,
                   i,
                   static_cast<unsigned>(ptr[i]),
                   static_cast<unsigned>(expected));
      return false;
    }
  }
  std::printf("[ok] CPU verified %s first %u bytes pattern=0x%02x\n",
              name, check_n, static_cast<unsigned>(value));
  return true;
}

static bool run_with_mem(rknn_context ctx,
                         rknn_tensor_mem* mem,
                         rknn_tensor_attr* attr,
                         const char* name,
                         bool non_block,
                         double* elapsed_ms) {
  int ret = rknn_set_io_mem(ctx, mem, attr);
  if (ret != RKNN_SUCC) {
    std::fprintf(stderr, "[fail] rknn_set_io_mem(%s) failed: %d\n", name, ret);
    return false;
  }
  rknn_run_extend ext{};
  ext.non_block = non_block ? 1 : 0;
  ext.timeout_ms = 10000;
  const int64_t t0 = now_us();
  ret = rknn_run(ctx, &ext);
  const int64_t t1 = now_us();
  if (ret != RKNN_SUCC) {
    std::fprintf(stderr, "[fail] rknn_run(%s non_block=%d) failed: %d\n",
                 name, static_cast<int>(non_block), ret);
    return false;
  }
  *elapsed_ms = (t1 - t0) / 1000.0;
  std::printf("[ok] rknn_run(%s non_block=%d) returned in %.3f ms frame_id=%llu\n",
              name,
              static_cast<int>(non_block),
              *elapsed_ms,
              static_cast<unsigned long long>(ext.frame_id));
  return true;
}

static bool wait_outputs(rknn_context ctx, uint32_t n_output, const char* name, double* elapsed_ms) {
  if (n_output == 0 || n_output > 16) {
    std::fprintf(stderr, "[fail] invalid n_output=%u\n", n_output);
    return false;
  }
  std::vector<rknn_output> outs(n_output);
  for (uint32_t i = 0; i < n_output; ++i) {
    outs[i].index = i;
    outs[i].want_float = 0;
    outs[i].is_prealloc = 0;
  }
  rknn_output_extend output_ext{};
  const int64_t t0 = now_us();
  const int ret = rknn_outputs_get(ctx, n_output, outs.data(), &output_ext);
  const int64_t t1 = now_us();
  if (ret != RKNN_SUCC) {
    std::fprintf(stderr, "[fail] rknn_outputs_get(%s) failed: %d\n", name, ret);
    return false;
  }
  *elapsed_ms = (t1 - t0) / 1000.0;
  std::printf("[ok] rknn_outputs_get(%s) waited %.3f ms frame_id=%llu\n",
              name,
              *elapsed_ms,
              static_cast<unsigned long long>(output_ext.frame_id));
  rknn_outputs_release(ctx, n_output, outs.data());
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s /path/to/model.rknn [bytes_to_write]\n", argv[0]);
    return 2;
  }
  const char* model_path = argv[1];
  uint32_t bytes_to_write = 4096;
  if (argc >= 3) {
    bytes_to_write = static_cast<uint32_t>(std::max(1, std::atoi(argv[2])));
  }

  rknn_context ctx = 0;
  int ret = rknn_init(&ctx, const_cast<char*>(model_path), 0, 0, nullptr);
  if (ret != RKNN_SUCC) {
    std::fprintf(stderr, "[fail] rknn_init failed: %d model=%s\n", ret, model_path);
    return 1;
  }
  std::printf("[ok] rknn_init: %s\n", model_path);

  rknn_tensor_attr attr{};
  std::string error;
  if (!query_native_input_attr(ctx, &attr, &error)) {
    std::fprintf(stderr, "[fail] %s\n", error.c_str());
    rknn_destroy(ctx);
    return 1;
  }
  rknn_input_output_num io_num{};
  ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
  if (ret != RKNN_SUCC || io_num.n_output == 0) {
    std::fprintf(stderr, "[fail] RKNN_QUERY_IN_OUT_NUM failed: %d n_output=%u\n",
                 ret, io_num.n_output);
    rknn_destroy(ctx);
    return 1;
  }
  std::printf("[rknn] n_input=%u n_output=%u\n", io_num.n_input, io_num.n_output);
  attr.index = 0;
  attr.type = RKNN_TENSOR_UINT8;
  attr.fmt = RKNN_TENSOR_NHWC;
  attr.pass_through = 0;
  attr.h_stride = 0;
  const uint32_t mem_size = attr.size_with_stride > 0 ? attr.size_with_stride : attr.size;
  bytes_to_write = std::min<uint32_t>(bytes_to_write, mem_size);
  std::printf("[rknn] input mem size=%u test_write=%u\n", mem_size, bytes_to_write);

  rknn_tensor_mem* mem_a = create_input_mem(ctx, mem_size, "input_mem_a");
  rknn_tensor_mem* mem_b = create_input_mem(ctx, mem_size, "input_mem_b");
  if (mem_a == nullptr || mem_b == nullptr) {
    if (mem_a) rknn_destroy_mem(ctx, mem_a);
    if (mem_b) rknn_destroy_mem(ctx, mem_b);
    rknn_destroy(ctx);
    return 1;
  }
  if (mem_a->fd == mem_b->fd || mem_a->virt_addr == mem_b->virt_addr) {
    std::fprintf(stderr, "[fail] RKNN returned non-independent input memories\n");
    rknn_destroy_mem(ctx, mem_a);
    rknn_destroy_mem(ctx, mem_b);
    rknn_destroy(ctx);
    return 1;
  }
  std::printf("[ok] two independent RKNN input memories allocated\n");

  cl_platform_id platform = nullptr;
  cl_device_id device = nullptr;
  if (!pick_opencl_device(&platform, &device, &error)) {
    std::fprintf(stderr, "[fail] %s\n", error.c_str());
    rknn_destroy_mem(ctx, mem_a);
    rknn_destroy_mem(ctx, mem_b);
    rknn_destroy(ctx);
    return 1;
  }
  std::string platform_ext;
  std::string device_ext;
  get_info_string_platform(platform, CL_PLATFORM_EXTENSIONS, &platform_ext);
  get_info_string_device(device, CL_DEVICE_EXTENSIONS, &device_ext);
  const bool has_import_ext =
      platform_ext.find("cl_arm_import_memory") != std::string::npos ||
      device_ext.find("cl_arm_import_memory") != std::string::npos;
  if (!has_import_ext) {
    std::fprintf(stderr, "[fail] OpenCL extension cl_arm_import_memory not advertised\n");
    rknn_destroy_mem(ctx, mem_a);
    rknn_destroy_mem(ctx, mem_b);
    rknn_destroy(ctx);
    return 1;
  }
  auto import_fn = reinterpret_cast<clImportMemoryARMFn>(
      clGetExtensionFunctionAddressForPlatform(platform, "clImportMemoryARM"));
  if (import_fn == nullptr) {
    std::fprintf(stderr, "[fail] clImportMemoryARM symbol not found\n");
    rknn_destroy_mem(ctx, mem_a);
    rknn_destroy_mem(ctx, mem_b);
    rknn_destroy(ctx);
    return 1;
  }

  cl_int clret = CL_SUCCESS;
  cl_context clctx = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &clret);
  if (clret != CL_SUCCESS || clctx == nullptr) {
    std::fprintf(stderr, "[fail] %s\n", cl_err("clCreateContext", clret).c_str());
    rknn_destroy_mem(ctx, mem_a);
    rknn_destroy_mem(ctx, mem_b);
    rknn_destroy(ctx);
    return 1;
  }
  cl_command_queue queue = clCreateCommandQueue(clctx, device, 0, &clret);
  if (clret != CL_SUCCESS || queue == nullptr) {
    std::fprintf(stderr, "[fail] %s\n", cl_err("clCreateCommandQueue", clret).c_str());
    clReleaseContext(clctx);
    rknn_destroy_mem(ctx, mem_a);
    rknn_destroy_mem(ctx, mem_b);
    rknn_destroy(ctx);
    return 1;
  }

  cl_mem imported_a = nullptr;
  cl_mem imported_b = nullptr;
  if (!import_mem(clctx, import_fn, mem_a, &imported_a, "input_mem_a", &error) ||
      !import_mem(clctx, import_fn, mem_b, &imported_b, "input_mem_b", &error)) {
    std::fprintf(stderr, "[fail] %s\n", error.c_str());
    if (imported_a) clReleaseMemObject(imported_a);
    if (imported_b) clReleaseMemObject(imported_b);
    clReleaseCommandQueue(queue);
    clReleaseContext(clctx);
    rknn_destroy_mem(ctx, mem_a);
    rknn_destroy_mem(ctx, mem_b);
    rknn_destroy(ctx);
    return 1;
  }

  cl_program program = nullptr;
  cl_kernel kernel = nullptr;
  if (!build_fill_kernel(clctx, device, &program, &kernel, &error)) {
    std::fprintf(stderr, "[fail] %s\n", error.c_str());
    clReleaseMemObject(imported_a);
    clReleaseMemObject(imported_b);
    clReleaseCommandQueue(queue);
    clReleaseContext(clctx);
    rknn_destroy_mem(ctx, mem_a);
    rknn_destroy_mem(ctx, mem_b);
    rknn_destroy(ctx);
    return 1;
  }

  double ms = 0.0;
  if (!fill_imported(queue, kernel, imported_a, bytes_to_write, 0x11, "input_mem_a", &ms, &error) ||
      !fill_imported(queue, kernel, imported_b, bytes_to_write, 0x51, "input_mem_b", &ms, &error) ||
      !sync_from_device(ctx, mem_a, "input_mem_a") ||
      !sync_from_device(ctx, mem_b, "input_mem_b") ||
      !verify_pattern(mem_a, bytes_to_write, 0x11, "input_mem_a") ||
      !verify_pattern(mem_b, bytes_to_write, 0x51, "input_mem_b")) {
    std::fprintf(stderr, "[fail] initial OpenCL import/write/verify failed: %s\n", error.c_str());
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseMemObject(imported_a);
    clReleaseMemObject(imported_b);
    clReleaseCommandQueue(queue);
    clReleaseContext(clctx);
    rknn_destroy_mem(ctx, mem_a);
    rknn_destroy_mem(ctx, mem_b);
    rknn_destroy(ctx);
    return 1;
  }

  if (!sync_to_device(ctx, mem_a, "input_mem_a") ||
      !run_with_mem(ctx, mem_a, &attr, "input_mem_a", false, &ms) ||
      !sync_to_device(ctx, mem_b, "input_mem_b") ||
      !run_with_mem(ctx, mem_b, &attr, "input_mem_b", false, &ms) ||
      !sync_to_device(ctx, mem_a, "input_mem_a") ||
      !run_with_mem(ctx, mem_a, &attr, "input_mem_a again", false, &ms)) {
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseMemObject(imported_a);
    clReleaseMemObject(imported_b);
    clReleaseCommandQueue(queue);
    clReleaseContext(clctx);
    rknn_destroy_mem(ctx, mem_a);
    rknn_destroy_mem(ctx, mem_b);
    rknn_destroy(ctx);
    return 1;
  }
  std::printf("[ok] blocking rknn_set_io_mem can switch A -> B -> A\n");

  if (!fill_imported(queue, kernel, imported_a, bytes_to_write, 0x21, "input_mem_a", &ms, &error) ||
      !sync_from_device(ctx, mem_a, "input_mem_a") ||
      !verify_pattern(mem_a, bytes_to_write, 0x21, "input_mem_a")) {
    std::fprintf(stderr, "[fail] rewrite A before non-block test failed: %s\n", error.c_str());
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseMemObject(imported_a);
    clReleaseMemObject(imported_b);
    clReleaseCommandQueue(queue);
    clReleaseContext(clctx);
    rknn_destroy_mem(ctx, mem_a);
    rknn_destroy_mem(ctx, mem_b);
    rknn_destroy(ctx);
    return 1;
  }

  if (!sync_to_device(ctx, mem_a, "input_mem_a") ||
      !run_with_mem(ctx, mem_a, &attr, "input_mem_a", true, &ms)) {
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseMemObject(imported_a);
    clReleaseMemObject(imported_b);
    clReleaseCommandQueue(queue);
    clReleaseContext(clctx);
    rknn_destroy_mem(ctx, mem_a);
    rknn_destroy_mem(ctx, mem_b);
    rknn_destroy(ctx);
    return 1;
  }

  double write_b_ms = 0.0;
  const int64_t overlap_t0 = now_us();
  const bool wrote_b = fill_imported(
      queue, kernel, imported_b, bytes_to_write, 0x71, "input_mem_b while A is running", &write_b_ms, &error);
  const int64_t overlap_t1 = now_us();
  if (!wrote_b) {
    std::fprintf(stderr, "[fail] OpenCL write B while A run is pending failed: %s\n", error.c_str());
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseMemObject(imported_a);
    clReleaseMemObject(imported_b);
    clReleaseCommandQueue(queue);
    clReleaseContext(clctx);
    rknn_destroy_mem(ctx, mem_a);
    rknn_destroy_mem(ctx, mem_b);
    rknn_destroy(ctx);
    return 1;
  }
  std::printf("[ok] overlap probe: OpenCL write B completed in %.3f ms after non-block rknn_run(A)\n",
              (overlap_t1 - overlap_t0) / 1000.0);

  double wait_ms = 0.0;
  if (!wait_outputs(ctx, io_num.n_output, "input_mem_a", &wait_ms)) {
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseMemObject(imported_a);
    clReleaseMemObject(imported_b);
    clReleaseCommandQueue(queue);
    clReleaseContext(clctx);
    rknn_destroy_mem(ctx, mem_a);
    rknn_destroy_mem(ctx, mem_b);
    rknn_destroy(ctx);
    return 1;
  }

  if (!sync_from_device(ctx, mem_b, "input_mem_b") ||
      !verify_pattern(mem_b, bytes_to_write, 0x71, "input_mem_b") ||
      !sync_from_device(ctx, mem_a, "input_mem_a") ||
      !verify_pattern(mem_a, bytes_to_write, 0x21, "input_mem_a")) {
    std::fprintf(stderr, "[fail] buffer content changed unexpectedly during overlap probe\n");
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseMemObject(imported_a);
    clReleaseMemObject(imported_b);
    clReleaseCommandQueue(queue);
    clReleaseContext(clctx);
    rknn_destroy_mem(ctx, mem_a);
    rknn_destroy_mem(ctx, mem_b);
    rknn_destroy(ctx);
    return 1;
  }

  if (!sync_to_device(ctx, mem_b, "input_mem_b") ||
      !run_with_mem(ctx, mem_b, &attr, "input_mem_b after overlap write", false, &ms)) {
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseMemObject(imported_a);
    clReleaseMemObject(imported_b);
    clReleaseCommandQueue(queue);
    clReleaseContext(clctx);
    rknn_destroy_mem(ctx, mem_a);
    rknn_destroy_mem(ctx, mem_b);
    rknn_destroy(ctx);
    return 1;
  }

  std::printf("[result] double RKNN input buffers look feasible for OpenCL/RKNN pipelining.\n");
  std::printf("[result] This probe shows memory independence and API switching; main pipeline still needs per-context double-buffer state before using it.\n");

  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseMemObject(imported_a);
  clReleaseMemObject(imported_b);
  clReleaseCommandQueue(queue);
  clReleaseContext(clctx);
  rknn_destroy_mem(ctx, mem_a);
  rknn_destroy_mem(ctx, mem_b);
  rknn_destroy(ctx);
  return 0;
}
