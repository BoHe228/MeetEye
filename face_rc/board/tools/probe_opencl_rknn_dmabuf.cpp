// Probe whether RKNN input dma-buf can be imported by Mali OpenCL and written
// directly by an OpenCL kernel. This is a standalone diagnostic, not a runtime
// dependency of the board pipeline.

#include "rknn_api.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// Minimal OpenCL declarations. The board image may not ship OpenCL headers.
using cl_bool = uint32_t;
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
constexpr cl_mem_flags CL_MEM_WRITE_ONLY = 1ULL << 1;
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

static std::string cl_err(const char* what, cl_int code) {
  char buf[128];
  const char* name = "UNKNOWN";
  switch (code) {
    case 0: name = "CL_SUCCESS"; break;
    case -1: name = "CL_DEVICE_NOT_FOUND"; break;
    case -2: name = "CL_DEVICE_NOT_AVAILABLE"; break;
    case -3: name = "CL_COMPILER_NOT_AVAILABLE"; break;
    case -4: name = "CL_MEM_OBJECT_ALLOCATION_FAILURE"; break;
    case -5: name = "CL_OUT_OF_RESOURCES"; break;
    case -6: name = "CL_OUT_OF_HOST_MEMORY"; break;
    case -30: name = "CL_INVALID_VALUE"; break;
    case -34: name = "CL_INVALID_CONTEXT"; break;
    case -37: name = "CL_INVALID_HOST_PTR"; break;
    case -61: name = "CL_INVALID_BUFFER_SIZE"; break;
    default: break;
  }
  std::snprintf(buf, sizeof(buf), "%s failed: %d (%s)", what, code, name);
  return std::string(buf);
}

static bool get_info_string_platform(cl_platform_id platform, cl_uint name, std::string* out) {
  size_t size = 0;
  cl_int ret = clGetPlatformInfo(platform, name, 0, nullptr, &size);
  if (ret != CL_SUCCESS || size == 0) {
    return false;
  }
  std::vector<char> buf(size + 1, '\0');
  ret = clGetPlatformInfo(platform, name, size, buf.data(), nullptr);
  if (ret != CL_SUCCESS) {
    return false;
  }
  *out = buf.data();
  return true;
}

static bool get_info_string_device(cl_device_id device, cl_uint name, std::string* out) {
  size_t size = 0;
  cl_int ret = clGetDeviceInfo(device, name, 0, nullptr, &size);
  if (ret != CL_SUCCESS || size == 0) {
    return false;
  }
  std::vector<char> buf(size + 1, '\0');
  ret = clGetDeviceInfo(device, name, size, buf.data(), nullptr);
  if (ret != CL_SUCCESS) {
    return false;
  }
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

static void print_attr(const rknn_tensor_attr& attr) {
  std::printf("[rknn] native input: n_dims=%u dims=", attr.n_dims);
  for (uint32_t i = 0; i < attr.n_dims; ++i) {
    std::printf("%s%u", i ? "x" : "", attr.dims[i]);
  }
  std::printf(" fmt=%s type=%d size=%u size_with_stride=%u w_stride=%u\n",
              get_format_string(attr.fmt),
              static_cast<int>(attr.type),
              attr.size,
              attr.size_with_stride,
              attr.w_stride);
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
  print_attr(attr);

  const uint32_t mem_size = attr.size_with_stride > 0 ? attr.size_with_stride : attr.size;
  rknn_tensor_mem* mem = rknn_create_mem(ctx, mem_size);
  if (mem == nullptr) {
    std::fprintf(stderr, "[fail] rknn_create_mem failed size=%u\n", mem_size);
    rknn_destroy(ctx);
    return 1;
  }
  std::printf("[ok] rknn_create_mem: size=%u virt=%p fd=%d phys=0x%llx\n",
              mem->size,
              mem->virt_addr,
              mem->fd,
              static_cast<unsigned long long>(mem->phys_addr));
  if (mem->fd < 0) {
    std::fprintf(stderr, "[fail] RKNN mem has no fd; cannot test OpenCL dma-buf import\n");
    rknn_destroy_mem(ctx, mem);
    rknn_destroy(ctx);
    return 1;
  }

  ret = rknn_set_io_mem(ctx, mem, &attr);
  if (ret != RKNN_SUCC) {
    std::fprintf(stderr, "[fail] rknn_set_io_mem failed: %d\n", ret);
    rknn_destroy_mem(ctx, mem);
    rknn_destroy(ctx);
    return 1;
  }
  std::printf("[ok] rknn_set_io_mem\n");

  cl_platform_id platform = nullptr;
  cl_device_id device = nullptr;
  if (!pick_opencl_device(&platform, &device, &error)) {
    std::fprintf(stderr, "[fail] %s\n", error.c_str());
    rknn_destroy_mem(ctx, mem);
    rknn_destroy(ctx);
    return 1;
  }
  std::string platform_ext;
  std::string device_ext;
  get_info_string_platform(platform, CL_PLATFORM_EXTENSIONS, &platform_ext);
  get_info_string_device(device, CL_DEVICE_EXTENSIONS, &device_ext);
  std::printf("[opencl] platform extensions: %s\n", platform_ext.c_str());
  std::printf("[opencl] device extensions: %s\n", device_ext.c_str());
  const bool has_import_ext =
      platform_ext.find("cl_arm_import_memory") != std::string::npos ||
      device_ext.find("cl_arm_import_memory") != std::string::npos;
  if (!has_import_ext) {
    std::fprintf(stderr, "[fail] OpenCL extension cl_arm_import_memory not advertised\n");
    rknn_destroy_mem(ctx, mem);
    rknn_destroy(ctx);
    return 1;
  }

  auto import_fn = reinterpret_cast<clImportMemoryARMFn>(
      clGetExtensionFunctionAddressForPlatform(platform, "clImportMemoryARM"));
  if (import_fn == nullptr) {
    std::fprintf(stderr, "[fail] clImportMemoryARM symbol not found\n");
    rknn_destroy_mem(ctx, mem);
    rknn_destroy(ctx);
    return 1;
  }
  std::printf("[ok] clImportMemoryARM resolved\n");

  cl_int clret = CL_SUCCESS;
  cl_context clctx = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &clret);
  if (clret != CL_SUCCESS || clctx == nullptr) {
    std::fprintf(stderr, "[fail] %s\n", cl_err("clCreateContext", clret).c_str());
    rknn_destroy_mem(ctx, mem);
    rknn_destroy(ctx);
    return 1;
  }
  cl_command_queue queue = clCreateCommandQueue(clctx, device, 0, &clret);
  if (clret != CL_SUCCESS || queue == nullptr) {
    std::fprintf(stderr, "[fail] %s\n", cl_err("clCreateCommandQueue", clret).c_str());
    clReleaseContext(clctx);
    rknn_destroy_mem(ctx, mem);
    rknn_destroy(ctx);
    return 1;
  }

  const cl_import_properties_arm props[] = {
      CL_IMPORT_TYPE_ARM,
      CL_IMPORT_TYPE_DMA_BUF_ARM,
      0,
  };
  const size_t import_size = mem->size;
  int fd_value = mem->fd;
  cl_mem imported = nullptr;
  clret = CL_SUCCESS;

  // For CL_IMPORT_TYPE_DMA_BUF_ARM, the memory argument is a pointer to the
  // dma-buf file descriptor, not the fd value cast to a pointer.
  const cl_mem_flags try_flags[] = {CL_MEM_READ_WRITE, CL_MEM_WRITE_ONLY};
  const char* try_names[] = {"CL_MEM_READ_WRITE", "CL_MEM_WRITE_ONLY"};
  for (int attempt = 0; attempt < 2 && imported == nullptr; ++attempt) {
    clret = CL_SUCCESS;
    imported = import_fn(
        clctx,
        try_flags[attempt],
        props,
        &fd_value,
        import_size,
        &clret);
    if (imported != nullptr && clret == CL_SUCCESS) {
      std::printf("[ok] clImportMemoryARM imported RKNN fd as cl_mem flags=%s\n", try_names[attempt]);
      break;
    }
    std::printf("[probe] clImportMemoryARM flags=%s -> %s\n",
                try_names[attempt], cl_err("clImportMemoryARM", clret).c_str());
    imported = nullptr;
  }
  if (clret != CL_SUCCESS || imported == nullptr) {
    std::fprintf(stderr, "[fail] %s fd=%d size=%zu\n",
                 cl_err("clImportMemoryARM", clret).c_str(), fd_value, import_size);
    clReleaseCommandQueue(queue);
    clReleaseContext(clctx);
    rknn_destroy_mem(ctx, mem);
    rknn_destroy(ctx);
    return 1;
  }

  cl_program program = nullptr;
  cl_kernel kernel = nullptr;
  if (!build_fill_kernel(clctx, device, &program, &kernel, &error)) {
    std::fprintf(stderr, "[fail] %s\n", error.c_str());
    clReleaseMemObject(imported);
    clReleaseCommandQueue(queue);
    clReleaseContext(clctx);
    rknn_destroy_mem(ctx, mem);
    rknn_destroy(ctx);
    return 1;
  }

  bytes_to_write = std::min<uint32_t>(bytes_to_write, mem->size);
  const uint8_t value = 0x31;
  clret = clSetKernelArg(kernel, 0, sizeof(cl_mem), &imported);
  clret |= clSetKernelArg(kernel, 1, sizeof(cl_uint), &bytes_to_write);
  clret |= clSetKernelArg(kernel, 2, sizeof(uint8_t), &value);
  if (clret != CL_SUCCESS) {
    std::fprintf(stderr, "[fail] %s\n", cl_err("clSetKernelArg", clret).c_str());
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseMemObject(imported);
    clReleaseCommandQueue(queue);
    clReleaseContext(clctx);
    rknn_destroy_mem(ctx, mem);
    rknn_destroy(ctx);
    return 1;
  }
  const size_t global = ((static_cast<size_t>(bytes_to_write) + 255) / 256) * 256;
  clret = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global, nullptr, 0, nullptr, nullptr);
  if (clret != CL_SUCCESS) {
    std::fprintf(stderr, "[fail] %s\n", cl_err("clEnqueueNDRangeKernel", clret).c_str());
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseMemObject(imported);
    clReleaseCommandQueue(queue);
    clReleaseContext(clctx);
    rknn_destroy_mem(ctx, mem);
    rknn_destroy(ctx);
    return 1;
  }
  clret = clFinish(queue);
  if (clret != CL_SUCCESS) {
    std::fprintf(stderr, "[fail] %s\n", cl_err("clFinish", clret).c_str());
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseMemObject(imported);
    clReleaseCommandQueue(queue);
    clReleaseContext(clctx);
    rknn_destroy_mem(ctx, mem);
    rknn_destroy(ctx);
    return 1;
  }
  std::printf("[ok] OpenCL kernel wrote %u bytes into imported RKNN input buffer\n", bytes_to_write);

  ret = rknn_mem_sync(ctx, mem, RKNN_MEMORY_SYNC_FROM_DEVICE);
  if (ret != RKNN_SUCC) {
    std::fprintf(stderr, "[fail] rknn_mem_sync(FROM_DEVICE) failed: %d\n", ret);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseMemObject(imported);
    clReleaseCommandQueue(queue);
    clReleaseContext(clctx);
    rknn_destroy_mem(ctx, mem);
    rknn_destroy(ctx);
    return 1;
  }
  std::printf("[ok] rknn_mem_sync(FROM_DEVICE)\n");

  const uint8_t* ptr = static_cast<const uint8_t*>(mem->virt_addr);
  bool verified = ptr != nullptr;
  for (uint32_t i = 0; verified && i < std::min<uint32_t>(bytes_to_write, 64); ++i) {
    const uint8_t expected = static_cast<uint8_t>(value + (i & 15));
    if (ptr[i] != expected) {
      std::fprintf(stderr, "[fail] CPU verify mismatch at %u: got=%u expected=%u\n",
                   i, static_cast<unsigned>(ptr[i]), static_cast<unsigned>(expected));
      verified = false;
    }
  }
  if (!verified) {
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseMemObject(imported);
    clReleaseCommandQueue(queue);
    clReleaseContext(clctx);
    rknn_destroy_mem(ctx, mem);
    rknn_destroy(ctx);
    return 1;
  }
  std::printf("[ok] CPU verified first bytes after OpenCL write + RKNN sync\n");
  std::printf("[result] OpenCL/RKNN dma-buf sharing appears feasible on this runtime.\n");

  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseMemObject(imported);
  clReleaseCommandQueue(queue);
  clReleaseContext(clctx);
  rknn_destroy_mem(ctx, mem);
  rknn_destroy(ctx);
  return 0;
}
