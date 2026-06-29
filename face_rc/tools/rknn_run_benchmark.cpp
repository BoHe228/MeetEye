// Minimal RKNN C API benchmark.
//
// This intentionally follows Rockchip's model-zoo benchmark timing scope:
// inputs are set once, warmup is excluded, and only rknn_run() is timed.

#include "rknn_api.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static int64_t now_us()
{
  using clock = std::chrono::steady_clock;
  return std::chrono::duration_cast<std::chrono::microseconds>(clock::now().time_since_epoch()).count();
}

static const char* tensor_type_name(rknn_tensor_type type)
{
  switch (type) {
  case RKNN_TENSOR_FLOAT32:
    return "float32";
  case RKNN_TENSOR_FLOAT16:
    return "float16";
  case RKNN_TENSOR_INT8:
    return "int8";
  case RKNN_TENSOR_UINT8:
    return "uint8";
  case RKNN_TENSOR_INT16:
    return "int16";
  case RKNN_TENSOR_UINT16:
    return "uint16";
  case RKNN_TENSOR_INT32:
    return "int32";
  case RKNN_TENSOR_UINT32:
    return "uint32";
  case RKNN_TENSOR_INT64:
    return "int64";
  case RKNN_TENSOR_BOOL:
    return "bool";
  default:
    return "unknown";
  }
}

static const char* tensor_fmt_name(rknn_tensor_format fmt)
{
  switch (fmt) {
  case RKNN_TENSOR_NCHW:
    return "NCHW";
  case RKNN_TENSOR_NHWC:
    return "NHWC";
  case RKNN_TENSOR_NC1HWC2:
    return "NC1HWC2";
  case RKNN_TENSOR_UNDEFINED:
    return "UNDEFINED";
  default:
    return "unknown";
  }
}

static void dump_attr(const char* prefix, const rknn_tensor_attr& attr)
{
  std::printf("%s index=%u name=%s dims=[", prefix, attr.index, attr.name);
  for (uint32_t i = 0; i < attr.n_dims; ++i) {
    std::printf("%s%d", i == 0 ? "" : ",", attr.dims[i]);
  }
  std::printf("] elems=%u size=%u fmt=%s type=%s qnt=%d zp=%d scale=%f\n",
              attr.n_elems,
              attr.size,
              tensor_fmt_name(attr.fmt),
              tensor_type_name(attr.type),
              attr.qnt_type,
              attr.zp,
              attr.scale);
}

static size_t type_size(rknn_tensor_type type)
{
  switch (type) {
  case RKNN_TENSOR_FLOAT32:
  case RKNN_TENSOR_INT32:
  case RKNN_TENSOR_UINT32:
    return 4;
  case RKNN_TENSOR_FLOAT16:
  case RKNN_TENSOR_INT16:
  case RKNN_TENSOR_UINT16:
    return 2;
  case RKNN_TENSOR_INT64:
    return 8;
  case RKNN_TENSOR_BOOL:
  case RKNN_TENSOR_INT8:
  case RKNN_TENSOR_UINT8:
  default:
    return 1;
  }
}

static bool load_file(const char* path, std::vector<uint8_t>& data)
{
  std::FILE* fp = std::fopen(path, "rb");
  if (!fp) {
    std::fprintf(stderr, "open model failed: %s\n", path);
    return false;
  }
  if (std::fseek(fp, 0, SEEK_END) != 0) {
    std::fclose(fp);
    std::fprintf(stderr, "seek model failed: %s\n", path);
    return false;
  }
  long size = std::ftell(fp);
  if (size <= 0) {
    std::fclose(fp);
    std::fprintf(stderr, "invalid model size: %ld\n", size);
    return false;
  }
  std::rewind(fp);
  data.resize(static_cast<size_t>(size));
  size_t read_size = std::fread(data.data(), 1, data.size(), fp);
  std::fclose(fp);
  if (read_size != data.size()) {
    std::fprintf(stderr, "read model failed: %zu/%zu bytes\n", read_size, data.size());
    return false;
  }
  return true;
}

int main(int argc, char** argv)
{
  if (argc < 2) {
    std::printf("Usage: %s model.rknn [loop_count=300] [warmup=30] [core_mask=7]\n", argv[0]);
    std::printf("core_mask on RK3588: 1=core0, 2=core1, 4=core2, 7=core0+1+2, 65535=all\n");
    return 2;
  }

  const char* model_path = argv[1];
  int loop_count = argc > 2 ? std::atoi(argv[2]) : 300;
  int warmup = argc > 3 ? std::atoi(argv[3]) : 30;
  uint32_t core_mask = argc > 4 ? static_cast<uint32_t>(std::strtoul(argv[4], nullptr, 0)) : 7;
  if (loop_count <= 0) {
    std::fprintf(stderr, "loop_count must be > 0\n");
    return 2;
  }
  if (warmup < 0) {
    std::fprintf(stderr, "warmup must be >= 0\n");
    return 2;
  }

  std::printf("loading model: %s\n", model_path);
  std::fflush(stdout);
  std::vector<uint8_t> model_data;
  if (!load_file(model_path, model_data)) {
    return 1;
  }
  std::printf("model bytes: %zu\n", model_data.size());
  std::fflush(stdout);

  rknn_context ctx = 0;
  int ret = rknn_init(&ctx, model_data.data(), static_cast<uint32_t>(model_data.size()), 0, nullptr);
  if (ret < 0) {
    std::fprintf(stderr, "rknn_init failed: %d\n", ret);
    return 1;
  }

  rknn_sdk_version sdk_ver;
  ret = rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &sdk_ver, sizeof(sdk_ver));
  if (ret == RKNN_SUCC) {
    std::printf("rknn_api/rknnrt version: %s, driver version: %s\n", sdk_ver.api_version, sdk_ver.drv_version);
  }

  rknn_input_output_num io_num;
  ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
  if (ret != RKNN_SUCC) {
    std::fprintf(stderr, "RKNN_QUERY_IN_OUT_NUM failed: %d\n", ret);
    rknn_destroy(ctx);
    return 1;
  }
  std::printf("model input num: %u, output num: %u\n", io_num.n_input, io_num.n_output);

  std::vector<rknn_tensor_attr> input_attrs(io_num.n_input);
  for (uint32_t i = 0; i < io_num.n_input; ++i) {
    std::memset(&input_attrs[i], 0, sizeof(rknn_tensor_attr));
    input_attrs[i].index = i;
    ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &input_attrs[i], sizeof(rknn_tensor_attr));
    if (ret != RKNN_SUCC) {
      std::fprintf(stderr, "RKNN_QUERY_INPUT_ATTR[%u] failed: %d\n", i, ret);
      rknn_destroy(ctx);
      return 1;
    }
    dump_attr("input", input_attrs[i]);
  }

  std::vector<rknn_tensor_attr> output_attrs(io_num.n_output);
  for (uint32_t i = 0; i < io_num.n_output; ++i) {
    std::memset(&output_attrs[i], 0, sizeof(rknn_tensor_attr));
    output_attrs[i].index = i;
    ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &output_attrs[i], sizeof(rknn_tensor_attr));
    if (ret == RKNN_SUCC) {
      dump_attr("output", output_attrs[i]);
    }
  }

  std::vector<std::vector<uint8_t>> input_buffers(io_num.n_input);
  std::vector<rknn_input> inputs(io_num.n_input);
  for (uint32_t i = 0; i < io_num.n_input; ++i) {
    const size_t bytes = static_cast<size_t>(input_attrs[i].n_elems) * type_size(input_attrs[i].type);
    input_buffers[i].assign(bytes, 0);
    std::memset(&inputs[i], 0, sizeof(rknn_input));
    inputs[i].index = i;
    inputs[i].pass_through = 0;
    inputs[i].type = input_attrs[i].type;
    inputs[i].fmt = input_attrs[i].fmt;
    inputs[i].size = static_cast<uint32_t>(input_buffers[i].size());
    inputs[i].buf = input_buffers[i].data();
  }

  ret = rknn_inputs_set(ctx, io_num.n_input, inputs.data());
  if (ret < 0) {
    std::fprintf(stderr, "rknn_inputs_set failed: %d\n", ret);
    rknn_destroy(ctx);
    return 1;
  }

  ret = rknn_set_core_mask(ctx, static_cast<rknn_core_mask>(core_mask));
  if (ret < 0) {
    std::fprintf(stderr, "rknn_set_core_mask(%u) failed: %d\n", core_mask, ret);
    rknn_destroy(ctx);
    return 1;
  }

  std::printf("warmup=%d loops=%d core_mask=%u\n", warmup, loop_count, core_mask);
  for (int i = 0; i < warmup; ++i) {
    ret = rknn_run(ctx, nullptr);
    if (ret < 0) {
      std::fprintf(stderr, "warmup rknn_run failed at %d: %d\n", i, ret);
      rknn_destroy(ctx);
      return 1;
    }
  }

  std::vector<double> times_ms;
  times_ms.reserve(loop_count);
  int64_t wall_start = now_us();
  for (int i = 0; i < loop_count; ++i) {
    int64_t t0 = now_us();
    ret = rknn_run(ctx, nullptr);
    int64_t t1 = now_us();
    if (ret < 0) {
      std::fprintf(stderr, "rknn_run failed at %d: %d\n", i, ret);
      rknn_destroy(ctx);
      return 1;
    }
    times_ms.push_back((t1 - t0) / 1000.0);
  }
  int64_t wall_end = now_us();

  std::vector<double> sorted = times_ms;
  std::sort(sorted.begin(), sorted.end());
  double sum = 0.0;
  for (double v : times_ms) {
    sum += v;
  }
  auto pct = [&](double q) {
    if (sorted.empty()) {
      return 0.0;
    }
    double pos = (sorted.size() - 1) * q;
    size_t lo = static_cast<size_t>(pos);
    size_t hi = std::min(lo + 1, sorted.size() - 1);
    double frac = pos - lo;
    return sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
  };

  double mean = sum / times_ms.size();
  double wall_s = (wall_end - wall_start) / 1000000.0;
  std::printf("\n===== C API rknn_run Benchmark =====\n");
  std::printf("model:     %s\n", model_path);
  std::printf("mean:      %.3f ms  (%.2f FPS by mean rknn_run)\n", mean, mean > 0 ? 1000.0 / mean : 0.0);
  std::printf("median:    %.3f ms\n", pct(0.50));
  std::printf("p90 / p95: %.3f / %.3f ms\n", pct(0.90), pct(0.95));
  std::printf("min / max: %.3f / %.3f ms\n", sorted.front(), sorted.back());
  std::printf("wall FPS:  %.2f\n", wall_s > 0 ? loop_count / wall_s : 0.0);

  rknn_destroy(ctx);
  return 0;
}
