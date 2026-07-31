// Minimal AdaFace RKNN embedding runtime.
//
// The public C API accepts one normalized NCHW float32 face tensor
// (3x112x112, values in [-1, 1]) and returns one L2-normalized embedding.

#include "rknn_api.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr int kExpectedChannels = 3;

struct AdaFaceRknn {
  rknn_context ctx = 0;
  rknn_tensor_attr input_attr{};
  std::vector<rknn_tensor_attr> output_attrs;
  int input_c = 0;
  int input_h = 0;
  int input_w = 0;
  int feature_dim = 0;
  int feature_output_index = 0;
  std::vector<float> input_scratch;
};

static void set_error(char* err, int err_len, const std::string& msg) {
  if (err == nullptr || err_len <= 0) {
    return;
  }
  std::snprintf(err, static_cast<size_t>(err_len), "%s", msg.c_str());
}

static int get_dim(const rknn_tensor_attr& attr, uint32_t idx) {
  return idx < attr.n_dims ? static_cast<int>(attr.dims[idx]) : 0;
}

static int tensor_elements(const rknn_tensor_attr& attr) {
  int n = 1;
  for (uint32_t i = 0; i < attr.n_dims; ++i) {
    const int dim = static_cast<int>(attr.dims[i]);
    if (dim <= 0) {
      return 0;
    }
    n *= dim;
  }
  return n;
}

static int infer_input_shape(AdaFaceRknn* runner, char* err, int err_len) {
  const rknn_tensor_attr& in = runner->input_attr;
  if (in.n_dims != 4) {
    set_error(err, err_len, "AdaFace input must be 4D");
    return -1;
  }
  if (in.fmt == RKNN_TENSOR_NCHW) {
    runner->input_c = get_dim(in, 1);
    runner->input_h = get_dim(in, 2);
    runner->input_w = get_dim(in, 3);
  } else if (in.fmt == RKNN_TENSOR_NHWC) {
    runner->input_h = get_dim(in, 1);
    runner->input_w = get_dim(in, 2);
    runner->input_c = get_dim(in, 3);
  } else {
    set_error(err, err_len, "AdaFace input format must be NCHW or NHWC");
    return -1;
  }
  if (runner->input_c != kExpectedChannels || runner->input_h <= 0 || runner->input_w <= 0) {
    set_error(err, err_len, "unexpected AdaFace input shape");
    return -1;
  }
  return 0;
}

static int pick_feature_output(AdaFaceRknn* runner, char* err, int err_len) {
  int best_idx = -1;
  int best_dim = 0;
  for (size_t i = 0; i < runner->output_attrs.size(); ++i) {
    const int n = tensor_elements(runner->output_attrs[i]);
    if (n == 512) {
      runner->feature_output_index = static_cast<int>(i);
      runner->feature_dim = n;
      return 0;
    }
    if (n > best_dim) {
      best_idx = static_cast<int>(i);
      best_dim = n;
    }
  }
  if (best_idx < 0 || best_dim < 128) {
    set_error(err, err_len, "cannot find AdaFace feature output");
    return -1;
  }
  runner->feature_output_index = best_idx;
  runner->feature_dim = best_dim;
  return 0;
}

static void normalize_feature(float* feature, int n) {
  double sum = 0.0;
  for (int i = 0; i < n; ++i) {
    sum += static_cast<double>(feature[i]) * static_cast<double>(feature[i]);
  }
  const double norm = std::sqrt(sum);
  if (norm <= 1.0e-12) {
    return;
  }
  const float inv = static_cast<float>(1.0 / norm);
  for (int i = 0; i < n; ++i) {
    feature[i] *= inv;
  }
}

static const float* prepare_input(AdaFaceRknn* runner, const float* nchw) {
  if (runner->input_attr.fmt == RKNN_TENSOR_NCHW) {
    return nchw;
  }
  const int c = runner->input_c;
  const int h = runner->input_h;
  const int w = runner->input_w;
  runner->input_scratch.assign(static_cast<size_t>(h) * w * c, 0.0f);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      for (int ch = 0; ch < c; ++ch) {
        runner->input_scratch[(static_cast<size_t>(y) * w + x) * c + ch] =
            nchw[(static_cast<size_t>(ch) * h + y) * w + x];
      }
    }
  }
  return runner->input_scratch.data();
}

static rknn_core_mask normalize_core_mask(int core_mask) {
  switch (core_mask) {
    case 1:
      return RKNN_NPU_CORE_0;
    case 2:
      return RKNN_NPU_CORE_1;
    case 3:
      return RKNN_NPU_CORE_0_1;
    case 4:
      return RKNN_NPU_CORE_2;
    case 7:
      return RKNN_NPU_CORE_0_1_2;
    case 0xffff:
      return RKNN_NPU_CORE_ALL;
    case 0:
    default:
      return RKNN_NPU_CORE_AUTO;
  }
}

}  // namespace

extern "C" {

int adaface_rknn_create_with_core_mask(const char* model_path,
                                       int core_mask,
                                       void** handle,
                                       char* err,
                                       int err_len) {
  if (handle == nullptr || model_path == nullptr || model_path[0] == '\0') {
    set_error(err, err_len, "invalid AdaFace model path");
    return -1;
  }
  *handle = nullptr;

  AdaFaceRknn* runner = new AdaFaceRknn();
  int ret = rknn_init(&runner->ctx, const_cast<char*>(model_path), 0, 0, nullptr);
  if (ret != RKNN_SUCC) {
    set_error(err, err_len, "rknn_init(AdaFace) failed: " + std::to_string(ret));
    delete runner;
    return ret;
  }

  ret = rknn_set_core_mask(runner->ctx, normalize_core_mask(core_mask));
  if (ret != RKNN_SUCC) {
    set_error(err, err_len, "rknn_set_core_mask(AdaFace) failed: " + std::to_string(ret));
    rknn_destroy(runner->ctx);
    delete runner;
    return ret;
  }

  rknn_input_output_num io_num{};
  ret = rknn_query(runner->ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
  if (ret != RKNN_SUCC || io_num.n_input != 1 || io_num.n_output < 1) {
    set_error(err, err_len, "invalid AdaFace RKNN input/output count");
    rknn_destroy(runner->ctx);
    delete runner;
    return ret == RKNN_SUCC ? -1 : ret;
  }

  std::memset(&runner->input_attr, 0, sizeof(runner->input_attr));
  runner->input_attr.index = 0;
  ret = rknn_query(runner->ctx, RKNN_QUERY_INPUT_ATTR, &runner->input_attr, sizeof(runner->input_attr));
  if (ret != RKNN_SUCC) {
    set_error(err, err_len, "RKNN_QUERY_INPUT_ATTR(AdaFace) failed: " + std::to_string(ret));
    rknn_destroy(runner->ctx);
    delete runner;
    return ret;
  }
  if (infer_input_shape(runner, err, err_len) != 0) {
    rknn_destroy(runner->ctx);
    delete runner;
    return -1;
  }

  runner->output_attrs.resize(io_num.n_output);
  for (uint32_t i = 0; i < io_num.n_output; ++i) {
    std::memset(&runner->output_attrs[i], 0, sizeof(rknn_tensor_attr));
    runner->output_attrs[i].index = i;
    ret = rknn_query(runner->ctx, RKNN_QUERY_OUTPUT_ATTR, &runner->output_attrs[i], sizeof(rknn_tensor_attr));
    if (ret != RKNN_SUCC) {
      set_error(err, err_len, "RKNN_QUERY_OUTPUT_ATTR(AdaFace) failed: " + std::to_string(ret));
      rknn_destroy(runner->ctx);
      delete runner;
      return ret;
    }
  }
  if (pick_feature_output(runner, err, err_len) != 0) {
    rknn_destroy(runner->ctx);
    delete runner;
    return -1;
  }

  *handle = runner;
  return 0;
}

int adaface_rknn_create(const char* model_path, void** handle, char* err, int err_len) {
  return adaface_rknn_create_with_core_mask(model_path, 0, handle, err, err_len);
}

void adaface_rknn_destroy(void* handle) {
  AdaFaceRknn* runner = static_cast<AdaFaceRknn*>(handle);
  if (runner == nullptr) {
    return;
  }
  if (runner->ctx != 0) {
    rknn_destroy(runner->ctx);
  }
  delete runner;
}

int adaface_rknn_get_shape(void* handle,
                           int* input_h,
                           int* input_w,
                           int* input_c,
                           int* feature_dim,
                           char* err,
                           int err_len) {
  AdaFaceRknn* runner = static_cast<AdaFaceRknn*>(handle);
  if (runner == nullptr) {
    set_error(err, err_len, "invalid AdaFace handle");
    return -1;
  }
  if (input_h != nullptr) {
    *input_h = runner->input_h;
  }
  if (input_w != nullptr) {
    *input_w = runner->input_w;
  }
  if (input_c != nullptr) {
    *input_c = runner->input_c;
  }
  if (feature_dim != nullptr) {
    *feature_dim = runner->feature_dim;
  }
  return 0;
}

int adaface_rknn_infer(void* handle,
                       const float* input_nchw,
                       int input_count,
                       float* feature,
                       int feature_cap,
                       char* err,
                       int err_len) {
  AdaFaceRknn* runner = static_cast<AdaFaceRknn*>(handle);
  if (runner == nullptr || input_nchw == nullptr || feature == nullptr) {
    set_error(err, err_len, "invalid AdaFace inference argument");
    return -1;
  }
  const int expected = runner->input_c * runner->input_h * runner->input_w;
  if (input_count != expected) {
    set_error(err, err_len, "AdaFace input_count mismatch");
    return -1;
  }
  if (feature_cap < runner->feature_dim) {
    set_error(err, err_len, "AdaFace feature buffer too small");
    return -1;
  }

  const float* input_ptr = prepare_input(runner, input_nchw);
  rknn_input in{};
  in.index = 0;
  in.pass_through = 0;
  in.type = RKNN_TENSOR_FLOAT32;
  in.fmt = runner->input_attr.fmt;
  in.size = static_cast<uint32_t>(expected * sizeof(float));
  in.buf = const_cast<float*>(input_ptr);

  int ret = rknn_inputs_set(runner->ctx, 1, &in);
  if (ret != RKNN_SUCC) {
    set_error(err, err_len, "rknn_inputs_set(AdaFace) failed: " + std::to_string(ret));
    return ret;
  }
  ret = rknn_run(runner->ctx, nullptr);
  if (ret != RKNN_SUCC) {
    set_error(err, err_len, "rknn_run(AdaFace) failed: " + std::to_string(ret));
    return ret;
  }

  std::vector<rknn_output> outs(runner->output_attrs.size());
  for (size_t i = 0; i < outs.size(); ++i) {
    outs[i].index = static_cast<uint32_t>(i);
    outs[i].want_float = 1;
    outs[i].is_prealloc = 0;
  }
  ret = rknn_outputs_get(runner->ctx, static_cast<uint32_t>(outs.size()), outs.data(), nullptr);
  if (ret != RKNN_SUCC) {
    set_error(err, err_len, "rknn_outputs_get(AdaFace) failed: " + std::to_string(ret));
    return ret;
  }

  const int out_idx = runner->feature_output_index;
  const float* src = static_cast<const float*>(outs[static_cast<size_t>(out_idx)].buf);
  if (src == nullptr) {
    rknn_outputs_release(runner->ctx, static_cast<uint32_t>(outs.size()), outs.data());
    set_error(err, err_len, "empty AdaFace feature output");
    return -1;
  }
  std::copy(src, src + runner->feature_dim, feature);
  rknn_outputs_release(runner->ctx, static_cast<uint32_t>(outs.size()), outs.data());
  normalize_feature(feature, runner->feature_dim);
  return runner->feature_dim;
}

}  // extern "C"
