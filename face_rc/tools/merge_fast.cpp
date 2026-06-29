// Fast spatial merge/dedup/NMS path for face_rc direct-slice detections.
//
// This mirrors the Python PanoramaSlicer path used when ReID is disabled:
// slice-local detections -> panorama coordinates -> spatial dedup ->
// area-weighted NMS -> final spatial dedup.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <set>
#include <string>
#include <vector>

namespace {

struct Box {
  float x1 = 0.0f;
  float y1 = 0.0f;
  float x2 = 0.0f;
  float y2 = 0.0f;
};

struct Detection {
  Box box;
  float score = 0.0f;
  int label = 0;
  int slice_idx = 0;
  float keypoints[15]{};
  float area = 0.0f;
};

static int64_t now_us()
{
  using clock = std::chrono::steady_clock;
  return std::chrono::duration_cast<std::chrono::microseconds>(clock::now().time_since_epoch()).count();
}

static void set_error(char* err, int err_len, const std::string& msg)
{
  if (err == nullptr || err_len <= 0) {
    return;
  }
  std::snprintf(err, static_cast<size_t>(err_len), "%s", msg.c_str());
}

static float clip_f(float value, float low, float high)
{
  return std::max(low, std::min(value, high));
}

static float box_iou(const Box& a, const Box& b)
{
  const float x1 = std::max(a.x1, b.x1);
  const float y1 = std::max(a.y1, b.y1);
  const float x2 = std::min(a.x2, b.x2);
  const float y2 = std::min(a.y2, b.y2);
  const float inter = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
  const float area_a = std::max(0.0f, a.x2 - a.x1) * std::max(0.0f, a.y2 - a.y1);
  const float area_b = std::max(0.0f, b.x2 - b.x1) * std::max(0.0f, b.y2 - b.y1);
  return inter / (area_a + area_b - inter + 1e-6f);
}

static void intersection_stats(const Box& a,
                               const Box& b,
                               float* inter,
                               float* min_area,
                               float* iw,
                               float* ih,
                               float* min_w,
                               float* min_h,
                               float* area_a,
                               float* area_b)
{
  const float x1 = std::max(a.x1, b.x1);
  const float y1 = std::max(a.y1, b.y1);
  const float x2 = std::min(a.x2, b.x2);
  const float y2 = std::min(a.y2, b.y2);
  *iw = std::max(0.0f, x2 - x1);
  *ih = std::max(0.0f, y2 - y1);
  *inter = (*iw) * (*ih);
  const float w1 = std::max(0.0f, a.x2 - a.x1);
  const float h1 = std::max(0.0f, a.y2 - a.y1);
  const float w2 = std::max(0.0f, b.x2 - b.x1);
  const float h2 = std::max(0.0f, b.y2 - b.y1);
  *area_a = w1 * h1;
  *area_b = w2 * h2;
  *min_area = std::min(*area_a, *area_b);
  *min_w = std::min(w1, w2);
  *min_h = std::min(h1, h2);
}

static float center_distance_norm(const Box& a, const Box& b)
{
  const float cx1 = (a.x1 + a.x2) * 0.5f;
  const float cy1 = (a.y1 + a.y2) * 0.5f;
  const float cx2 = (b.x1 + b.x2) * 0.5f;
  const float cy2 = (b.y1 + b.y2) * 0.5f;
  const float h1 = std::max(a.y2 - a.y1, 1.0f);
  const float h2 = std::max(b.y2 - b.y1, 1.0f);
  const float avg_h = (h1 + h2) * 0.5f;
  const float dx = cx1 - cx2;
  const float dy = cy1 - cy2;
  return std::sqrt(dx * dx + dy * dy) / avg_h;
}

static bool old_spatial_duplicate(const Box& a, const Box& b)
{
  float inter = 0.0f;
  float min_area = 0.0f;
  float iw = 0.0f;
  float ih = 0.0f;
  float min_w = 0.0f;
  float min_h = 0.0f;
  float area_a = 0.0f;
  float area_b = 0.0f;
  intersection_stats(a, b, &inter, &min_area, &iw, &ih, &min_w, &min_h, &area_a, &area_b);
  if (min_area <= 0.0f) {
    return false;
  }

  const float min_iou = inter / (min_area + 1e-6f);
  if (min_iou > 0.7f) {
    return true;
  }

  const float cd_norm = center_distance_norm(a, b);
  const float x_cover = min_w > 0.0f ? iw / (min_w + 1e-6f) : 0.0f;
  const float y_cover = min_h > 0.0f ? ih / (min_h + 1e-6f) : 0.0f;
  const float area_ratio = std::min(area_a, area_b) / (std::max(area_a, area_b) + 1e-6f);
  return cd_norm < 0.8f && x_cover > 0.35f && y_cover > 0.35f && area_ratio > 0.25f;
}

static Box convert_box_to_original(const Box& local,
                                   float start_x,
                                   int wrap_around,
                                   float original_width)
{
  Box out = local;
  if (!wrap_around) {
    out.x1 = local.x1 + start_x;
    out.x2 = local.x2 + start_x;
  } else if (start_x < 0.0f) {
    const float right_width = -start_x;
    if (local.x2 <= right_width) {
      out.x1 = local.x1 + (original_width + start_x);
      out.x2 = local.x2 + (original_width + start_x);
    } else if (local.x1 >= right_width) {
      out.x1 = local.x1 - right_width;
      out.x2 = local.x2 - right_width;
    } else {
      out.x1 = local.x1 - right_width;
      out.x2 = local.x2 - right_width;
      if (out.x2 > original_width) {
        out.x2 = original_width;
      }
    }
  } else {
    const float left_width = original_width - start_x;
    if (local.x2 <= left_width) {
      out.x1 = local.x1 + start_x;
      out.x2 = local.x2 + start_x;
    } else if (local.x1 >= left_width) {
      out.x1 = local.x1 - left_width;
      out.x2 = local.x2 - left_width;
    } else {
      out.x1 = local.x1 + start_x;
      out.x2 = local.x2 + start_x;
      if (out.x2 > original_width) {
        out.x2 = original_width;
      }
    }
  }

  out.x1 = clip_f(out.x1, 0.0f, original_width - 1.0f);
  out.x2 = clip_f(out.x2, 0.0f, original_width - 1.0f);
  if (out.x1 > out.x2) {
    std::swap(out.x1, out.x2);
  }
  return out;
}

static float convert_x_to_original(float x,
                                   float start_x,
                                   int wrap_around,
                                   float original_width)
{
  float out = x;
  if (!wrap_around) {
    out = x + start_x;
  } else if (start_x < 0.0f) {
    const float right_width = -start_x;
    out = x <= right_width ? x + (original_width + start_x) : x - right_width;
  } else {
    const float left_width = original_width - start_x;
    out = x <= left_width ? x + start_x : x - left_width;
  }
  return clip_f(out, 0.0f, original_width - 1.0f);
}

static std::vector<int> area_weighted_nms(const std::vector<Detection>& dets,
                                          const std::vector<int>& indices,
                                          float nms_iou_thresh)
{
  if (indices.size() <= 1) {
    return indices;
  }

  float max_area = 0.0f;
  for (int idx : indices) {
    max_area = std::max(max_area, dets[static_cast<size_t>(idx)].area);
  }

  std::vector<int> sorted = indices;
  std::sort(sorted.begin(), sorted.end(), [&](int lhs, int rhs) {
    const Detection& a = dets[static_cast<size_t>(lhs)];
    const Detection& b = dets[static_cast<size_t>(rhs)];
    const float wa = a.score * (0.6f + 0.4f * (a.area / (max_area + 1e-6f)));
    const float wb = b.score * (0.6f + 0.4f * (b.area / (max_area + 1e-6f)));
    return wa > wb;
  });

  std::vector<int> keep;
  std::vector<uint8_t> removed(sorted.size(), 0);
  for (size_t i = 0; i < sorted.size(); ++i) {
    if (removed[i]) {
      continue;
    }
    keep.push_back(sorted[i]);
    const Box& cur = dets[static_cast<size_t>(sorted[i])].box;
    for (size_t j = i + 1; j < sorted.size(); ++j) {
      if (!removed[j] && box_iou(cur, dets[static_cast<size_t>(sorted[j])].box) >= nms_iou_thresh) {
        removed[j] = 1;
      }
    }
  }
  return keep;
}

static std::vector<int> final_spatial_dedup(const std::vector<Detection>& dets,
                                            const std::vector<int>& keep_indices)
{
  if (keep_indices.size() <= 1) {
    return keep_indices;
  }

  std::set<int> labels;
  for (int idx : keep_indices) {
    labels.insert(dets[static_cast<size_t>(idx)].label);
  }

  std::set<int> selected;
  for (int label : labels) {
    std::vector<int> label_indices;
    for (int idx : keep_indices) {
      if (dets[static_cast<size_t>(idx)].label == label) {
        label_indices.push_back(idx);
      }
    }
    if (label_indices.size() <= 1) {
      selected.insert(label_indices.begin(), label_indices.end());
      continue;
    }

    float max_area = 0.0f;
    for (int idx : label_indices) {
      max_area = std::max(max_area, dets[static_cast<size_t>(idx)].area);
    }
    std::sort(label_indices.begin(), label_indices.end(), [&](int lhs, int rhs) {
      const Detection& a = dets[static_cast<size_t>(lhs)];
      const Detection& b = dets[static_cast<size_t>(rhs)];
      const float wa = a.score * (0.6f + 0.4f * (a.area / (max_area + 1e-6f)));
      const float wb = b.score * (0.6f + 0.4f * (b.area / (max_area + 1e-6f)));
      return wa > wb;
    });

    std::vector<int> kept_for_label;
    for (int idx : label_indices) {
      bool duplicate = false;
      for (int kept_idx : kept_for_label) {
        if (old_spatial_duplicate(dets[static_cast<size_t>(idx)].box,
                                  dets[static_cast<size_t>(kept_idx)].box)) {
          duplicate = true;
          break;
        }
      }
      if (!duplicate) {
        kept_for_label.push_back(idx);
      }
    }
    selected.insert(kept_for_label.begin(), kept_for_label.end());
  }

  std::vector<int> final_keep;
  for (int idx : keep_indices) {
    if (selected.count(idx)) {
      final_keep.push_back(idx);
    }
  }
  return final_keep;
}

}  // namespace

extern "C" {

int face_merge_fast(const float* boxes,
                    const float* scores,
                    const int* labels,
                    const float* keypoints,
                    const int* slice_indices,
                    int num_dets,
                    const float* slice_start_x,
                    const int* slice_wrap_around,
                    int num_slices,
                    float original_width,
                    float overlap_ratio,
                    float iou_threshold,
                    float nms_iou_thresh,
                    float* out_boxes,
                    float* out_scores,
                    int* out_labels,
                    float* out_keypoints,
                    int* out_count,
                    int* stats,
                    double* timings,
                    char* err,
                    int err_len)
{
  if (out_count != nullptr) {
    *out_count = 0;
  }
  if (boxes == nullptr || scores == nullptr || labels == nullptr || keypoints == nullptr ||
      slice_indices == nullptr || slice_start_x == nullptr || slice_wrap_around == nullptr ||
      out_boxes == nullptr || out_scores == nullptr || out_labels == nullptr ||
      out_keypoints == nullptr || out_count == nullptr) {
    set_error(err, err_len, "invalid null argument");
    return -1;
  }
  if (num_dets < 0 || num_slices <= 0 || original_width <= 1.0f) {
    set_error(err, err_len, "invalid sizes");
    return -1;
  }

  int64_t t0 = now_us();
  std::vector<Detection> dets;
  dets.reserve(static_cast<size_t>(num_dets));
  for (int i = 0; i < num_dets; ++i) {
    const int slice_idx = slice_indices[i];
    if (slice_idx < 0 || slice_idx >= num_slices) {
      continue;
    }
    Box local;
    local.x1 = boxes[i * 4 + 0];
    local.y1 = boxes[i * 4 + 1];
    local.x2 = boxes[i * 4 + 2];
    local.y2 = boxes[i * 4 + 3];

    Detection det;
    det.box = convert_box_to_original(
        local,
        slice_start_x[slice_idx],
        slice_wrap_around[slice_idx],
        original_width);
    det.score = scores[i];
    det.label = labels[i];
    det.slice_idx = slice_idx;
    det.area = std::max(0.0f, det.box.x2 - det.box.x1) * std::max(0.0f, det.box.y2 - det.box.y1);
    for (int k = 0; k < 5; ++k) {
      const int base = k * 3;
      det.keypoints[base + 0] = convert_x_to_original(
          keypoints[i * 15 + base + 0],
          slice_start_x[slice_idx],
          slice_wrap_around[slice_idx],
          original_width);
      det.keypoints[base + 1] = keypoints[i * 15 + base + 1];
      det.keypoints[base + 2] = keypoints[i * 15 + base + 2];
    }
    dets.push_back(det);
  }
  int64_t t_coords = now_us();

  std::vector<uint8_t> suppressed(dets.size(), 0);
  const float panorama_width = original_width;
  const int last_slice_idx = num_slices - 1;
  const float slice_width = panorama_width / static_cast<float>(num_slices);
  const float max_dist = 3.0f * slice_width * overlap_ratio;

  for (size_t i = 0; i < dets.size(); ++i) {
    if (suppressed[i]) {
      continue;
    }
    for (size_t j = i + 1; j < dets.size(); ++j) {
      if (suppressed[j]) {
        continue;
      }

      const Detection& di = dets[i];
      const Detection& dj = dets[j];
      bool is_same_target = false;

      if (di.slice_idx == dj.slice_idx) {
        float inter = 0.0f;
        float min_area = 0.0f;
        float iw = 0.0f;
        float ih = 0.0f;
        float min_w = 0.0f;
        float min_h = 0.0f;
        float area_i = 0.0f;
        float area_j = 0.0f;
        intersection_stats(di.box, dj.box, &inter, &min_area, &iw, &ih, &min_w, &min_h, &area_i, &area_j);
        const float intra_min_iou = inter / (min_area + 1e-6f);
        is_same_target = intra_min_iou > 0.7f;
      } else {
        const bool is_wrap_pair =
            num_slices >= 3 &&
            ((di.slice_idx == 0 && dj.slice_idx == last_slice_idx) ||
             (dj.slice_idx == 0 && di.slice_idx == last_slice_idx));
        const bool is_adjacent_pair = std::abs(di.slice_idx - dj.slice_idx) == 1;

        if (is_wrap_pair) {
          Box bi = di.box;
          Box bj = dj.box;
          const float cxi = (bi.x1 + bi.x2) * 0.5f;
          const float cxj = (bj.x1 + bj.x2) * 0.5f;
          if (cxj > cxi) {
            bj.x1 -= panorama_width;
            bj.x2 -= panorama_width;
          } else {
            bi.x1 -= panorama_width;
            bi.x2 -= panorama_width;
          }

          float inter = 0.0f;
          float min_area = 0.0f;
          float iw = 0.0f;
          float ih = 0.0f;
          float min_w = 0.0f;
          float min_h = 0.0f;
          float area_i = 0.0f;
          float area_j = 0.0f;
          intersection_stats(bi, bj, &inter, &min_area, &iw, &ih, &min_w, &min_h, &area_i, &area_j);
          const float wrap_min_iou = inter / (min_area + 1e-6f);
          is_same_target = wrap_min_iou > iou_threshold;
        } else if (is_adjacent_pair) {
          const float cxi = (di.box.x1 + di.box.x2) * 0.5f;
          const float cxj = (dj.box.x1 + dj.box.x2) * 0.5f;
          const float boundary_x = static_cast<float>(std::max(di.slice_idx, dj.slice_idx)) * slice_width;
          if (std::fabs(cxi - boundary_x) <= max_dist && std::fabs(cxj - boundary_x) <= max_dist) {
            float inter = 0.0f;
            float min_area = 0.0f;
            float iw = 0.0f;
            float ih = 0.0f;
            float min_w = 0.0f;
            float min_h = 0.0f;
            float area_i = 0.0f;
            float area_j = 0.0f;
            intersection_stats(di.box, dj.box, &inter, &min_area, &iw, &ih, &min_w, &min_h, &area_i, &area_j);
            const float min_iou = inter / (min_area + 1e-6f);
            if (min_iou > iou_threshold) {
              is_same_target = true;
            } else {
              const float cyi = (di.box.y1 + di.box.y2) * 0.5f;
              const float cyj = (dj.box.y1 + dj.box.y2) * 0.5f;
              const float hi = di.box.y2 - di.box.y1;
              const float hj = dj.box.y2 - dj.box.y1;
              const float avg_h = std::max((hi + hj) * 0.5f, 1.0f);
              const float dx = cxi - cxj;
              const float dy = cyi - cyj;
              const float cd_norm = std::sqrt(dx * dx + dy * dy) / avg_h;
              is_same_target = cd_norm < 0.8f;
            }
          }
        }
      }

      if (is_same_target) {
        if (di.score >= dj.score) {
          suppressed[j] = 1;
        } else {
          suppressed[i] = 1;
          break;
        }
      }
    }
  }
  int64_t t_dedup = now_us();

  std::set<int> labels_set;
  for (size_t i = 0; i < dets.size(); ++i) {
    if (!suppressed[i]) {
      labels_set.insert(dets[i].label);
    }
  }
  std::vector<int> keep_indices;
  for (int label : labels_set) {
    std::vector<int> label_indices;
    for (size_t i = 0; i < dets.size(); ++i) {
      if (!suppressed[i] && dets[i].label == label) {
        label_indices.push_back(static_cast<int>(i));
      }
    }
    std::vector<int> kept = area_weighted_nms(dets, label_indices, nms_iou_thresh);
    keep_indices.insert(keep_indices.end(), kept.begin(), kept.end());
  }
  int64_t t_nms = now_us();

  const int nms_keep_count = static_cast<int>(keep_indices.size());
  keep_indices = final_spatial_dedup(dets, keep_indices);
  int64_t t_final = now_us();

  int count = 0;
  for (int idx : keep_indices) {
    const Detection& det = dets[static_cast<size_t>(idx)];
    out_boxes[count * 4 + 0] = det.box.x1;
    out_boxes[count * 4 + 1] = det.box.y1;
    out_boxes[count * 4 + 2] = det.box.x2;
    out_boxes[count * 4 + 3] = det.box.y2;
    out_scores[count] = det.score;
    out_labels[count] = det.label;
    for (int k = 0; k < 15; ++k) {
      out_keypoints[count * 15 + k] = det.keypoints[k];
    }
    ++count;
  }
  *out_count = count;
  int64_t t_build = now_us();

  if (stats != nullptr) {
    stats[0] = static_cast<int>(dets.size());
    stats[1] = nms_keep_count;
    stats[2] = count;
  }
  if (timings != nullptr) {
    timings[0] = (t_coords - t0) / 1000.0;
    timings[1] = (t_dedup - t_coords) / 1000.0;
    timings[2] = (t_nms - t_dedup) / 1000.0;
    timings[3] = (t_final - t_nms) / 1000.0;
    timings[4] = (t_build - t_final) / 1000.0;
  }
  return 0;
}

}  // extern "C"
