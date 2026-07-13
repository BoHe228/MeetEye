#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::high_resolution_clock;

static double elapsed_ms(const Clock::time_point& start, const Clock::time_point& end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

static void set_error(char* err, int err_len, const char* msg) {
  if (err == nullptr || err_len <= 0) {
    return;
  }
  std::snprintf(err, static_cast<size_t>(err_len), "%s", msg);
}

static inline float box_iou(const float* a, const float* b) {
  const float xx1 = std::max(a[0], b[0]);
  const float yy1 = std::max(a[1], b[1]);
  const float xx2 = std::min(a[2], b[2]);
  const float yy2 = std::min(a[3], b[3]);
  const float w = std::max(0.0f, xx2 - xx1);
  const float h = std::max(0.0f, yy2 - yy1);
  const float inter = w * h;
  const float area_a = std::max(0.0f, a[2] - a[0]) * std::max(0.0f, a[3] - a[1]);
  const float area_b = std::max(0.0f, b[2] - b[0]) * std::max(0.0f, b[3] - b[1]);
  const float denom = area_a + area_b - inter;
  if (denom <= 1e-12f) {
    return 0.0f;
  }
  return inter / denom;
}

static std::vector<std::pair<int, int>> hungarian_maximize(const float* scores, int n_dets, int n_trks) {
  std::vector<std::pair<int, int>> result;
  if (n_dets <= 0 || n_trks <= 0) {
    return result;
  }

  const bool transposed = n_dets > n_trks;
  const int rows = transposed ? n_trks : n_dets;
  const int cols = transposed ? n_dets : n_trks;
  const double inf = std::numeric_limits<double>::infinity();
  const double eps = 1e-12;

  std::vector<double> u(static_cast<size_t>(rows + 1), 0.0);
  std::vector<double> v(static_cast<size_t>(cols + 1), 0.0);
  std::vector<int> p(static_cast<size_t>(cols + 1), 0);
  std::vector<int> way(static_cast<size_t>(cols + 1), 0);

  auto weight_at = [&](int row, int col) -> double {
    if (transposed) {
      return static_cast<double>(scores[static_cast<size_t>(col) * n_trks + row]);
    }
    return static_cast<double>(scores[static_cast<size_t>(row) * n_trks + col]);
  };

  for (int i = 1; i <= rows; ++i) {
    p[0] = i;
    int j0 = 0;
    std::vector<double> minv(static_cast<size_t>(cols + 1), inf);
    std::vector<char> used(static_cast<size_t>(cols + 1), 0);
    do {
      used[static_cast<size_t>(j0)] = 1;
      const int i0 = p[static_cast<size_t>(j0)];
      double delta = inf;
      int j1 = 0;
      for (int j = 1; j <= cols; ++j) {
        if (used[static_cast<size_t>(j)]) {
          continue;
        }
        const double cost = -weight_at(i0 - 1, j - 1);
        const double cur = cost - u[static_cast<size_t>(i0)] - v[static_cast<size_t>(j)];
        if (cur < minv[static_cast<size_t>(j)] - eps) {
          minv[static_cast<size_t>(j)] = cur;
          way[static_cast<size_t>(j)] = j0;
        }
        if (minv[static_cast<size_t>(j)] < delta - eps) {
          delta = minv[static_cast<size_t>(j)];
          j1 = j;
        }
      }
      for (int j = 0; j <= cols; ++j) {
        if (used[static_cast<size_t>(j)]) {
          u[static_cast<size_t>(p[static_cast<size_t>(j)])] += delta;
          v[static_cast<size_t>(j)] -= delta;
        } else {
          minv[static_cast<size_t>(j)] -= delta;
        }
      }
      j0 = j1;
    } while (p[static_cast<size_t>(j0)] != 0);

    do {
      const int j1 = way[static_cast<size_t>(j0)];
      p[static_cast<size_t>(j0)] = p[static_cast<size_t>(j1)];
      j0 = j1;
    } while (j0 != 0);
  }

  result.reserve(static_cast<size_t>(rows));
  for (int j = 1; j <= cols; ++j) {
    const int assigned_row = p[static_cast<size_t>(j)];
    if (assigned_row == 0) {
      continue;
    }
    const int row = assigned_row - 1;
    const int col = j - 1;
    if (transposed) {
      result.emplace_back(col, row);
    } else {
      result.emplace_back(row, col);
    }
  }
  std::sort(result.begin(), result.end());
  return result;
}

static void emit_unmatched(int n_dets,
                           int n_trks,
                           const int* matches,
                           int match_count,
                           int* out_unmatched_dets,
                           int* out_unmatched_dets_count,
                           int* out_unmatched_trks,
                           int* out_unmatched_trks_count) {
  std::vector<char> det_matched(static_cast<size_t>(n_dets), 0);
  std::vector<char> trk_matched(static_cast<size_t>(n_trks), 0);
  for (int k = 0; k < match_count; ++k) {
    const int det = matches[static_cast<size_t>(k) * 2];
    const int trk = matches[static_cast<size_t>(k) * 2 + 1];
    if (det >= 0 && det < n_dets) {
      det_matched[static_cast<size_t>(det)] = 1;
    }
    if (trk >= 0 && trk < n_trks) {
      trk_matched[static_cast<size_t>(trk)] = 1;
    }
  }

  int det_count = 0;
  for (int i = 0; i < n_dets; ++i) {
    if (!det_matched[static_cast<size_t>(i)]) {
      out_unmatched_dets[det_count++] = i;
    }
  }
  int trk_count = 0;
  for (int j = 0; j < n_trks; ++j) {
    if (!trk_matched[static_cast<size_t>(j)]) {
      out_unmatched_trks[trk_count++] = j;
    }
  }
  *out_unmatched_dets_count = det_count;
  *out_unmatched_trks_count = trk_count;
}

static inline float corner_direction_cost(float det_y,
                                          float det_x,
                                          float obs_y,
                                          float obs_x,
                                          float vel_y,
                                          float vel_x,
                                          float valid,
                                          float det_score,
                                          float vdc_weight) {
  const float dy = det_y - obs_y;
  const float dx = det_x - obs_x;
  const float norm = std::sqrt(dx * dx + dy * dy) + 1e-6f;
  const float dir_y = dy / norm;
  const float dir_x = dx / norm;
  float dot = vel_x * dir_x + vel_y * dir_y;
  dot = std::max(-1.0f, std::min(1.0f, dot));
  const float diff_angle = (static_cast<float>(M_PI) / 2.0f - std::fabs(std::acos(dot))) /
                           static_cast<float>(M_PI);
  return valid * diff_angle * vdc_weight * det_score;
}

static void filter_by_iou_and_emit(const float* iou_scores,
                                   int n_dets,
                                   int n_trks,
                                   float threshold,
                                   const std::vector<std::pair<int, int>>& assigned,
                                   int* out_matches,
                                   int* out_count,
                                   int* out_unmatched_dets,
                                   int* out_unmatched_dets_count,
                                   int* out_unmatched_trks,
                                   int* out_unmatched_trks_count) {
  int valid_count = 0;
  for (const auto& pair : assigned) {
    const int det = pair.first;
    const int trk = pair.second;
    const float iou = iou_scores[static_cast<size_t>(det) * n_trks + trk];
    if (iou < threshold) {
      continue;
    }
    out_matches[static_cast<size_t>(valid_count) * 2] = det;
    out_matches[static_cast<size_t>(valid_count) * 2 + 1] = trk;
    valid_count += 1;
  }
  *out_count = valid_count;
  emit_unmatched(n_dets, n_trks, out_matches, valid_count,
                 out_unmatched_dets, out_unmatched_dets_count,
                 out_unmatched_trks, out_unmatched_trks_count);
}

}  // namespace

extern "C" {

// Computes the HybridSORT BYTE/OCR second-stage association score matrix and
// handles the full second-stage assignment natively for small BYTE/OCR matrices.
//
// mode:
//   0 = plain IoU score matrix (OCR path)
//   1 = BYTE score matrix: IoU - abs(det_score - trk_simple_score) * score_weight
//
// stats:
//   [0] handled: 1 if native output is final, 0 if caller must run fallback
//   [1] raw candidates above threshold before score adjustment
//   [2] candidates above threshold after score adjustment
//   [3] reason: 1 empty, 2 raw_below, 3 no_adjusted, 4 unique_complete,
//               5 hungarian
//   [4] max row candidate count
//   [5] max column candidate count
//
// timings:
//   [0] total native wall time in ms
int tracker_assoc_fast(const float* dets,
                       int n_dets,
                       int det_stride,
                       const float* trks,
                       int n_trks,
                       int trk_stride,
                       int mode,
                       float threshold,
                       float score_weight,
                       float* out_scores,
                       int* out_matches,
                       int* out_count,
                       int* out_unmatched_dets,
                       int* out_unmatched_dets_count,
                       int* out_unmatched_trks,
                       int* out_unmatched_trks_count,
                       int* stats,
                       double* timings,
                       char* err,
                       int err_len) {
  const auto t0 = Clock::now();

  if (stats != nullptr) {
    std::memset(stats, 0, sizeof(int) * 8);
  }
  if (timings != nullptr) {
    std::memset(timings, 0, sizeof(double) * 4);
  }
  if (out_count != nullptr) {
    *out_count = 0;
  }
  if (out_unmatched_dets_count != nullptr) {
    *out_unmatched_dets_count = 0;
  }
  if (out_unmatched_trks_count != nullptr) {
    *out_unmatched_trks_count = 0;
  }

  if (dets == nullptr || trks == nullptr || out_scores == nullptr ||
      out_matches == nullptr || out_count == nullptr ||
      out_unmatched_dets == nullptr || out_unmatched_dets_count == nullptr ||
      out_unmatched_trks == nullptr || out_unmatched_trks_count == nullptr ||
      stats == nullptr || timings == nullptr) {
    set_error(err, err_len, "null pointer argument");
    return -1;
  }
  if (n_dets < 0 || n_trks < 0 || det_stride < 5 || trk_stride < 5) {
    set_error(err, err_len, "invalid shape/stride argument");
    return -2;
  }
  if (mode != 0 && mode != 1) {
    set_error(err, err_len, "invalid mode");
    return -3;
  }

  if (n_dets == 0 || n_trks == 0) {
    emit_unmatched(n_dets, n_trks, out_matches, 0,
                   out_unmatched_dets, out_unmatched_dets_count,
                   out_unmatched_trks, out_unmatched_trks_count);
    stats[0] = 1;
    stats[3] = 1;
    timings[0] = elapsed_ms(t0, Clock::now());
    return 0;
  }

  float raw_max = -1.0f;
  int raw_candidates = 0;

  for (int i = 0; i < n_dets; ++i) {
    const float* det = dets + static_cast<size_t>(i) * det_stride;
    for (int j = 0; j < n_trks; ++j) {
      const float* trk = trks + static_cast<size_t>(j) * trk_stride;
      const float raw_iou = box_iou(det, trk);
      raw_max = std::max(raw_max, raw_iou);
      if (raw_iou > threshold) {
        raw_candidates += 1;
      }
      out_scores[static_cast<size_t>(i) * n_trks + j] = raw_iou;
    }
  }

  stats[1] = raw_candidates;

  if (raw_max <= threshold) {
    emit_unmatched(n_dets, n_trks, out_matches, 0,
                   out_unmatched_dets, out_unmatched_dets_count,
                   out_unmatched_trks, out_unmatched_trks_count);
    stats[0] = 1;
    stats[3] = 2;
    timings[0] = elapsed_ms(t0, Clock::now());
    return 0;
  }

  std::vector<int> row_counts(static_cast<size_t>(n_dets), 0);
  std::vector<int> col_counts(static_cast<size_t>(n_trks), 0);
  std::vector<int> cand_rows;
  std::vector<int> cand_cols;
  cand_rows.reserve(static_cast<size_t>(std::min(n_dets, n_trks)));
  cand_cols.reserve(static_cast<size_t>(std::min(n_dets, n_trks)));

  int adjusted_candidates = 0;
  float adjusted_max = -1.0f;
  for (int i = 0; i < n_dets; ++i) {
    const float* det = dets + static_cast<size_t>(i) * det_stride;
    for (int j = 0; j < n_trks; ++j) {
      const float* trk = trks + static_cast<size_t>(j) * trk_stride;
      float score = out_scores[static_cast<size_t>(i) * n_trks + j];
      if (mode == 1) {
        const float trk_score = (trk_stride > 5) ? trk[5] : trk[4];
        score -= std::fabs(det[4] - trk_score) * score_weight;
        out_scores[static_cast<size_t>(i) * n_trks + j] = score;
      }
      adjusted_max = std::max(adjusted_max, score);
      if (score > threshold) {
        row_counts[static_cast<size_t>(i)] += 1;
        col_counts[static_cast<size_t>(j)] += 1;
        cand_rows.push_back(i);
        cand_cols.push_back(j);
        adjusted_candidates += 1;
      }
    }
  }
  stats[2] = adjusted_candidates;

  if (adjusted_max < threshold) {
    emit_unmatched(n_dets, n_trks, out_matches, 0,
                   out_unmatched_dets, out_unmatched_dets_count,
                   out_unmatched_trks, out_unmatched_trks_count);
    stats[0] = 1;
    stats[3] = 3;
    timings[0] = elapsed_ms(t0, Clock::now());
    return 0;
  }

  const int max_row = *std::max_element(row_counts.begin(), row_counts.end());
  const int max_col = *std::max_element(col_counts.begin(), col_counts.end());
  stats[4] = max_row;
  stats[5] = max_col;

  const int complete_count = std::min(n_dets, n_trks);
  if (max_row <= 1 && max_col <= 1 && adjusted_candidates == complete_count) {
    for (int k = 0; k < adjusted_candidates; ++k) {
      out_matches[static_cast<size_t>(k) * 2] = cand_rows[static_cast<size_t>(k)];
      out_matches[static_cast<size_t>(k) * 2 + 1] = cand_cols[static_cast<size_t>(k)];
    }
    *out_count = adjusted_candidates;
    emit_unmatched(n_dets, n_trks, out_matches, adjusted_candidates,
                   out_unmatched_dets, out_unmatched_dets_count,
                   out_unmatched_trks, out_unmatched_trks_count);
    stats[0] = 1;
    stats[3] = 4;
    timings[0] = elapsed_ms(t0, Clock::now());
    return 0;
  }

  const std::vector<std::pair<int, int>> assigned = hungarian_maximize(out_scores, n_dets, n_trks);
  int valid_count = 0;
  for (const auto& pair : assigned) {
    const int det = pair.first;
    const int trk = pair.second;
    const float score = out_scores[static_cast<size_t>(det) * n_trks + trk];
    if (score < threshold) {
      continue;
    }
    out_matches[static_cast<size_t>(valid_count) * 2] = det;
    out_matches[static_cast<size_t>(valid_count) * 2 + 1] = trk;
    valid_count += 1;
  }
  *out_count = valid_count;
  emit_unmatched(n_dets, n_trks, out_matches, valid_count,
                 out_unmatched_dets, out_unmatched_dets_count,
                 out_unmatched_trks, out_unmatched_trks_count);

  stats[0] = 1;
  stats[3] = 5;
  timings[0] = elapsed_ms(t0, Clock::now());
  return 0;
}

// Native implementation of HybridSORT first association for the current face_rc
// plain-IoU path:
//   score = IoU + four-corner velocity direction cost
//   optional TCM score adjustment: -abs(det_score - trk_kalman_score) * score_weight
// Filtering still uses raw IoU, matching associate_4_points(_with_score).
int tracker_assoc_first_fast(const float* dets,
                             int n_dets,
                             int det_stride,
                             const float* trks,
                             int n_trks,
                             int trk_stride,
                             const float* vel_lt,
                             const float* vel_rt,
                             const float* vel_lb,
                             const float* vel_rb,
                             const float* previous_obs,
                             int obs_stride,
                             float threshold,
                             float vdc_weight,
                             int use_score,
                             float score_weight,
                             float* out_scores,
                             float* out_iou,
                             int* out_matches,
                             int* out_count,
                             int* out_unmatched_dets,
                             int* out_unmatched_dets_count,
                             int* out_unmatched_trks,
                             int* out_unmatched_trks_count,
                             int* stats,
                             double* timings,
                             char* err,
                             int err_len) {
  const auto t0 = Clock::now();

  if (stats != nullptr) {
    std::memset(stats, 0, sizeof(int) * 8);
  }
  if (timings != nullptr) {
    std::memset(timings, 0, sizeof(double) * 4);
  }
  if (out_count != nullptr) {
    *out_count = 0;
  }
  if (out_unmatched_dets_count != nullptr) {
    *out_unmatched_dets_count = 0;
  }
  if (out_unmatched_trks_count != nullptr) {
    *out_unmatched_trks_count = 0;
  }

  if (dets == nullptr || trks == nullptr || vel_lt == nullptr || vel_rt == nullptr ||
      vel_lb == nullptr || vel_rb == nullptr || previous_obs == nullptr ||
      out_scores == nullptr || out_iou == nullptr || out_matches == nullptr ||
      out_count == nullptr || out_unmatched_dets == nullptr ||
      out_unmatched_dets_count == nullptr || out_unmatched_trks == nullptr ||
      out_unmatched_trks_count == nullptr || stats == nullptr || timings == nullptr) {
    set_error(err, err_len, "null pointer argument");
    return -1;
  }
  if (n_dets < 0 || n_trks < 0 || det_stride < 5 || trk_stride < 5 || obs_stride < 5) {
    set_error(err, err_len, "invalid shape/stride argument");
    return -2;
  }

  if (n_dets == 0 || n_trks == 0) {
    emit_unmatched(n_dets, n_trks, out_matches, 0,
                   out_unmatched_dets, out_unmatched_dets_count,
                   out_unmatched_trks, out_unmatched_trks_count);
    stats[0] = 1;
    stats[3] = 1;
    timings[0] = elapsed_ms(t0, Clock::now());
    return 0;
  }

  float iou_max = -1.0f;
  int candidates = 0;
  std::vector<int> row_counts(static_cast<size_t>(n_dets), 0);
  std::vector<int> col_counts(static_cast<size_t>(n_trks), 0);
  std::vector<int> cand_rows;
  std::vector<int> cand_cols;
  cand_rows.reserve(static_cast<size_t>(std::min(n_dets, n_trks)));
  cand_cols.reserve(static_cast<size_t>(std::min(n_dets, n_trks)));

  for (int i = 0; i < n_dets; ++i) {
    const float* det = dets + static_cast<size_t>(i) * det_stride;
    const float det_score = det[det_stride - 1];
    for (int j = 0; j < n_trks; ++j) {
      const float* trk = trks + static_cast<size_t>(j) * trk_stride;
      const float* obs = previous_obs + static_cast<size_t>(j) * obs_stride;
      const float iou = box_iou(det, trk);
      out_iou[static_cast<size_t>(i) * n_trks + j] = iou;
      iou_max = std::max(iou_max, iou);
      if (iou > threshold) {
        row_counts[static_cast<size_t>(i)] += 1;
        col_counts[static_cast<size_t>(j)] += 1;
        cand_rows.push_back(i);
        cand_cols.push_back(j);
        candidates += 1;
      }

      const float valid = (obs[4] >= 0.0f) ? 1.0f : 0.0f;
      float score = iou;
      score += corner_direction_cost(det[1], det[0], obs[1], obs[0],
                                     vel_lt[static_cast<size_t>(j) * 2],
                                     vel_lt[static_cast<size_t>(j) * 2 + 1],
                                     valid, det_score, vdc_weight);
      score += corner_direction_cost(det[3], det[0], obs[3], obs[0],
                                     vel_rt[static_cast<size_t>(j) * 2],
                                     vel_rt[static_cast<size_t>(j) * 2 + 1],
                                     valid, det_score, vdc_weight);
      score += corner_direction_cost(det[1], det[2], obs[1], obs[2],
                                     vel_lb[static_cast<size_t>(j) * 2],
                                     vel_lb[static_cast<size_t>(j) * 2 + 1],
                                     valid, det_score, vdc_weight);
      score += corner_direction_cost(det[3], det[2], obs[3], obs[2],
                                     vel_rb[static_cast<size_t>(j) * 2],
                                     vel_rb[static_cast<size_t>(j) * 2 + 1],
                                     valid, det_score, vdc_weight);
      if (use_score) {
        const float trk_score = trk[4];
        score -= std::fabs(det_score - trk_score) * score_weight;
      }
      out_scores[static_cast<size_t>(i) * n_trks + j] = score;
    }
  }

  stats[1] = candidates;

  if (iou_max <= threshold) {
    emit_unmatched(n_dets, n_trks, out_matches, 0,
                   out_unmatched_dets, out_unmatched_dets_count,
                   out_unmatched_trks, out_unmatched_trks_count);
    stats[0] = 1;
    stats[3] = 2;
    timings[0] = elapsed_ms(t0, Clock::now());
    return 0;
  }

  const int max_row = *std::max_element(row_counts.begin(), row_counts.end());
  const int max_col = *std::max_element(col_counts.begin(), col_counts.end());
  stats[4] = max_row;
  stats[5] = max_col;

  if (max_row <= 1 && max_col <= 1) {
    int count = 0;
    for (int k = 0; k < candidates; ++k) {
      out_matches[static_cast<size_t>(count) * 2] = cand_rows[static_cast<size_t>(k)];
      out_matches[static_cast<size_t>(count) * 2 + 1] = cand_cols[static_cast<size_t>(k)];
      count += 1;
    }
    *out_count = count;
    emit_unmatched(n_dets, n_trks, out_matches, count,
                   out_unmatched_dets, out_unmatched_dets_count,
                   out_unmatched_trks, out_unmatched_trks_count);
    stats[0] = 1;
    stats[3] = 4;
    timings[0] = elapsed_ms(t0, Clock::now());
    return 0;
  }

  const std::vector<std::pair<int, int>> assigned = hungarian_maximize(out_scores, n_dets, n_trks);
  filter_by_iou_and_emit(out_iou, n_dets, n_trks, threshold, assigned,
                         out_matches, out_count,
                         out_unmatched_dets, out_unmatched_dets_count,
                         out_unmatched_trks, out_unmatched_trks_count);

  stats[0] = 1;
  stats[3] = 5;
  timings[0] = elapsed_ms(t0, Clock::now());
  return 0;
}

}  // extern "C"
