#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::high_resolution_clock;

constexpr double kEps = 1e-6;
constexpr float kPi = 3.14159265358979323846f;

static double elapsed_ms(const Clock::time_point& start, const Clock::time_point& end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

static void set_error(char* err, int err_len, const char* msg) {
  if (err == nullptr || err_len <= 0) {
    return;
  }
  std::snprintf(err, static_cast<size_t>(err_len), "%s", msg);
}

template <typename T>
static T clamp_value(T value, T lo, T hi) {
  return std::max(lo, std::min(hi, value));
}

static inline float box_iou(const double* a, const double* b) {
  const double xx1 = std::max(a[0], b[0]);
  const double yy1 = std::max(a[1], b[1]);
  const double xx2 = std::min(a[2], b[2]);
  const double yy2 = std::min(a[3], b[3]);
  const double w = std::max(0.0, xx2 - xx1);
  const double h = std::max(0.0, yy2 - yy1);
  const double inter = w * h;
  const double area_a = std::max(0.0, a[2] - a[0]) * std::max(0.0, a[3] - a[1]);
  const double area_b = std::max(0.0, b[2] - b[0]) * std::max(0.0, b[3] - b[1]);
  const double denom = area_a + area_b - inter;
  if (denom <= 1e-12) {
    return 0.0f;
  }
  return static_cast<float>(inter / denom);
}

static inline float box_iou_arr(const std::array<double, 5>& a,
                                const std::array<double, 5>& b) {
  return box_iou(a.data(), b.data());
}

static std::vector<std::pair<int, int>> hungarian_maximize(const std::vector<float>& scores,
                                                           int n_dets,
                                                           int n_trks) {
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
                           const std::vector<std::pair<int, int>>& matches,
                           std::vector<int>* unmatched_dets,
                           std::vector<int>* unmatched_trks) {
  unmatched_dets->clear();
  unmatched_trks->clear();
  std::vector<char> det_matched(static_cast<size_t>(n_dets), 0);
  std::vector<char> trk_matched(static_cast<size_t>(n_trks), 0);
  for (const auto& match : matches) {
    if (match.first >= 0 && match.first < n_dets) {
      det_matched[static_cast<size_t>(match.first)] = 1;
    }
    if (match.second >= 0 && match.second < n_trks) {
      trk_matched[static_cast<size_t>(match.second)] = 1;
    }
  }
  for (int i = 0; i < n_dets; ++i) {
    if (!det_matched[static_cast<size_t>(i)]) {
      unmatched_dets->push_back(i);
    }
  }
  for (int j = 0; j < n_trks; ++j) {
    if (!trk_matched[static_cast<size_t>(j)]) {
      unmatched_trks->push_back(j);
    }
  }
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
  const float diff_angle = (kPi / 2.0f - std::fabs(std::acos(dot))) / kPi;
  return valid * diff_angle * vdc_weight * det_score;
}

struct Params {
  double det_thresh = 0.5;
  double low_thresh = 0.1;
  int max_age = 30;
  int min_hits = 1;
  double iou_threshold = 0.3;
  int delta_t = 3;
  double inertia = 0.2;
  bool use_byte = true;
  bool tcm_first_step = true;
  double tcm_first_step_weight = 1.0;
  bool tcm_byte_step = true;
  double tcm_byte_step_weight = 1.0;
  double new_track_thresh = 0.5;
  double new_track_overlap_thresh = 0.6;
  double lost_velocity_decay = 0.85;
};

struct Observation {
  int age = 0;
  std::array<double, 5> box{};
};

static void bbox_to_z(const std::array<double, 5>& bbox, double z[5]) {
  const double w = bbox[2] - bbox[0];
  const double h = bbox[3] - bbox[1];
  z[0] = bbox[0] + w / 2.0;
  z[1] = bbox[1] + h / 2.0;
  z[2] = w * h;
  z[3] = bbox[4];
  z[4] = w / (h + kEps);
}

static void x_to_bbox4(const double x[9], std::array<double, 4>* out) {
  const double area = std::max(x[2], kEps);
  const double aspect = std::max(x[4], kEps);
  const double w = std::sqrt(std::max(area * aspect, kEps));
  const double h = area / std::max(w, kEps);
  (*out)[0] = x[0] - w / 2.0;
  (*out)[1] = x[1] - h / 2.0;
  (*out)[2] = x[0] + w / 2.0;
  (*out)[3] = x[1] + h / 2.0;
}

static void x_to_bbox5(const double x[9], std::array<double, 5>* out) {
  std::array<double, 4> b{};
  x_to_bbox4(x, &b);
  (*out)[0] = b[0];
  (*out)[1] = b[1];
  (*out)[2] = b[2];
  (*out)[3] = b[3];
  (*out)[4] = x[3];
}

static double box_sum(const std::array<double, 5>& box) {
  return box[0] + box[1] + box[2] + box[3] + box[4];
}

static bool is_finite_box6(const std::array<double, 6>& row) {
  for (double v : row) {
    if (!std::isfinite(v)) {
      return false;
    }
  }
  return true;
}

static bool invert_5x5(const double input[5][5], double inv[5][5]) {
  double aug[5][10];
  for (int r = 0; r < 5; ++r) {
    for (int c = 0; c < 5; ++c) {
      aug[r][c] = input[r][c];
      aug[r][c + 5] = (r == c) ? 1.0 : 0.0;
    }
  }

  for (int col = 0; col < 5; ++col) {
    int pivot = col;
    double best = std::fabs(aug[col][col]);
    for (int r = col + 1; r < 5; ++r) {
      const double candidate = std::fabs(aug[r][col]);
      if (candidate > best) {
        best = candidate;
        pivot = r;
      }
    }
    if (best < 1e-12) {
      return false;
    }
    if (pivot != col) {
      for (int c = 0; c < 10; ++c) {
        std::swap(aug[col][c], aug[pivot][c]);
      }
    }
    const double div = aug[col][col];
    for (int c = 0; c < 10; ++c) {
      aug[col][c] /= div;
    }
    for (int r = 0; r < 5; ++r) {
      if (r == col) {
        continue;
      }
      const double factor = aug[r][col];
      if (std::fabs(factor) < 1e-20) {
        continue;
      }
      for (int c = 0; c < 10; ++c) {
        aug[r][c] -= factor * aug[col][c];
      }
    }
  }

  for (int r = 0; r < 5; ++r) {
    for (int c = 0; c < 5; ++c) {
      inv[r][c] = aug[r][c + 5];
    }
  }
  return true;
}

struct Track {
  explicit Track(const std::array<double, 5>& bbox, const Params& params)
      : p(params), confidence(bbox[4]) {
    id = -1;
    time_since_update = 0;
    hits = 0;
    hit_streak = 0;
    age = 0;
    confidence_pre_valid = false;
    last_observation.fill(-1.0);
    last_observation_save.fill(-1.0);
    velocity_lt.fill(0.0);
    velocity_rt.fill(0.0);
    velocity_lb.fill(0.0);
    velocity_rb.fill(0.0);
    has_velocity = false;

    std::memset(x, 0, sizeof(x));
    std::memset(P, 0, sizeof(P));
    std::memset(Q, 0, sizeof(Q));
    std::memset(R, 0, sizeof(R));
    for (int i = 0; i < 9; ++i) {
      P[i][i] = 1.0;
      Q[i][i] = 1.0;
    }
    for (int i = 0; i < 5; ++i) {
      R[i][i] = 1.0;
    }
    for (int i = 2; i < 5; ++i) {
      R[i][i] *= 10.0;
    }
    for (int i = 5; i < 9; ++i) {
      P[i][i] *= 1000.0;
    }
    for (int r = 0; r < 9; ++r) {
      for (int c = 0; c < 9; ++c) {
        P[r][c] *= 10.0;
      }
    }
    Q[8][8] *= 0.01;
    Q[7][7] *= 0.01;
    for (int i = 5; i < 9; ++i) {
      Q[i][i] *= 0.01;
    }

    double z[5];
    bbox_to_z(bbox, z);
    for (int i = 0; i < 5; ++i) {
      x[i] = z[i];
    }
  }

  Params p;
  int id = -1;
  int time_since_update = 0;
  int hits = 0;
  int hit_streak = 0;
  int age = 0;
  double x[9]{};
  double P[9][9]{};
  double Q[9][9]{};
  double R[5][5]{};
  std::array<double, 5> last_observation{};
  std::array<double, 5> last_observation_save{};
  std::vector<Observation> observations;
  std::array<double, 2> velocity_lt{};
  std::array<double, 2> velocity_rt{};
  std::array<double, 2> velocity_lb{};
  std::array<double, 2> velocity_rb{};
  bool has_velocity = false;
  bool confidence_pre_valid = false;
  double confidence_pre = 0.0;
  double confidence = 0.0;

  bool find_observation(int target_age, std::array<double, 5>* out) const {
    for (auto it = observations.rbegin(); it != observations.rend(); ++it) {
      if (it->age == target_age) {
        *out = it->box;
        return true;
      }
    }
    return false;
  }

  std::array<double, 5> k_previous_obs() const {
    if (observations.empty()) {
      return std::array<double, 5>{{-1.0, -1.0, -1.0, -1.0, -1.0}};
    }
    for (int i = 0; i < p.delta_t; ++i) {
      const int dt = p.delta_t - i;
      std::array<double, 5> obs{};
      if (find_observation(age - dt, &obs)) {
        return obs;
      }
    }
    return observations.back().box;
  }

  static std::array<double, 8> corner_speed_scalars(const std::array<double, 5>& prev,
                                                    const std::array<double, 5>& cur) {
    std::array<double, 8> out{};
    double dy = cur[1] - prev[1];
    double dx = cur[0] - prev[0];
    double norm = std::sqrt(dy * dy + dx * dx) + kEps;
    out[0] = dy / norm;
    out[1] = dx / norm;

    dy = cur[3] - prev[3];
    dx = cur[0] - prev[0];
    norm = std::sqrt(dy * dy + dx * dx) + kEps;
    out[2] = dy / norm;
    out[3] = dx / norm;

    dy = cur[1] - prev[1];
    dx = cur[2] - prev[2];
    norm = std::sqrt(dy * dy + dx * dx) + kEps;
    out[4] = dy / norm;
    out[5] = dx / norm;

    dy = cur[3] - prev[3];
    dx = cur[2] - prev[2];
    norm = std::sqrt(dy * dy + dx * dx) + kEps;
    out[6] = dy / norm;
    out[7] = dx / norm;
    return out;
  }

  static void store_normalized_pair(std::array<double, 2>* vec, double y, double x_value) {
    const double norm = std::sqrt(y * y + x_value * x_value) + kEps;
    (*vec)[0] = y / norm;
    (*vec)[1] = x_value / norm;
  }

  void kf_predict() {
    if ((x[7] + x[2]) <= 0.0) {
      x[7] *= 0.0;
    }
    for (int i = 0; i < 4; ++i) {
      x[i] += x[i + 5];
    }
    for (int r = 0; r < 4; ++r) {
      for (int c = 0; c < 9; ++c) {
        P[r][c] += P[r + 5][c];
      }
    }
    for (int r = 0; r < 9; ++r) {
      for (int c = 0; c < 4; ++c) {
        P[r][c] += P[r][c + 5];
      }
    }
    for (int r = 0; r < 9; ++r) {
      for (int c = 0; c < 9; ++c) {
        P[r][c] += Q[r][c];
      }
    }
  }

  void decay_lost_velocity() {
    const double decay = clamp_value(p.lost_velocity_decay, 0.0, 1.0);
    if (decay >= 0.999999) {
      return;
    }
    for (int i = 5; i < 9; ++i) {
      x[i] *= decay;
    }
    for (int i = 0; i < 2; ++i) {
      velocity_lt[i] *= decay;
      velocity_rt[i] *= decay;
      velocity_lb[i] *= decay;
      velocity_rb[i] *= decay;
    }
  }

  bool kf_update(const std::array<double, 5>& bbox) {
    double z[5];
    bbox_to_z(bbox, z);

    double y[5];
    for (int i = 0; i < 5; ++i) {
      y[i] = z[i] - x[i];
    }

    double S[5][5];
    for (int r = 0; r < 5; ++r) {
      for (int c = 0; c < 5; ++c) {
        S[r][c] = P[r][c] + R[r][c];
      }
    }
    double SI[5][5];
    if (!invert_5x5(S, SI)) {
      return false;
    }

    double K[9][5];
    for (int r = 0; r < 9; ++r) {
      for (int c = 0; c < 5; ++c) {
        double value = 0.0;
        for (int k = 0; k < 5; ++k) {
          value += P[r][k] * SI[k][c];
        }
        K[r][c] = value;
      }
    }

    for (int r = 0; r < 9; ++r) {
      double delta = 0.0;
      for (int c = 0; c < 5; ++c) {
        delta += K[r][c] * y[c];
      }
      x[r] += delta;
    }

    double I_KH[9][9];
    for (int r = 0; r < 9; ++r) {
      for (int c = 0; c < 9; ++c) {
        I_KH[r][c] = (r == c) ? 1.0 : 0.0;
      }
      for (int c = 0; c < 5; ++c) {
        I_KH[r][c] -= K[r][c];
      }
    }

    double temp[9][9];
    for (int r = 0; r < 9; ++r) {
      for (int c = 0; c < 9; ++c) {
        double value = 0.0;
        for (int k = 0; k < 9; ++k) {
          value += I_KH[r][k] * P[k][c];
        }
        temp[r][c] = value;
      }
    }

    double new_p[9][9];
    for (int r = 0; r < 9; ++r) {
      for (int c = 0; c < 9; ++c) {
        double value = 0.0;
        for (int k = 0; k < 9; ++k) {
          value += temp[r][k] * I_KH[c][k];
        }
        new_p[r][c] = value;
      }
    }

    double kr[9][5];
    for (int r = 0; r < 9; ++r) {
      for (int c = 0; c < 5; ++c) {
        double value = 0.0;
        for (int k = 0; k < 5; ++k) {
          value += K[r][k] * R[k][c];
        }
        kr[r][c] = value;
      }
    }
    for (int r = 0; r < 9; ++r) {
      for (int c = 0; c < 9; ++c) {
        double value = 0.0;
        for (int k = 0; k < 5; ++k) {
          value += kr[r][k] * K[c][k];
        }
        P[r][c] = new_p[r][c] + value;
      }
    }
    return true;
  }

  std::array<double, 6> predict() {
    kf_predict();
    if (x[2] <= 0.0) {
      x[2] = kEps;
    }
    if (x[4] <= 0.0) {
      x[4] = kEps;
    }

    age += 1;
    if (time_since_update > 0) {
      hit_streak = 0;
    }
    time_since_update += 1;

    std::array<double, 5> bbox5{};
    x_to_bbox5(x, &bbox5);
    double kalman_score = x[3];
    if (kalman_score < p.det_thresh) {
      kalman_score = p.det_thresh;
    } else if (kalman_score > 1.0) {
      kalman_score = 1.0;
    }

    double simple_score = confidence_pre_valid ? (confidence - (confidence_pre - confidence)) : confidence;
    if (simple_score < 0.1) {
      simple_score = 0.1;
    } else if (simple_score > p.det_thresh) {
      simple_score = p.det_thresh;
    }

    return std::array<double, 6>{{bbox5[0], bbox5[1], bbox5[2], bbox5[3],
                                  kalman_score, simple_score}};
  }

  void update(const std::array<double, 5>& bbox) {
    bool kf_damp = false;
    const bool low_score_update = bbox[4] < p.det_thresh;
    if (box_sum(last_observation) >= 0.0) {
      std::array<double, 5> previous_box{};
      bool previous_found = false;
      bool speeds_found = false;
      std::array<double, 8> speeds_sum{};
      speeds_sum.fill(0.0);

      for (int i = 0; i < p.delta_t; ++i) {
        std::array<double, 5> obs{};
        if (find_observation(age - i - 1, &obs)) {
          previous_box = obs;
          previous_found = true;
          const std::array<double, 8> speeds = corner_speed_scalars(previous_box, bbox);
          if (speeds_found) {
            for (int k = 0; k < 8; ++k) {
              speeds_sum[k] += speeds[k];
            }
          } else {
            speeds_sum = speeds;
            speeds_found = true;
          }
        }
      }
      if (!previous_found) {
        previous_box = last_observation;
        speeds_sum = corner_speed_scalars(previous_box, bbox);
        speeds_found = true;
      }

      const double cx_ref = (previous_box[0] + previous_box[2]) * 0.5;
      const double cy_ref = (previous_box[1] + previous_box[3]) * 0.5;
      const double cx_cur = (bbox[0] + bbox[2]) * 0.5;
      const double cy_cur = (bbox[1] + bbox[3]) * 0.5;
      const double disp = std::sqrt((cx_cur - cx_ref) * (cx_cur - cx_ref) +
                                    (cy_cur - cy_ref) * (cy_cur - cy_ref));
      const double avg_h = std::max((previous_box[3] - previous_box[1] +
                                     bbox[3] - bbox[1]) * 0.5,
                                    1.0);
      if (speeds_found && disp >= 0.05 * avg_h) {
        store_normalized_pair(&velocity_lt, speeds_sum[0], speeds_sum[1]);
        store_normalized_pair(&velocity_rt, speeds_sum[2], speeds_sum[3]);
        store_normalized_pair(&velocity_lb, speeds_sum[4], speeds_sum[5]);
        store_normalized_pair(&velocity_rb, speeds_sum[6], speeds_sum[7]);
        has_velocity = true;
      } else if (has_velocity && !low_score_update) {
        for (int i = 0; i < 2; ++i) {
          velocity_lt[i] *= 0.3;
          velocity_rt[i] *= 0.3;
          velocity_lb[i] *= 0.3;
          velocity_rb[i] *= 0.3;
        }
        kf_damp = true;
      }
    }

    last_observation = bbox;
    last_observation_save = bbox;
    observations.push_back(Observation{age, bbox});
    if (observations.size() > 1024) {
      observations.erase(observations.begin(), observations.begin() + 256);
    }

    time_since_update = 0;
    hits += 1;
    hit_streak += 1;
    kf_update(bbox);
    if (kf_damp) {
      x[5] *= 0.3;
      x[6] *= 0.3;
    }
    confidence_pre = confidence;
    confidence_pre_valid = true;
    confidence = bbox[4];
  }

  void update_none() {
    decay_lost_velocity();
    confidence_pre_valid = false;
  }

  std::array<double, 4> get_state() const {
    std::array<double, 4> out{};
    x_to_bbox4(x, &out);
    return out;
  }
};

struct AssocResult {
  std::vector<std::pair<int, int>> matches;
  std::vector<int> unmatched_dets;
  std::vector<int> unmatched_trks;
  int reason = 0;
};

static AssocResult associate_first(const std::vector<std::array<double, 5>>& dets,
                                   const std::vector<std::array<double, 6>>& trks,
                                   const std::vector<std::unique_ptr<Track>>& tracks,
                                   const Params& p) {
  AssocResult result;
  const int n_dets = static_cast<int>(dets.size());
  const int n_trks = static_cast<int>(trks.size());
  if (n_dets == 0 || n_trks == 0) {
    emit_unmatched(n_dets, n_trks, result.matches, &result.unmatched_dets, &result.unmatched_trks);
    result.reason = 1;
    return result;
  }

  std::vector<float> iou_scores(static_cast<size_t>(n_dets) * n_trks, 0.0f);
  std::vector<float> score_matrix(static_cast<size_t>(n_dets) * n_trks, 0.0f);
  std::vector<int> row_counts(static_cast<size_t>(n_dets), 0);
  std::vector<int> col_counts(static_cast<size_t>(n_trks), 0);
  std::vector<std::pair<int, int>> candidates;
  candidates.reserve(static_cast<size_t>(std::min(n_dets, n_trks)));
  float iou_max = -1.0f;

  for (int i = 0; i < n_dets; ++i) {
    const float det_score = static_cast<float>(dets[static_cast<size_t>(i)][4]);
    for (int j = 0; j < n_trks; ++j) {
      const auto& trk_row = trks[static_cast<size_t>(j)];
      const double trk_box[5] = {trk_row[0], trk_row[1], trk_row[2], trk_row[3], trk_row[4]};
      const float iou = box_iou(dets[static_cast<size_t>(i)].data(), trk_box);
      iou_scores[static_cast<size_t>(i) * n_trks + j] = iou;
      iou_max = std::max(iou_max, iou);
      if (iou > p.iou_threshold) {
        row_counts[static_cast<size_t>(i)] += 1;
        col_counts[static_cast<size_t>(j)] += 1;
        candidates.emplace_back(i, j);
      }

      const std::array<double, 5> obs = tracks[static_cast<size_t>(j)]->k_previous_obs();
      const float valid = (obs[4] >= 0.0) ? 1.0f : 0.0f;
      const auto& trk = *tracks[static_cast<size_t>(j)];
      float score = iou;
      score += corner_direction_cost(static_cast<float>(dets[static_cast<size_t>(i)][1]),
                                     static_cast<float>(dets[static_cast<size_t>(i)][0]),
                                     static_cast<float>(obs[1]),
                                     static_cast<float>(obs[0]),
                                     static_cast<float>(trk.velocity_lt[0]),
                                     static_cast<float>(trk.velocity_lt[1]),
                                     valid, det_score, static_cast<float>(p.inertia));
      score += corner_direction_cost(static_cast<float>(dets[static_cast<size_t>(i)][3]),
                                     static_cast<float>(dets[static_cast<size_t>(i)][0]),
                                     static_cast<float>(obs[3]),
                                     static_cast<float>(obs[0]),
                                     static_cast<float>(trk.velocity_rt[0]),
                                     static_cast<float>(trk.velocity_rt[1]),
                                     valid, det_score, static_cast<float>(p.inertia));
      score += corner_direction_cost(static_cast<float>(dets[static_cast<size_t>(i)][1]),
                                     static_cast<float>(dets[static_cast<size_t>(i)][2]),
                                     static_cast<float>(obs[1]),
                                     static_cast<float>(obs[2]),
                                     static_cast<float>(trk.velocity_lb[0]),
                                     static_cast<float>(trk.velocity_lb[1]),
                                     valid, det_score, static_cast<float>(p.inertia));
      score += corner_direction_cost(static_cast<float>(dets[static_cast<size_t>(i)][3]),
                                     static_cast<float>(dets[static_cast<size_t>(i)][2]),
                                     static_cast<float>(obs[3]),
                                     static_cast<float>(obs[2]),
                                     static_cast<float>(trk.velocity_rb[0]),
                                     static_cast<float>(trk.velocity_rb[1]),
                                     valid, det_score, static_cast<float>(p.inertia));
      if (p.tcm_first_step) {
        score -= std::fabs(det_score - static_cast<float>(trk_row[4])) *
                 static_cast<float>(p.tcm_first_step_weight);
      }
      score_matrix[static_cast<size_t>(i) * n_trks + j] = score;
    }
  }

  if (iou_max <= p.iou_threshold) {
    emit_unmatched(n_dets, n_trks, result.matches, &result.unmatched_dets, &result.unmatched_trks);
    result.reason = 2;
    return result;
  }

  const int max_row = *std::max_element(row_counts.begin(), row_counts.end());
  const int max_col = *std::max_element(col_counts.begin(), col_counts.end());
  if (max_row <= 1 && max_col <= 1) {
    result.matches = candidates;
    emit_unmatched(n_dets, n_trks, result.matches, &result.unmatched_dets, &result.unmatched_trks);
    result.reason = 4;
    return result;
  }

  const std::vector<std::pair<int, int>> assigned =
      hungarian_maximize(score_matrix, n_dets, n_trks);
  for (const auto& match : assigned) {
    const float iou = iou_scores[static_cast<size_t>(match.first) * n_trks + match.second];
    if (iou >= p.iou_threshold) {
      result.matches.push_back(match);
    }
  }
  emit_unmatched(n_dets, n_trks, result.matches, &result.unmatched_dets, &result.unmatched_trks);
  result.reason = 5;
  return result;
}

static AssocResult associate_second(const std::vector<std::array<double, 5>>& dets,
                                    const std::vector<std::array<double, 6>>& trks,
                                    const Params& p,
                                    bool byte_mode) {
  AssocResult result;
  const int n_dets = static_cast<int>(dets.size());
  const int n_trks = static_cast<int>(trks.size());
  if (n_dets == 0 || n_trks == 0) {
    emit_unmatched(n_dets, n_trks, result.matches, &result.unmatched_dets, &result.unmatched_trks);
    result.reason = 1;
    return result;
  }

  std::vector<float> score_matrix(static_cast<size_t>(n_dets) * n_trks, 0.0f);
  std::vector<int> row_counts(static_cast<size_t>(n_dets), 0);
  std::vector<int> col_counts(static_cast<size_t>(n_trks), 0);
  std::vector<std::pair<int, int>> candidates;
  candidates.reserve(static_cast<size_t>(std::min(n_dets, n_trks)));
  float raw_max = -1.0f;
  float adjusted_max = -1.0f;

  for (int i = 0; i < n_dets; ++i) {
    for (int j = 0; j < n_trks; ++j) {
      const auto& trk_row = trks[static_cast<size_t>(j)];
      const double trk_box[5] = {trk_row[0], trk_row[1], trk_row[2], trk_row[3], trk_row[4]};
      const float raw_iou = box_iou(dets[static_cast<size_t>(i)].data(), trk_box);
      raw_max = std::max(raw_max, raw_iou);
      float score = raw_iou;
      if (byte_mode && p.tcm_byte_step) {
        score -= std::fabs(static_cast<float>(dets[static_cast<size_t>(i)][4] - trk_row[5])) *
                 static_cast<float>(p.tcm_byte_step_weight);
      }
      adjusted_max = std::max(adjusted_max, score);
      score_matrix[static_cast<size_t>(i) * n_trks + j] = score;
      if (score > p.iou_threshold) {
        row_counts[static_cast<size_t>(i)] += 1;
        col_counts[static_cast<size_t>(j)] += 1;
        candidates.emplace_back(i, j);
      }
    }
  }

  if (raw_max <= p.iou_threshold || adjusted_max < p.iou_threshold) {
    emit_unmatched(n_dets, n_trks, result.matches, &result.unmatched_dets, &result.unmatched_trks);
    result.reason = (raw_max <= p.iou_threshold) ? 2 : 3;
    return result;
  }

  const int max_row = *std::max_element(row_counts.begin(), row_counts.end());
  const int max_col = *std::max_element(col_counts.begin(), col_counts.end());
  if (max_row <= 1 && max_col <= 1 &&
      static_cast<int>(candidates.size()) == std::min(n_dets, n_trks)) {
    result.matches = candidates;
    emit_unmatched(n_dets, n_trks, result.matches, &result.unmatched_dets, &result.unmatched_trks);
    result.reason = 4;
    return result;
  }

  const std::vector<std::pair<int, int>> assigned =
      hungarian_maximize(score_matrix, n_dets, n_trks);
  for (const auto& match : assigned) {
    const float score = score_matrix[static_cast<size_t>(match.first) * n_trks + match.second];
    if (score >= p.iou_threshold) {
      result.matches.push_back(match);
    }
  }
  emit_unmatched(n_dets, n_trks, result.matches, &result.unmatched_dets, &result.unmatched_trks);
  result.reason = 5;
  return result;
}

static std::vector<int> remap_subset_indices(const std::vector<int>& base,
                                             const std::vector<int>& subset_indices) {
  std::vector<int> out;
  out.reserve(subset_indices.size());
  for (int idx : subset_indices) {
    if (idx >= 0 && idx < static_cast<int>(base.size())) {
      out.push_back(base[static_cast<size_t>(idx)]);
    }
  }
  return out;
}

struct HybridSortNative {
  explicit HybridSortNative(const Params& params) : p(params) {}

  Params p;
  int frame_count = 0;
  int next_id = 0;
  std::vector<std::unique_ptr<Track>> trackers;

  int update(const float* input,
             int n_input,
             int det_stride,
             float img_h,
             float img_w,
             float size_h,
             float size_w,
             float* out_rows,
             int max_out_rows,
             int* out_count,
             int* stats,
             int stats_len,
             double* timings,
             int timings_len,
             char* err,
             int err_len) {
    const auto t0 = Clock::now();
    auto last = t0;
    auto mark = [&](int idx) {
      if (timings != nullptr && idx >= 0 && idx < timings_len) {
        const auto now = Clock::now();
        timings[idx] = elapsed_ms(last, now);
        last = now;
      }
    };

    if (out_count != nullptr) {
      *out_count = 0;
    }
    if (stats != nullptr && stats_len > 0) {
      std::memset(stats, 0, sizeof(int) * static_cast<size_t>(stats_len));
    }
    if (timings != nullptr && timings_len > 0) {
      std::memset(timings, 0, sizeof(double) * static_cast<size_t>(timings_len));
    }
    if (input == nullptr || out_rows == nullptr || out_count == nullptr ||
        stats == nullptr || timings == nullptr) {
      set_error(err, err_len, "null pointer argument");
      return -1;
    }
    if (n_input < 0 || det_stride < 5 || max_out_rows < 0) {
      set_error(err, err_len, "invalid shape/stride argument");
      return -2;
    }

    frame_count += 1;
    const int trackers_before = static_cast<int>(trackers.size());
    double scale = 1.0;
    if (img_h > 0.0f && img_w > 0.0f && size_h > 0.0f && size_w > 0.0f) {
      scale = std::min(static_cast<double>(size_h) / img_h,
                       static_cast<double>(size_w) / img_w);
      if (scale <= 0.0 || !std::isfinite(scale)) {
        scale = 1.0;
      }
    }

    std::vector<std::array<double, 5>> dets_high;
    std::vector<std::array<double, 5>> dets_second;
    dets_high.reserve(static_cast<size_t>(n_input));
    dets_second.reserve(static_cast<size_t>(n_input));
    for (int i = 0; i < n_input; ++i) {
      const float* row = input + static_cast<size_t>(i) * det_stride;
      const double score = row[4];
      std::array<double, 5> box{{row[0] / scale, row[1] / scale, row[2] / scale,
                                 row[3] / scale, score}};
      if (score > p.low_thresh && score < p.det_thresh) {
        dets_second.push_back(box);
      }
      if (score > p.det_thresh) {
        dets_high.push_back(box);
      }
    }
    mark(1);

    std::vector<std::array<double, 6>> trk_rows;
    trk_rows.reserve(trackers.size());
    std::vector<std::unique_ptr<Track>> kept;
    kept.reserve(trackers.size());
    for (auto& tracker : trackers) {
      std::array<double, 6> row = tracker->predict();
      if (is_finite_box6(row)) {
        trk_rows.push_back(row);
        kept.push_back(std::move(tracker));
      }
    }
    trackers.swap(kept);
    mark(2);

    std::vector<std::array<double, 6>> last_boxes;
    last_boxes.reserve(trackers.size());
    for (const auto& tracker : trackers) {
      const auto& lo = tracker->last_observation;
      last_boxes.push_back(std::array<double, 6>{{lo[0], lo[1], lo[2], lo[3], lo[4], lo[4]}});
    }
    mark(3);

    AssocResult first = associate_first(dets_high, trk_rows, trackers, p);
    const int matches_first = static_cast<int>(first.matches.size());
    mark(4);

    for (const auto& match : first.matches) {
      trackers[static_cast<size_t>(match.second)]->update(dets_high[static_cast<size_t>(match.first)]);
    }
    mark(5);

    int matches_byte = 0;
    std::vector<int> unmatched_trks = first.unmatched_trks;
    if (p.use_byte && !dets_second.empty() && !unmatched_trks.empty()) {
      std::vector<std::array<double, 6>> u_trks;
      u_trks.reserve(unmatched_trks.size());
      for (int idx : unmatched_trks) {
        if (idx >= 0 && idx < static_cast<int>(trk_rows.size())) {
          u_trks.push_back(trk_rows[static_cast<size_t>(idx)]);
        }
      }
      AssocResult byte = associate_second(dets_second, u_trks, p, true);
      for (const auto& match : byte.matches) {
        const int trk_idx = unmatched_trks[static_cast<size_t>(match.second)];
        trackers[static_cast<size_t>(trk_idx)]->update(dets_second[static_cast<size_t>(match.first)]);
      }
      matches_byte = static_cast<int>(byte.matches.size());
      unmatched_trks = remap_subset_indices(unmatched_trks, byte.unmatched_trks);
    }
    mark(6);

    int matches_ocr = 0;
    std::vector<int> unmatched_dets = first.unmatched_dets;
    if (!unmatched_dets.empty() && !unmatched_trks.empty()) {
      std::vector<std::array<double, 5>> left_dets;
      std::vector<std::array<double, 6>> left_trks;
      left_dets.reserve(unmatched_dets.size());
      left_trks.reserve(unmatched_trks.size());
      for (int idx : unmatched_dets) {
        left_dets.push_back(dets_high[static_cast<size_t>(idx)]);
      }
      for (int idx : unmatched_trks) {
        left_trks.push_back(last_boxes[static_cast<size_t>(idx)]);
      }
      AssocResult ocr = associate_second(left_dets, left_trks, p, false);
      for (const auto& match : ocr.matches) {
        const int det_idx = unmatched_dets[static_cast<size_t>(match.first)];
        const int trk_idx = unmatched_trks[static_cast<size_t>(match.second)];
        trackers[static_cast<size_t>(trk_idx)]->update(dets_high[static_cast<size_t>(det_idx)]);
      }
      matches_ocr = static_cast<int>(ocr.matches.size());
      unmatched_dets = remap_subset_indices(unmatched_dets, ocr.unmatched_dets);
      unmatched_trks = remap_subset_indices(unmatched_trks, ocr.unmatched_trks);
    }
    mark(7);

    for (int idx : unmatched_trks) {
      if (idx >= 0 && idx < static_cast<int>(trackers.size())) {
        trackers[static_cast<size_t>(idx)]->update_none();
      }
    }
    mark(8);

    int created_tracks = 0;
    for (int idx : unmatched_dets) {
      const auto& det = dets_high[static_cast<size_t>(idx)];
      if (det[4] < p.new_track_thresh) {
        continue;
      }
      if (p.new_track_overlap_thresh < 1.0 && !trackers.empty()) {
        bool overlaps_existing = false;
        for (const auto& tracker : trackers) {
          std::array<double, 4> state = tracker->get_state();
          const std::array<double, 5> state_box{{state[0], state[1], state[2], state[3], det[4]}};
          if (box_iou_arr(det, state_box) > p.new_track_overlap_thresh) {
            overlaps_existing = true;
            break;
          }
        }
        if (overlaps_existing) {
          continue;
        }
      }
      trackers.emplace_back(new Track(det, p));
      created_tracks += 1;
    }
    mark(9);

    std::vector<std::array<float, 5>> ret;
    ret.reserve(trackers.size());
    for (int i = static_cast<int>(trackers.size()) - 1; i >= 0; --i) {
      Track& trk = *trackers[static_cast<size_t>(i)];
      std::array<double, 4> d{};
      if (box_sum(trk.last_observation) < 0.0) {
        d = trk.get_state();
      } else {
        d[0] = trk.last_observation[0];
        d[1] = trk.last_observation[1];
        d[2] = trk.last_observation[2];
        d[3] = trk.last_observation[3];
      }
      const bool is_confirmed = (trk.id >= 0) || (trk.hit_streak >= p.min_hits);
      if (trk.time_since_update < 1 && is_confirmed) {
        if (trk.id < 0) {
          trk.id = next_id;
          next_id += 1;
        }
        ret.push_back(std::array<float, 5>{{static_cast<float>(d[0]), static_cast<float>(d[1]),
                                            static_cast<float>(d[2]), static_cast<float>(d[3]),
                                            static_cast<float>(trk.id + 1)}});
      }

      if (trk.time_since_update > p.max_age || (trk.id < 0 && trk.time_since_update >= 1)) {
        trackers.erase(trackers.begin() + i);
      }
    }
    mark(10);

    if (static_cast<int>(ret.size()) > max_out_rows) {
      set_error(err, err_len, "output buffer too small");
      return -3;
    }
    for (size_t i = 0; i < ret.size(); ++i) {
      for (int c = 0; c < 5; ++c) {
        out_rows[i * 5 + c] = ret[i][c];
      }
    }
    *out_count = static_cast<int>(ret.size());

    if (stats_len > 0) stats[0] = n_input;
    if (stats_len > 1) stats[1] = static_cast<int>(dets_high.size());
    if (stats_len > 2) stats[2] = static_cast<int>(dets_second.size());
    if (stats_len > 3) stats[3] = trackers_before;
    if (stats_len > 4) stats[4] = static_cast<int>(trackers.size());
    if (stats_len > 5) stats[5] = matches_first;
    if (stats_len > 6) stats[6] = matches_byte;
    if (stats_len > 7) stats[7] = matches_ocr;
    if (stats_len > 8) stats[8] = created_tracks;
    if (stats_len > 9) stats[9] = static_cast<int>(ret.size());
    if (stats_len > 10) stats[10] = static_cast<int>(unmatched_dets.size());
    if (stats_len > 11) stats[11] = static_cast<int>(unmatched_trks.size());
    if (stats_len > 12) stats[12] = 1;
    if (timings_len > 0) {
      timings[0] = elapsed_ms(t0, Clock::now());
    }
    return 0;
  }
};

}  // namespace

extern "C" {

int hybrid_sort_native_create(float det_thresh,
                              float low_thresh,
                              int max_age,
                              int min_hits,
                              float iou_threshold,
                              int delta_t,
                              float inertia,
                              int use_byte,
                              int tcm_first_step,
                              float tcm_first_step_weight,
                              int tcm_byte_step,
                              float tcm_byte_step_weight,
                              float new_track_thresh,
                              float new_track_overlap_thresh,
                              float lost_velocity_decay,
                              void** out_handle,
                              char* err,
                              int err_len) {
  if (out_handle == nullptr) {
    set_error(err, err_len, "null output handle");
    return -1;
  }
  Params p;
  p.det_thresh = det_thresh;
  p.low_thresh = low_thresh;
  p.max_age = std::max(0, max_age);
  p.min_hits = std::max(1, min_hits);
  p.iou_threshold = iou_threshold;
  p.delta_t = std::max(1, delta_t);
  p.inertia = inertia;
  p.use_byte = (use_byte != 0);
  p.tcm_first_step = (tcm_first_step != 0);
  p.tcm_first_step_weight = tcm_first_step_weight;
  p.tcm_byte_step = (tcm_byte_step != 0);
  p.tcm_byte_step_weight = tcm_byte_step_weight;
  p.new_track_thresh = new_track_thresh;
  p.new_track_overlap_thresh = new_track_overlap_thresh;
  p.lost_velocity_decay = clamp_value(static_cast<double>(lost_velocity_decay), 0.0, 1.0);
  *out_handle = new HybridSortNative(p);
  return 0;
}

int hybrid_sort_native_update(void* handle,
                              const float* input,
                              int n_input,
                              int det_stride,
                              float img_h,
                              float img_w,
                              float size_h,
                              float size_w,
                              float* out_rows,
                              int max_out_rows,
                              int* out_count,
                              int* stats,
                              int stats_len,
                              double* timings,
                              int timings_len,
                              char* err,
                              int err_len) {
  if (handle == nullptr) {
    set_error(err, err_len, "null handle");
    return -1;
  }
  return static_cast<HybridSortNative*>(handle)->update(
      input, n_input, det_stride, img_h, img_w, size_h, size_w,
      out_rows, max_out_rows, out_count, stats, stats_len, timings, timings_len, err, err_len);
}

int hybrid_sort_native_get_tracks(void* handle,
                                  float* out_rows,
                                  int max_rows,
                                  int* out_count,
                                  char* err,
                                  int err_len) {
  if (out_count != nullptr) {
    *out_count = 0;
  }
  if (handle == nullptr || out_rows == nullptr || out_count == nullptr) {
    set_error(err, err_len, "null pointer argument");
    return -1;
  }
  HybridSortNative* native = static_cast<HybridSortNative*>(handle);
  const int count = static_cast<int>(native->trackers.size());
  if (count > max_rows) {
    set_error(err, err_len, "track output buffer too small");
    return -2;
  }
  for (int i = 0; i < count; ++i) {
    const Track& trk = *native->trackers[static_cast<size_t>(i)];
    const std::array<double, 4> state = trk.get_state();
    float* row = out_rows + static_cast<size_t>(i) * 11;
    row[0] = static_cast<float>(trk.id);
    row[1] = static_cast<float>(trk.time_since_update);
    for (int c = 0; c < 5; ++c) {
      row[2 + c] = static_cast<float>(trk.last_observation[static_cast<size_t>(c)]);
    }
    for (int c = 0; c < 4; ++c) {
      row[7 + c] = static_cast<float>(state[static_cast<size_t>(c)]);
    }
  }
  *out_count = count;
  return 0;
}

int hybrid_sort_native_reset_ids(void* handle, char* err, int err_len) {
  if (handle == nullptr) {
    set_error(err, err_len, "null handle");
    return -1;
  }
  static_cast<HybridSortNative*>(handle)->next_id = 0;
  return 0;
}

void hybrid_sort_native_destroy(void* handle) {
  delete static_cast<HybridSortNative*>(handle);
}

}  // extern "C"
