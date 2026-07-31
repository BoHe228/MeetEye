class FaceRecognitionRuntime {
 public:
  struct FaceUpdateDecision {
    std::string kind;
    std::string reason;
    float best_existing = 0.0f;
    float best_other = 0.0f;
    float sample_similarity = 0.0f;
    float pose_library_similarity = 0.0f;
    float pose_library_max_similarity = 0.0f;
    bool has_pose_library_similarity = false;
    int pose_sample_missing_frames = -1;
    bool pose_sample_quality_ok = false;
    bool locked_pose_quality_ok = false;

    bool updated() const { return !kind.empty(); }
  };

  FaceRecognitionRuntime(const Config& cfg,
                         const MapMeta& meta,
                         const std::vector<float>* base_map_x,
                         const std::vector<float>* base_map_y)
      : cfg_(cfg), meta_(meta), base_map_x_(base_map_x), base_map_y_(base_map_y) {
    char err[4096] = {0};
    std::vector<int> core_masks;
    if (cfg_.face_rec_async || is_wide_angle_projection(meta_.projection_mode)) {
      core_masks.push_back(4);
      adaface_runner_count_ = 1;
    } else {
      core_masks.push_back(1);
      core_masks.push_back(2);
      core_masks.push_back(4);
      adaface_runner_count_ =
          cfg_.face_rec_max_per_frame > 0 ? cfg_.face_rec_max_per_frame : 3;
      adaface_runner_count_ = std::max(1, std::min(adaface_runner_count_, 3));
    }
    for (int i = 0; i < adaface_runner_count_; ++i) {
      int ret = adaface_rknn_create_with_core_mask(
          cfg_.face_rec_model_path.c_str(),
          core_masks[static_cast<size_t>(i)],
          adafaces_[static_cast<size_t>(i)].out(),
          err,
          sizeof(err));
      if (ret != 0) {
        throw std::runtime_error(std::string("adaface_rknn_create failed: ") + err);
      }
    }
    int ret = adaface_rknn_get_shape(
        adafaces_[0].get(), &input_h_, &input_w_, &input_c_, &feature_dim_, err, sizeof(err));
    if (ret != 0) {
      throw std::runtime_error(std::string("adaface_rknn_get_shape failed: ") + err);
    }
    if (input_h_ != 112 || input_w_ != 112 || input_c_ != 3 || feature_dim_ <= 0) {
      std::ostringstream oss;
      oss << "unsupported AdaFace RKNN shape: input="
          << input_w_ << "x" << input_h_ << "x" << input_c_
          << " feature_dim=" << feature_dim_;
      throw std::runtime_error(oss.str());
    }
    if (cfg_.face_rec_dynamic_library) {
      mkdir(cfg_.face_rec_dynamic_library_dir.c_str(), 0775);
    }
    if (cfg_.face_rec_debug_crops) {
      ensure_dir_exists(cfg_.face_rec_debug_crop_dir);
    }
    load_feature_library(cfg_.face_rec_library_dir);
    if (cfg_.face_rec_async) {
      start_async_worker();
    }
    std::cerr << "[FaceRecCpp] enabled: model=" << cfg_.face_rec_model_path
              << " static_ids=" << identities_.size()
              << " dynamic=" << (cfg_.face_rec_dynamic_library ? "on" : "off")
              << " runners=" << adaface_runner_count_
              << " core_masks=";
    for (int i = 0; i < adaface_runner_count_; ++i) {
      if (i > 0) {
        std::cerr << ",";
      }
      std::cerr << core_masks[static_cast<size_t>(i)];
    }
    std::cerr << " bind=raw_track_id";
    if (cfg_.face_rec_async) {
      std::cerr << " async=on"
                << " max_pending=" << cfg_.face_rec_async_max_pending
                << " max_age_frames=" << cfg_.face_rec_async_max_age;
    }
    if (cfg_.face_rec_debug_crops) {
      std::cerr << " debug_crops=" << cfg_.face_rec_debug_crop_dir;
    }
    std::cerr << "\n";
  }

  ~FaceRecognitionRuntime() {
    stop_async_worker();
  }

  struct FaceRecInferTask {
    int frame_id = 0;
    int binding_id = 0;
    int public_id = 0;
    int display_id = 0;
    std::array<float, kDetectionFields> det{};
    std::vector<float> input;
    std::vector<float> feature;
    int ret = 0;
    std::string error;
  };

  struct DisplayFaceMemory {
    int binding_id = 0;
    int public_id = 0;
    int frame_id = 0;
    std::string face_id;
    std::array<float, 4> bbox{{0.0f, 0.0f, 0.0f, 0.0f}};
  };

  static int public_id_for_detection(const FrameResult& result, int idx) {
    if (idx >= 0 && idx < static_cast<int>(result.track_ids.size()) &&
        result.track_ids[static_cast<size_t>(idx)] > 0) {
      return result.track_ids[static_cast<size_t>(idx)];
    }
    return -1;
  }

  static int face_binding_id_for_detection(const FrameResult& result, int idx) {
    if (idx >= 0 && idx < static_cast<int>(result.raw_track_ids.size()) &&
        result.raw_track_ids[static_cast<size_t>(idx)] > 0) {
      return result.raw_track_ids[static_cast<size_t>(idx)];
    }
    return -1;
  }

  void process_frame(const Image& source, FrameResult& result, int frame_id) {
    std::set<std::string> used_this_frame;
    std::vector<int> order;
    for (int i = 0; i < result.detection_count; ++i) {
      order.push_back(i);
    }
    std::sort(order.begin(), order.end(), [&](int a, int b) {
      const float* da = result.detections.data() + static_cast<size_t>(a) * kDetectionFields;
      const float* db = result.detections.data() + static_cast<size_t>(b) * kDetectionFields;
      const float aa = std::max(0.0f, da[2] - da[0]) * std::max(0.0f, da[3] - da[1]);
      const float ab = std::max(0.0f, db[2] - db[0]) * std::max(0.0f, db[3] - db[1]);
      if (std::abs(aa - ab) > 1.0f) {
        return aa > ab;
      }
      return da[4] > db[4];
    });

    std::set<int> active_public_ids;
    std::set<int> active_binding_ids;
    binding_reappeared_after_gap_.clear();
    std::map<std::string, int> reserved_face_owner;
    for (int i = 0; i < result.detection_count; ++i) {
      const int public_id = public_id_for_detection(result, i);
      const int binding_id = face_binding_id_for_detection(result, i);
      if (public_id > 0) {
        active_public_ids.insert(public_id);
      }
      if (binding_id <= 0) {
        continue;
      }
      active_binding_ids.insert(binding_id);
      const auto bind_it = bindings_.find(binding_id);
      if (bind_it == bindings_.end() ||
          !bind_it->second.has_face ||
          bind_it->second.face_id.empty()) {
        continue;
      }
      auto owner_it = reserved_face_owner.find(bind_it->second.face_id);
      if (owner_it == reserved_face_owner.end() ||
          binding_id < owner_it->second) {
        reserved_face_owner[bind_it->second.face_id] = binding_id;
      }
    }
    auto face_reserved_by_other =
        [&](const std::string& face_id, int binding_id) -> bool {
      const auto owner_it = reserved_face_owner.find(face_id);
      return owner_it != reserved_face_owner.end() && owner_it->second != binding_id;
    };
    auto reserved_owner_for_face = [&](const std::string& face_id) -> int {
      const auto owner_it = reserved_face_owner.find(face_id);
      return owner_it != reserved_face_owner.end() ? owner_it->second : 0;
    };

    std::vector<FaceRecInferTask> infer_tasks;
    std::set<int> queued_binding_ids;
    std::map<int, std::vector<float>> frame_features;
    int scheduled_face_rec_this_frame = 0;
    for (int idx : order) {
      if (idx < 0 || idx >= result.detection_count) {
        continue;
      }
      const int public_id = public_id_for_detection(result, idx);
      const int binding_id = face_binding_id_for_detection(result, idx);
      if (public_id > 0) {
        active_public_ids.insert(public_id);
      }
      if (binding_id <= 0) {
        continue;
      }
      active_binding_ids.insert(binding_id);
      const auto previous_active_it = last_active_frame_.find(binding_id);
      if (previous_active_it != last_active_frame_.end() &&
          frame_id - previous_active_it->second > 1) {
        binding_reappeared_after_gap_.insert(binding_id);
        reappeared_probe_until_frame_[binding_id] =
            frame_id + cfg_.face_rec_reappeared_probe_frames;
      }
      last_active_frame_[binding_id] = frame_id;
      const float* det = result.detections.data() + static_cast<size_t>(idx) * kDetectionFields;
      FaceRecognitionBinding carry;
      auto bind_it = bindings_.find(binding_id);
      if (bind_it != bindings_.end() &&
          bind_it->second.has_face &&
          face_reserved_by_other(bind_it->second.face_id, binding_id)) {
        last_attempt_frame_.erase(binding_id);
        reappeared_probe_until_frame_.erase(binding_id);
        current_.erase(binding_id);
        bindings_.erase(bind_it);
        bind_it = bindings_.end();
      }
      if (bind_it == bindings_.end()) {
        maybe_start_tentative_face_from_display(
            result,
            idx,
            public_id,
            binding_id,
            det,
            frame_id,
            active_binding_ids,
            &used_this_frame,
            &reserved_face_owner);
        bind_it = bindings_.find(binding_id);
      }
      if (bind_it != bindings_.end()) {
        carry = bind_it->second;
        carry.quality_passed = false;
        carry.updated = false;
        carry.event = carry.tentative ? "tentative_face_carry" : "carry";
        carry.has_debug = true;
        carry.debug_reason = carry.event;
        current_[binding_id] = carry;
        if (carry.has_face && !carry.face_id.empty()) {
          used_this_frame.insert(carry.face_id);
        }
      }

      std::string quality_reason;
      if (!quality_pass(det, &quality_reason)) {
        FaceRecognitionBinding debug_binding;
        if (bind_it != bindings_.end()) {
          debug_binding = bind_it->second;
          debug_binding.event =
              debug_binding.tentative ? "tentative_face_carry" : "carry";
        } else {
          pending_dynamic_enroll_.erase(binding_id);
        }
        debug_binding.has_debug = true;
        debug_binding.updated = false;
        debug_binding.debug_reason = quality_reason.empty() ? "quality_gate" : quality_reason;
        debug_binding.debug_rknn_ran = false;
        debug_binding.quality_passed = false;
        current_[binding_id] = debug_binding;
        continue;
      }

      const FaceRecognitionBinding* cooldown_binding =
          bind_it != bindings_.end() ? &bind_it->second : nullptr;
      const std::string cooldown_state =
          face_rec_cooldown_state(binding_id, cooldown_binding, frame_id);
      const int cooldown_frames = face_rec_cooldown_frames(cooldown_state);
      const auto last_it = last_attempt_frame_.find(binding_id);
      const bool cooldown = last_it != last_attempt_frame_.end() &&
                            frame_id - last_it->second < cooldown_frames;
      if (cooldown) {
        FaceRecognitionBinding debug_binding;
        if (bind_it != bindings_.end()) {
          debug_binding = bind_it->second;
          debug_binding.event =
              debug_binding.tentative ? "tentative_face_carry" : "carry";
        }
        debug_binding.quality_passed = true;
        debug_binding.updated = false;
        debug_binding.has_debug = true;
        debug_binding.debug_reason = std::string("cooldown_") + cooldown_state;
        debug_binding.debug_rknn_ran = false;
        current_[binding_id] = debug_binding;
        continue;
      }
      if (queued_binding_ids.find(binding_id) != queued_binding_ids.end()) {
        current_[binding_id].quality_passed = true;
        current_[binding_id].updated = false;
        current_[binding_id].has_debug = true;
        current_[binding_id].debug_reason = "pending_same_raw";
        current_[binding_id].debug_rknn_ran = false;
        continue;
      }
      if (cfg_.face_rec_max_per_frame > 0 &&
          scheduled_face_rec_this_frame >= cfg_.face_rec_max_per_frame) {
        current_[binding_id].quality_passed = true;
        current_[binding_id].updated = false;
        current_[binding_id].has_debug = true;
        current_[binding_id].debug_reason = "max_per_frame";
        current_[binding_id].debug_rknn_ran = false;
        continue;
      }

      std::vector<float> input;
      if (!extract_aligned_face_tensor(source, det, &input)) {
        FaceRecognitionBinding debug_binding;
        if (bind_it != bindings_.end()) {
          debug_binding = bind_it->second;
          debug_binding.event =
              debug_binding.tentative ? "tentative_face_carry" : "carry";
        }
        debug_binding.has_debug = true;
        debug_binding.updated = false;
        debug_binding.debug_reason = "extract_failed";
        debug_binding.debug_rknn_ran = false;
        debug_binding.quality_passed = true;
        current_[binding_id] = debug_binding;
        continue;
      }
      int display_id = public_id;
      if (idx < static_cast<int>(result.display_ids.size()) &&
          result.display_ids[static_cast<size_t>(idx)] > 0) {
        display_id = result.display_ids[static_cast<size_t>(idx)];
      }
      save_debug_crops(source, result.frame_index, public_id, display_id, det, input);
      FaceRecInferTask task;
      task.frame_id = frame_id;
      task.binding_id = binding_id;
      task.public_id = public_id;
      task.display_id = display_id;
      std::copy(det, det + kDetectionFields, task.det.begin());
      task.input = std::move(input);
      task.feature.assign(static_cast<size_t>(feature_dim_), 0.0f);
      if (cfg_.face_rec_async) {
        const std::string async_reason = submit_async_task(std::move(task));
        FaceRecognitionBinding debug_binding;
        if (bind_it != bindings_.end()) {
          debug_binding = bind_it->second;
          debug_binding.event =
              debug_binding.tentative ? "tentative_face_carry" : "carry";
        }
        debug_binding.quality_passed = true;
        debug_binding.updated = false;
        debug_binding.has_debug = true;
        debug_binding.debug_reason = async_reason;
        debug_binding.debug_rknn_ran = false;
        current_[binding_id] = debug_binding;
        if (async_reason == "async_submitted") {
          queued_binding_ids.insert(binding_id);
          scheduled_face_rec_this_frame += 1;
        }
        if (async_reason == "async_submitted" ||
            async_reason == "async_inflight" ||
            async_reason == "async_queue_full") {
          last_attempt_frame_[binding_id] = frame_id;
        }
        continue;
      }
      infer_tasks.push_back(std::move(task));
      queued_binding_ids.insert(binding_id);
      last_attempt_frame_[binding_id] = frame_id;
      scheduled_face_rec_this_frame += 1;
      continue;
    }

    if (cfg_.face_rec_async) {
      collect_async_results(
          frame_id,
          active_binding_ids,
          &used_this_frame,
          &reserved_face_owner,
          &frame_features);
    } else {
      run_infer_tasks(&infer_tasks);
      for (const FaceRecInferTask& task : infer_tasks) {
        apply_infer_result(
            task,
            frame_id,
            &used_this_frame,
            &reserved_face_owner,
            &frame_features);
      }
    }

    for (auto& item : current_) {
      FaceRecognitionBinding& binding = item.second;
      if (!binding.has_debug) {
        continue;
      }
      const auto self_it = frame_features.find(item.first);
      if (self_it == frame_features.end()) {
        continue;
      }
      int nearest_public_id = 0;
      float nearest_score = -2.0f;
      for (const auto& other : frame_features) {
        if (other.first == item.first || other.second.size() != self_it->second.size()) {
          continue;
        }
        const float score = dot(self_it->second, other.second);
        if (nearest_public_id <= 0 || score > nearest_score) {
          nearest_public_id = other.first;
          nearest_score = score;
        }
      }
      if (nearest_public_id > 0) {
        binding.debug_has_nearest_feature = true;
        binding.debug_nearest_public_id = nearest_public_id;
        binding.debug_nearest_feature_score = nearest_score;
      }
    }

    for (auto it = bindings_.begin(); it != bindings_.end();) {
      if (active_binding_ids.find(it->first) == active_binding_ids.end()) {
        const auto active_it = last_active_frame_.find(it->first);
        const int last_active =
            active_it != last_active_frame_.end() ? active_it->second : frame_id;
        if (frame_id - last_active > cfg_.face_rec_binding_ttl) {
          last_attempt_frame_.erase(it->first);
          last_active_frame_.erase(it->first);
          reappeared_probe_until_frame_.erase(it->first);
          pending_face_relink_.erase(it->first);
          it = bindings_.erase(it);
        } else {
          ++it;
        }
      } else {
        ++it;
      }
    }
    for (auto it = current_.begin(); it != current_.end();) {
      if (active_binding_ids.find(it->first) == active_binding_ids.end()) {
        pending_face_relink_.erase(it->first);
        it = current_.erase(it);
      } else {
        ++it;
      }
    }
    update_display_face_memory(result, frame_id);
    cleanup_display_face_memory(frame_id);
    cleanup_rejected_tentative_faces(active_binding_ids);
    cleanup_pending_dynamic_enroll(active_binding_ids, frame_id);
  }

  void run_infer_tasks(std::vector<FaceRecInferTask>* tasks) {
    if (tasks == nullptr || tasks->empty()) {
      return;
    }
    const int runner_count = std::max(1, std::min(adaface_runner_count_, 3));
    for (size_t offset = 0; offset < tasks->size(); offset += static_cast<size_t>(runner_count)) {
      const int wave_count = static_cast<int>(
          std::min(static_cast<size_t>(runner_count), tasks->size() - offset));
      std::vector<std::future<void>> futures;
      futures.reserve(static_cast<size_t>(wave_count));
      for (int i = 0; i < wave_count; ++i) {
        FaceRecInferTask* task = &(*tasks)[offset + static_cast<size_t>(i)];
        void* handle = adafaces_[static_cast<size_t>(i)].get();
        futures.emplace_back(std::async(std::launch::async, [this, task, handle]() {
          run_single_infer_task(task, handle);
        }));
      }
      for (std::future<void>& future : futures) {
        future.get();
      }
    }
  }

  void run_single_infer_task(FaceRecInferTask* task, void* handle) {
    if (task == nullptr || handle == nullptr) {
      if (task != nullptr) {
        task->ret = -1;
        task->error = "invalid AdaFace runner";
      }
      return;
    }
    try {
      char err[4096] = {0};
      task->ret = adaface_rknn_infer(
          handle,
          task->input.data(),
          static_cast<int>(task->input.size()),
          task->feature.data(),
          static_cast<int>(task->feature.size()),
          err,
          sizeof(err));
      task->error = err;
    } catch (const std::exception& e) {
      task->ret = -1;
      task->error = e.what();
    } catch (...) {
      task->ret = -1;
      task->error = "unknown AdaFace inference error";
    }
  }

  void start_async_worker() {
    if (async_worker_.joinable()) {
      return;
    }
    async_stop_ = false;
    async_worker_ = std::thread([this]() { async_worker_loop(); });
  }

  void stop_async_worker() {
    {
      std::lock_guard<std::mutex> lock(async_mutex_);
      async_stop_ = true;
    }
    async_cv_.notify_all();
    if (async_worker_.joinable()) {
      async_worker_.join();
    }
    {
      std::lock_guard<std::mutex> lock(async_mutex_);
      async_queue_.clear();
      async_done_.clear();
      async_pending_bindings_.clear();
    }
  }

  void async_worker_loop() {
    for (;;) {
      FaceRecInferTask task;
      {
        std::unique_lock<std::mutex> lock(async_mutex_);
        async_cv_.wait(lock, [this]() {
          return async_stop_ || !async_queue_.empty();
        });
        if (async_stop_) {
          return;
        }
        task = std::move(async_queue_.front());
        async_queue_.pop_front();
      }

      run_single_infer_task(&task, adafaces_[0].get());

      {
        std::lock_guard<std::mutex> lock(async_mutex_);
        async_done_.push_back(std::move(task));
      }
    }
  }

  std::string submit_async_task(FaceRecInferTask task) {
    if (!cfg_.face_rec_async) {
      return "async_disabled";
    }
    if (task.binding_id <= 0) {
      return "async_invalid";
    }
    std::lock_guard<std::mutex> lock(async_mutex_);
    if (async_pending_bindings_.find(task.binding_id) != async_pending_bindings_.end()) {
      return "async_inflight";
    }
    if (static_cast<int>(async_pending_bindings_.size()) >=
        cfg_.face_rec_async_max_pending) {
      return "async_queue_full";
    }
    async_pending_bindings_.insert(task.binding_id);
    async_queue_.push_back(std::move(task));
    async_cv_.notify_one();
    return "async_submitted";
  }

  void collect_async_results(
      int frame_id,
      const std::set<int>& active_binding_ids,
      std::set<std::string>* used_this_frame,
      std::map<std::string, int>* reserved_face_owner,
      std::map<int, std::vector<float>>* frame_features) {
    std::deque<FaceRecInferTask> done;
    {
      std::lock_guard<std::mutex> lock(async_mutex_);
      done.swap(async_done_);
      for (const FaceRecInferTask& task : done) {
        async_pending_bindings_.erase(task.binding_id);
      }
    }
    for (const FaceRecInferTask& task : done) {
      if (task.binding_id <= 0 ||
          active_binding_ids.find(task.binding_id) == active_binding_ids.end()) {
        continue;
      }
      if (task.frame_id <= 0 ||
          frame_id - task.frame_id > cfg_.face_rec_async_max_age) {
        auto current_it = current_.find(task.binding_id);
        if (current_it != current_.end()) {
          current_it->second.has_debug = true;
          current_it->second.debug_reason = "async_stale_drop";
          current_it->second.debug_rknn_ran = false;
        }
        continue;
      }
      apply_infer_result(
          task,
          frame_id,
          used_this_frame,
          reserved_face_owner,
          frame_features);
    }
  }

  std::string face_rec_cooldown_state(
      int binding_id,
      const FaceRecognitionBinding* binding,
      int frame_id) const {
    const auto reappeared_it = reappeared_probe_until_frame_.find(binding_id);
    if ((binding != nullptr && binding->tentative) ||
        binding_reappeared_after_gap_.find(binding_id) !=
            binding_reappeared_after_gap_.end() ||
        (reappeared_it != reappeared_probe_until_frame_.end() &&
         frame_id <= reappeared_it->second)) {
      return "reappeared_after_gap";
    }
    if (binding == nullptr || !binding->has_face || binding->face_id.empty()) {
      return "unbound";
    }
    if (binding->face_check_count < cfg_.face_rec_newly_bound_checks) {
      return "newly_bound";
    }
    return "stable_bound";
  }

  int face_rec_cooldown_frames(const std::string& state) const {
    if (state == "reappeared_after_gap") {
      return cfg_.face_rec_reappeared_cooldown;
    }
    if (state == "newly_bound") {
      return cfg_.face_rec_newly_bound_cooldown;
    }
    if (state == "stable_bound") {
      return cfg_.face_rec_stable_cooldown;
    }
    return cfg_.face_rec_cooldown;
  }

  void mark_face_check_success(int binding_id,
                               int frame_id,
                               FaceRecognitionBinding* binding) {
    if (binding == nullptr || !binding->has_face || binding->face_id.empty()) {
      return;
    }
    int previous_count = 0;
    int first_frame = frame_id;
    const auto old_it = bindings_.find(binding_id);
    if (old_it != bindings_.end() &&
        old_it->second.has_face &&
        old_it->second.face_id == binding->face_id) {
      previous_count = std::max(0, old_it->second.face_check_count);
      first_frame =
          old_it->second.face_bound_frame > 0 ? old_it->second.face_bound_frame : frame_id;
    }
    binding->face_check_count = std::max(1, previous_count + 1);
    binding->face_bound_frame = first_frame;
  }

  static std::array<float, 4> face_bbox4_from_ptr(const float* p) {
    if (p == nullptr) {
      return std::array<float, 4>{{0.0f, 0.0f, 0.0f, 0.0f}};
    }
    return std::array<float, 4>{{p[0], p[1], p[2], p[3]}};
  }

  static float face_bbox_width(const std::array<float, 4>& b) {
    return std::max(0.0f, b[2] - b[0]);
  }

  static float face_bbox_height(const std::array<float, 4>& b) {
    return std::max(0.0f, b[3] - b[1]);
  }

  static bool face_finite_bbox(const std::array<float, 4>& b) {
    return std::isfinite(b[0]) && std::isfinite(b[1]) &&
           std::isfinite(b[2]) && std::isfinite(b[3]) &&
           face_bbox_width(b) > 0.0f && face_bbox_height(b) > 0.0f;
  }

  float face_center_distance_norm(const std::array<float, 4>& a,
                                  const std::array<float, 4>& b) const {
    const float ax = (a[0] + a[2]) * 0.5f;
    const float ay = (a[1] + a[3]) * 0.5f;
    const float bx = (b[0] + b[2]) * 0.5f;
    const float by = (b[1] + b[3]) * 0.5f;
    const float avg_h =
        std::max((face_bbox_height(a) + face_bbox_height(b)) * 0.5f, 1.0f);
    float dx = ax - bx;
    const float wrap_w = face_panorama_width();
    if (wrap_w > 1.0f) {
      dx = std::fmod(dx + wrap_w * 0.5f, wrap_w);
      if (dx < 0.0f) {
        dx += wrap_w;
      }
      dx -= wrap_w * 0.5f;
    }
    const float dy = ay - by;
    return std::sqrt(dx * dx + dy * dy) / avg_h;
  }

  float face_panorama_width() const {
    if (meta_.process_width > 0) {
      return static_cast<float>(meta_.process_width);
    }
    if (meta_.base_map_width > 0) {
      return static_cast<float>(meta_.base_map_width);
    }
    if (meta_.img_width > 0) {
      return static_cast<float>(meta_.img_width);
    }
    return 0.0f;
  }

  void update_display_face_memory(const FrameResult& result, int frame_id) {
    for (int i = 0; i < result.detection_count; ++i) {
      if (i >= static_cast<int>(result.track_ids.size()) ||
          i >= static_cast<int>(result.raw_track_ids.size()) ||
          i >= static_cast<int>(result.display_ids.size())) {
        continue;
      }
      const int public_id = result.track_ids[static_cast<size_t>(i)];
      const int binding_id = result.raw_track_ids[static_cast<size_t>(i)];
      const int display_id = result.display_ids[static_cast<size_t>(i)];
      if (public_id <= 0 || binding_id <= 0 || display_id <= 0) {
        continue;
      }
      const auto current_it = current_.find(binding_id);
      if (current_it == current_.end() || !current_it->second.has_face ||
          current_it->second.face_id.empty()) {
        continue;
      }
      const float* det =
          result.detections.data() + static_cast<size_t>(i) * kDetectionFields;
      DisplayFaceMemory& memory = display_face_memory_[display_id];
      memory.binding_id = binding_id;
      memory.public_id = public_id;
      memory.frame_id = frame_id;
      memory.face_id = current_it->second.face_id;
      memory.bbox = face_bbox4_from_ptr(det);
    }
  }

  void cleanup_display_face_memory(int frame_id) {
    for (auto it = display_face_memory_.begin(); it != display_face_memory_.end();) {
      if (frame_id - it->second.frame_id > cfg_.face_rec_binding_ttl) {
        it = display_face_memory_.erase(it);
      } else {
        ++it;
      }
    }
  }

  void cleanup_rejected_tentative_faces(const std::set<int>& active_public_ids) {
    for (auto it = rejected_tentative_faces_.begin();
         it != rejected_tentative_faces_.end();) {
      if (active_public_ids.find(it->first) == active_public_ids.end()) {
        it = rejected_tentative_faces_.erase(it);
      } else {
        ++it;
      }
    }
  }

  void cleanup_pending_dynamic_enroll(const std::set<int>& active_binding_ids,
                                      int frame_id) {
    for (auto it = pending_dynamic_enroll_.begin();
         it != pending_dynamic_enroll_.end();) {
      const bool inactive = active_binding_ids.find(it->first) == active_binding_ids.end();
      const bool expired =
          cfg_.face_rec_binding_ttl > 0 &&
          frame_id - it->second.last_frame > cfg_.face_rec_binding_ttl;
      if (inactive || expired || bindings_.find(it->first) != bindings_.end()) {
        it = pending_dynamic_enroll_.erase(it);
      } else {
        ++it;
      }
    }
  }

  int count_local_face_targets(const FrameResult& result,
                               int self_idx,
                               const std::array<float, 4>& bbox,
                               float center_thresh) const {
    if (!face_finite_bbox(bbox) || center_thresh <= 0.0f) {
      return 1;
    }
    int count = 0;
    for (int i = 0; i < result.detection_count; ++i) {
      if (face_binding_id_for_detection(result, i) <= 0) {
        continue;
      }
      const float* other_det =
          result.detections.data() + static_cast<size_t>(i) * kDetectionFields;
      const std::array<float, 4> other_bbox = face_bbox4_from_ptr(other_det);
      if (!face_finite_bbox(other_bbox)) {
        continue;
      }
      if (i == self_idx ||
          face_center_distance_norm(bbox, other_bbox) < center_thresh) {
        count += 1;
      }
    }
    return count;
  }

  bool maybe_start_tentative_face_from_display(
      const FrameResult& result,
      int idx,
      int public_id,
      int binding_id,
      const float* det,
      int frame_id,
      const std::set<int>& active_binding_ids,
      std::set<std::string>* used_this_frame,
      std::map<std::string, int>* reserved_face_owner) {
    if (!cfg_.face_rec_display_tentative || det == nullptr ||
        used_this_frame == nullptr || reserved_face_owner == nullptr ||
        public_id <= 0 || binding_id <= 0 || idx < 0) {
      return false;
    }
    const std::array<float, 4> bbox = face_bbox4_from_ptr(det);
    if (count_local_face_targets(
            result, idx, bbox, cfg_.display_id_reuse_center_thresh) > 1) {
      return false;
    }

    int old_binding_id = 0;
    int old_public_id = 0;
    std::string old_face_id;
    FaceRecognitionBinding old_binding;
    if (idx < static_cast<int>(result.display_ids.size()) &&
        result.display_ids[static_cast<size_t>(idx)] > 0) {
      const int display_id = result.display_ids[static_cast<size_t>(idx)];
      const auto memory_it = display_face_memory_.find(display_id);
      if (memory_it != display_face_memory_.end() &&
          memory_it->second.binding_id > 0 &&
          !memory_it->second.face_id.empty() &&
          memory_it->second.binding_id != binding_id &&
          frame_id - memory_it->second.frame_id <= cfg_.face_rec_binding_ttl) {
        if (face_finite_bbox(memory_it->second.bbox) &&
            face_center_distance_norm(bbox, memory_it->second.bbox) >=
                cfg_.display_id_reuse_center_thresh) {
          return false;
        }
        old_binding_id = memory_it->second.binding_id;
        old_public_id = memory_it->second.public_id;
        old_face_id = memory_it->second.face_id;
        const auto old_bind = bindings_.find(old_binding_id);
        if (old_bind != bindings_.end() && old_bind->second.has_face &&
            old_bind->second.face_id == old_face_id && !old_bind->second.tentative) {
          old_binding = old_bind->second;
        } else {
          old_binding.has_face = true;
          old_binding.face_id = old_face_id;
          old_binding.dynamic = is_dynamic_identity(old_face_id);
        }
      }
    }
    if (old_binding_id <= 0 || old_binding_id == binding_id || old_face_id.empty() ||
        active_binding_ids.find(old_binding_id) != active_binding_ids.end() ||
        rejected_tentative_faces_.find(std::make_pair(binding_id, old_face_id)) !=
            rejected_tentative_faces_.end()) {
      return false;
    }
    const auto reserved_it = reserved_face_owner->find(old_face_id);
    if (reserved_it != reserved_face_owner->end() &&
        reserved_it->second != binding_id) {
      return false;
    }
    if (used_this_frame->find(old_face_id) != used_this_frame->end()) {
      return false;
    }

    FaceRecognitionBinding tentative = old_binding;
    tentative.score = 0.0f;
    tentative.second_score = 0.0f;
    tentative.quality_passed = false;
    tentative.updated = false;
    tentative.tentative = true;
    tentative.tentative_fail_count = 0;
    tentative.event = "tentative_face_carry";
    tentative.has_debug = true;
    tentative.debug_reason = tentative.event;
    tentative.debug_rknn_ran = false;
    tentative.debug_best_name = old_face_id;
    tentative.debug_best_score = 0.0f;
    tentative.debug_second_score = 0.0f;
    tentative.debug_strong_identity_match = false;
    tentative.debug_best_reserved_by_other = false;
    tentative.debug_used_by_other_this_frame = false;
    tentative.debug_blocked_by_owner = false;
    tentative.debug_reserved_owner = old_binding_id;
    tentative.debug_has_nearest_feature = false;
    tentative.debug_nearest_public_id = 0;
    tentative.debug_nearest_feature_score = 0.0f;

    bindings_[binding_id] = tentative;
    current_[binding_id] = tentative;
    (*reserved_face_owner)[old_face_id] = binding_id;
    used_this_frame->insert(old_face_id);
    pending_face_relink_.erase(binding_id);
    last_attempt_frame_.erase(binding_id);
    last_active_frame_[binding_id] = frame_id;
    return true;
  }

  void apply_infer_result(
      const FaceRecInferTask& task,
      int frame_id,
      std::set<std::string>* used_this_frame,
      std::map<std::string, int>* reserved_face_owner,
      std::map<int, std::vector<float>>* frame_features) {
    if (used_this_frame == nullptr || reserved_face_owner == nullptr ||
        frame_features == nullptr) {
      return;
    }
    const int binding_id = task.binding_id;
    const int public_id = task.public_id;
    if (binding_id <= 0) {
      return;
    }
    const float* det = task.det.data();
    auto bind_it = bindings_.find(binding_id);
    auto face_reserved_by_other =
        [&](const std::string& face_id, int owner_binding_id) -> bool {
      const auto owner_it = reserved_face_owner->find(face_id);
      return owner_it != reserved_face_owner->end() && owner_it->second != owner_binding_id;
    };
    auto reserved_owner_for_face = [&](const std::string& face_id) -> int {
      const auto owner_it = reserved_face_owner->find(face_id);
      return owner_it != reserved_face_owner->end() ? owner_it->second : 0;
    };

    if (task.ret <= 0) {
      std::cerr << "[FaceRecCpp] infer failed for raw_track_id=" << binding_id
                << " public_id=" << public_id
                << ": " << task.error << "\n";
      FaceRecognitionBinding debug_binding;
      if (bind_it != bindings_.end()) {
        debug_binding = bind_it->second;
        debug_binding.event =
            debug_binding.tentative ? "tentative_face_carry" : "carry";
      }
      debug_binding.has_debug = true;
      debug_binding.debug_reason = "infer_failed";
      debug_binding.debug_rknn_ran = true;
      debug_binding.quality_passed = true;
      current_[binding_id] = debug_binding;
      return;
    }

    std::vector<float> feature = task.feature;
    normalize_feature(feature);
    const FaceFeatureQuality feature_quality = feature_quality_for_det(det);
    (*frame_features)[binding_id] = feature;

    FaceRecognitionBinding assigned;
    assigned.quality_passed = true;
    if (bind_it != bindings_.end() && bind_it->second.tentative) {
      assigned = bind_it->second;
      assigned.quality_passed = true;
      assigned.updated = false;
      const std::string current_face_id = assigned.face_id;
      assigned.score = identity_score(current_face_id, feature);
      assigned.second_score = 0.0f;
      assigned.has_debug = true;
      assigned.debug_rknn_ran = true;
      assigned.debug_best_name = current_face_id;
      assigned.debug_best_score = assigned.score;
      assigned.debug_second_score = 0.0f;
      assigned.debug_best_reserved_by_other =
          !current_face_id.empty() && face_reserved_by_other(current_face_id, binding_id);
      assigned.debug_used_by_other_this_frame = false;
      assigned.debug_reserved_owner =
          current_face_id.empty() ? 0 : reserved_owner_for_face(current_face_id);
      assigned.debug_blocked_by_owner = assigned.debug_best_reserved_by_other;
      if (!current_face_id.empty() &&
          !assigned.debug_best_reserved_by_other &&
          assigned.score >= cfg_.face_rec_tentative_threshold) {
        assigned.tentative = false;
        assigned.tentative_fail_count = 0;
        assigned.debug_strong_identity_match = true;
        const FaceUpdateDecision update =
            maybe_update_identity(current_face_id, feature, feature_quality, false, frame_id);
        assigned.updated = update.updated();
        assigned.event = assigned.updated
                             ? std::string("tentative_confirm_update_") + update.kind
                             : "tentative_confirm";
        assigned.debug_reason = assigned.event;
        apply_update_debug(update, &assigned);
        mark_face_check_success(binding_id, frame_id, &assigned);
        bindings_[binding_id] = assigned;
        current_[binding_id] = assigned;
        (*reserved_face_owner)[assigned.face_id] = binding_id;
        used_this_frame->insert(assigned.face_id);
        rejected_tentative_faces_.erase(std::make_pair(binding_id, assigned.face_id));
        pending_face_relink_.erase(binding_id);
        return;
      }

      assigned.tentative_fail_count += 1;
      assigned.debug_strong_identity_match = false;
      if (assigned.tentative_fail_count < cfg_.face_rec_tentative_fail_frames &&
          !current_face_id.empty() &&
          !assigned.debug_best_reserved_by_other) {
        assigned.event = "tentative_reject_pending";
        assigned.debug_reason = assigned.event;
        bindings_[binding_id] = assigned;
        current_[binding_id] = assigned;
        (*reserved_face_owner)[assigned.face_id] = binding_id;
        used_this_frame->insert(assigned.face_id);
        pending_face_relink_.erase(binding_id);
        return;
      }

      pending_face_relink_.erase(binding_id);
      if (!current_face_id.empty()) {
        rejected_tentative_faces_.insert(std::make_pair(binding_id, current_face_id));
        used_this_frame->erase(current_face_id);
        const auto owner_it = reserved_face_owner->find(current_face_id);
        if (owner_it != reserved_face_owner->end() && owner_it->second == binding_id) {
          reserved_face_owner->erase(owner_it);
        }
      }
      bindings_.erase(bind_it);
      bind_it = bindings_.end();
      assigned = FaceRecognitionBinding();
      assigned.quality_passed = true;
    }
    if (bind_it != bindings_.end()) {
      pending_dynamic_enroll_.erase(binding_id);
      assigned = bind_it->second;
      assigned.quality_passed = true;
      const std::string current_face_id = assigned.face_id;
      assigned.score = identity_score(current_face_id, feature);

      std::string relink_name;
      float relink_score = 0.0f;
      float relink_second_score = 0.0f;
      const bool relink_candidate = find_relink_candidate(
          current_face_id, feature, feature_quality,
          &relink_name, &relink_score, &relink_second_score);
      const bool relink_reserved_by_other =
          relink_candidate && face_reserved_by_other(relink_name, binding_id);
      const bool relink_used_this_frame =
          relink_candidate && used_this_frame->find(relink_name) != used_this_frame->end();
      if (relink_candidate && !relink_reserved_by_other && !relink_used_this_frame) {
        PendingFaceRelink& pending = pending_face_relink_[binding_id];
        if (pending.face_id == relink_name) {
          pending.count += 1;
        } else {
          pending.face_id = relink_name;
          pending.count = 1;
        }
        pending.last_frame = frame_id;
        if (pending.count >= cfg_.face_rec_relink_confirm_frames) {
          merge_dynamic_identity_into(current_face_id, relink_name);
          assigned.has_face = true;
          assigned.face_id = relink_name;
          assigned.score = relink_score;
          assigned.second_score = relink_second_score;
          assigned.dynamic = is_dynamic_identity(relink_name);
          assigned.lock_fail_count = 0;
          const FaceUpdateDecision update =
              maybe_update_identity(relink_name, feature, feature_quality, false, frame_id);
          assigned.updated = update.updated();
          assigned.event = assigned.updated
                               ? std::string("relink_update_") + update.kind
                               : "relink";
          assigned.has_debug = true;
          assigned.debug_reason = assigned.event;
          assigned.debug_rknn_ran = true;
          assigned.debug_best_name = relink_name;
          assigned.debug_best_score = relink_score;
          assigned.debug_second_score = relink_second_score;
          assigned.debug_strong_identity_match = true;
          assigned.debug_best_reserved_by_other = false;
          assigned.debug_used_by_other_this_frame = false;
          assigned.debug_blocked_by_owner = false;
          assigned.debug_reserved_owner = binding_id;
          apply_update_debug(update, &assigned);
          mark_face_check_success(binding_id, frame_id, &assigned);
          bindings_[binding_id] = assigned;
          current_[binding_id] = assigned;
          (*reserved_face_owner)[assigned.face_id] = binding_id;
          used_this_frame->insert(assigned.face_id);
          pending_face_relink_.erase(binding_id);
          return;
        }

        assigned.updated = false;
        assigned.second_score = relink_second_score;
        assigned.event = "relink_pending";
        assigned.has_debug = true;
        assigned.debug_reason = assigned.event;
        assigned.debug_rknn_ran = true;
        assigned.debug_best_name = relink_name;
        assigned.debug_best_score = relink_score;
        assigned.debug_second_score = relink_second_score;
        assigned.debug_strong_identity_match = true;
        assigned.debug_best_reserved_by_other = false;
        assigned.debug_used_by_other_this_frame = false;
        assigned.debug_blocked_by_owner = false;
        assigned.debug_reserved_owner = binding_id;
        bindings_[binding_id] = assigned;
        current_[binding_id] = assigned;
        used_this_frame->insert(current_face_id);
        return;
      }

      pending_face_relink_.erase(binding_id);
      const bool lock_verify_failed =
          cfg_.face_rec_lock_verify && feature_quality.sample_ok &&
          !current_face_id.empty() &&
          assigned.score < cfg_.face_rec_lock_min_similarity &&
          !relink_candidate;
      if (lock_verify_failed) {
        const bool can_release_locked_face =
            binding_reappeared_after_gap_.find(binding_id) != binding_reappeared_after_gap_.end();
        if (!can_release_locked_face) {
          assigned.lock_fail_count = 0;
          assigned.updated = false;
          assigned.second_score = 0.0f;
          assigned.event = "lock_low_keep";
          assigned.has_debug = true;
          assigned.debug_reason = assigned.event;
          assigned.debug_rknn_ran = true;
          assigned.debug_best_name = assigned.face_id;
          assigned.debug_best_score = assigned.score;
          assigned.debug_second_score = 0.0f;
          assigned.debug_strong_identity_match = false;
          assigned.debug_best_reserved_by_other = relink_reserved_by_other;
          assigned.debug_used_by_other_this_frame = relink_used_this_frame;
          assigned.debug_blocked_by_owner = false;
          assigned.debug_reserved_owner = binding_id;
          bindings_[binding_id] = assigned;
          current_[binding_id] = assigned;
          (*reserved_face_owner)[assigned.face_id] = binding_id;
          used_this_frame->insert(assigned.face_id);
          return;
        }
        assigned.lock_fail_count += 1;
        assigned.updated = false;
        assigned.second_score = 0.0f;
        assigned.event = assigned.lock_fail_count >= cfg_.face_rec_lock_fail_frames
                             ? "lock_release"
                             : "lock_release_pending";
        assigned.has_debug = true;
        assigned.debug_reason = assigned.event;
        assigned.debug_rknn_ran = true;
        assigned.debug_best_name = assigned.face_id;
        assigned.debug_best_score = assigned.score;
        assigned.debug_second_score = 0.0f;
        assigned.debug_strong_identity_match = false;
        assigned.debug_best_reserved_by_other = relink_reserved_by_other;
        assigned.debug_used_by_other_this_frame = relink_used_this_frame;
        assigned.debug_blocked_by_owner = false;
        assigned.debug_reserved_owner = binding_id;
        if (assigned.lock_fail_count < cfg_.face_rec_lock_fail_frames) {
          bindings_[binding_id] = assigned;
          current_[binding_id] = assigned;
          (*reserved_face_owner)[assigned.face_id] = binding_id;
          used_this_frame->insert(assigned.face_id);
          return;
        }
        used_this_frame->erase(current_face_id);
        const auto owner_it = reserved_face_owner->find(current_face_id);
        if (owner_it != reserved_face_owner->end() && owner_it->second == binding_id) {
          reserved_face_owner->erase(owner_it);
        }
        bindings_.erase(bind_it);
        bind_it = bindings_.end();
        assigned = FaceRecognitionBinding();
        assigned.quality_passed = true;
      }
      if (bind_it != bindings_.end()) {
        if (assigned.score >= cfg_.face_rec_lock_min_similarity) {
          assigned.lock_fail_count = 0;
        }
        const FaceUpdateDecision update =
            maybe_update_identity(current_face_id, feature, feature_quality, true, frame_id);
        assigned.updated = update.updated();
        assigned.second_score = 0.0f;
        assigned.event = assigned.updated
                             ? std::string("lock_update_") + update.kind
                             : "lock_keep";
        assigned.has_debug = true;
        assigned.debug_reason = assigned.event;
        assigned.debug_rknn_ran = true;
        assigned.debug_best_name = assigned.face_id;
        assigned.debug_best_score = assigned.score;
        assigned.debug_second_score = 0.0f;
        assigned.debug_strong_identity_match =
            assigned.score >= cfg_.face_rec_lock_min_similarity;
        assigned.debug_best_reserved_by_other = relink_reserved_by_other;
        assigned.debug_used_by_other_this_frame = relink_used_this_frame;
        assigned.debug_blocked_by_owner =
            relink_candidate && (relink_reserved_by_other || relink_used_this_frame);
        assigned.debug_reserved_owner =
            relink_candidate ? reserved_owner_for_face(relink_name) : binding_id;
        apply_update_debug(update, &assigned);
        mark_face_check_success(binding_id, frame_id, &assigned);
        bindings_[binding_id] = assigned;
        current_[binding_id] = assigned;
        used_this_frame->insert(assigned.face_id);
        return;
      }
    }

    std::string best_name;
    float best_score = 0.0f;
    float second_score = 0.0f;
    match_identity(feature, &best_name, &best_score, &second_score);
    const bool best_reserved_by_other =
        !best_name.empty() && face_reserved_by_other(best_name, binding_id);
    const bool used_by_other_this_frame =
        !best_name.empty() && used_this_frame->find(best_name) != used_this_frame->end();
    const bool strong_identity_match =
        !best_name.empty() &&
        best_score >= cfg_.face_rec_threshold &&
        best_score >= second_score + cfg_.face_rec_margin;
    const bool strong_match_blocked_by_owner =
        strong_identity_match &&
        (used_by_other_this_frame || best_reserved_by_other);
    FaceRecognitionBinding debug_binding;
    debug_binding.has_debug = true;
    debug_binding.debug_reason = "unassigned";
    debug_binding.debug_rknn_ran = true;
    debug_binding.quality_passed = true;
    debug_binding.debug_best_name = best_name;
    debug_binding.debug_best_score = best_score;
    debug_binding.debug_second_score = second_score;
    debug_binding.debug_strong_identity_match = strong_identity_match;
    debug_binding.debug_best_reserved_by_other = best_reserved_by_other;
    debug_binding.debug_used_by_other_this_frame = used_by_other_this_frame;
    debug_binding.debug_blocked_by_owner = strong_match_blocked_by_owner;
    debug_binding.debug_reserved_owner =
        best_name.empty() ? 0 : reserved_owner_for_face(best_name);
    if (!best_name.empty() &&
        strong_identity_match &&
        !used_by_other_this_frame &&
        !best_reserved_by_other) {
      pending_dynamic_enroll_.erase(binding_id);
      assigned.has_face = true;
      assigned.face_id = best_name;
      assigned.score = best_score;
      assigned.second_score = second_score;
      assigned.dynamic = is_dynamic_identity(best_name);
      const FaceUpdateDecision update =
          maybe_update_identity(best_name, feature, feature_quality, false, frame_id);
      assigned.updated = update.updated();
      assigned.event = assigned.updated
                           ? std::string("match_update_") + update.kind
                           : "match";
      assigned.has_debug = true;
      assigned.debug_reason = assigned.event;
      assigned.debug_rknn_ran = true;
      assigned.debug_best_name = best_name;
      assigned.debug_best_score = best_score;
      assigned.debug_second_score = second_score;
      assigned.debug_strong_identity_match = strong_identity_match;
      assigned.debug_best_reserved_by_other = best_reserved_by_other;
      assigned.debug_used_by_other_this_frame = used_by_other_this_frame;
      assigned.debug_blocked_by_owner = false;
      assigned.debug_reserved_owner = binding_id;
      apply_update_debug(update, &assigned);
    } else if (cfg_.face_rec_dynamic_library &&
               !strong_match_blocked_by_owner &&
               feature_quality.sample_ok) {
      int pending_hits = 0;
      bool pending_reset = false;
      const std::string enrolled_name = update_pending_dynamic_enroll(
          binding_id, feature, feature_quality, frame_id, &pending_hits, &pending_reset);
      if (!enrolled_name.empty()) {
        assigned.has_face = true;
        assigned.face_id = enrolled_name;
        assigned.score = best_score;
        assigned.second_score = second_score;
        assigned.dynamic = true;
        assigned.updated = true;
        assigned.event = "dynamic_enroll_confirm";
        assigned.has_debug = true;
        assigned.debug_reason = assigned.event;
        assigned.debug_rknn_ran = true;
        assigned.debug_best_name = best_name;
        assigned.debug_best_score = best_score;
        assigned.debug_second_score = second_score;
        assigned.debug_strong_identity_match = strong_identity_match;
        assigned.debug_best_reserved_by_other = best_reserved_by_other;
        assigned.debug_used_by_other_this_frame = used_by_other_this_frame;
        assigned.debug_blocked_by_owner = false;
        assigned.debug_reserved_owner = binding_id;
      } else {
        debug_binding.debug_reason =
            pending_reset ? "dynamic_enroll_pending_reset" : "dynamic_enroll_pending";
        debug_binding.event = debug_binding.debug_reason;
        debug_binding.score = best_score;
        debug_binding.second_score = second_score;
        debug_binding.debug_reserved_owner = binding_id;
        (void)pending_hits;
      }
    } else if (cfg_.face_rec_dynamic_library &&
               !strong_match_blocked_by_owner &&
               !feature_quality.sample_ok) {
      pending_dynamic_enroll_.erase(binding_id);
      debug_binding.debug_reason = "enroll_quality_gate";
    } else if (strong_match_blocked_by_owner) {
      pending_dynamic_enroll_.erase(binding_id);
      debug_binding.debug_reason = "strong_match_blocked_by_owner";
    } else if (!cfg_.face_rec_dynamic_library) {
      pending_dynamic_enroll_.erase(binding_id);
      debug_binding.debug_reason = "dynamic_library_disabled";
    } else {
      pending_dynamic_enroll_.erase(binding_id);
      debug_binding.debug_reason = "unassigned";
    }

    if (assigned.has_face) {
      mark_face_check_success(binding_id, frame_id, &assigned);
      bindings_[binding_id] = assigned;
      current_[binding_id] = assigned;
      (*reserved_face_owner)[assigned.face_id] = binding_id;
      used_this_frame->insert(assigned.face_id);
    } else if (bind_it == bindings_.end()) {
      current_[binding_id] = debug_binding;
    }
  }

  void annotate_targets(std::vector<TargetInfo>* targets) const {
    if (targets == nullptr) {
      return;
    }
    for (TargetInfo& target : *targets) {
      const int binding_id = target.raw_track_id > 0 ? target.raw_track_id : -1;
      if (binding_id <= 0) {
        continue;
      }
      const auto it = current_.find(binding_id);
      if (it == current_.end()) {
        continue;
      }
      const FaceRecognitionBinding& binding = it->second;
      if (binding.has_face) {
        target.has_face_recognition = true;
        target.face_id = binding.face_id;
        target.face_recognition_score = binding.score;
        target.face_recognition_second_score = binding.second_score;
        target.face_recognition_dynamic = binding.dynamic;
        target.face_recognition_quality = binding.quality_passed;
        target.face_recognition_updated = binding.updated;
        target.face_recognition_tentative = binding.tentative;
        target.face_recognition_event = binding.event;
      }
      if (binding.has_debug) {
        target.has_face_recognition_debug = true;
        target.face_recognition_debug_reason = binding.debug_reason;
        target.face_recognition_debug_best_name = binding.debug_best_name;
        target.face_recognition_debug_best_score = binding.debug_best_score;
        target.face_recognition_debug_second_score = binding.debug_second_score;
        target.face_recognition_debug_rknn_ran = binding.debug_rknn_ran;
        target.face_recognition_debug_strong_identity_match =
            binding.debug_strong_identity_match;
        target.face_recognition_debug_best_reserved_by_other =
            binding.debug_best_reserved_by_other;
        target.face_recognition_debug_used_by_other_this_frame =
            binding.debug_used_by_other_this_frame;
        target.face_recognition_debug_blocked_by_owner =
            binding.debug_blocked_by_owner;
        target.face_recognition_debug_reserved_owner =
            binding.debug_reserved_owner;
        target.face_recognition_debug_has_nearest_feature =
            binding.debug_has_nearest_feature;
        target.face_recognition_debug_nearest_public_id =
            binding.debug_nearest_public_id;
        target.face_recognition_debug_nearest_feature_score =
            binding.debug_nearest_feature_score;
        target.face_recognition_debug_pose_sample_quality_ok =
            binding.debug_pose_sample_quality_ok;
        target.face_recognition_debug_locked_pose_quality_ok =
            binding.debug_locked_pose_quality_ok;
        target.face_recognition_debug_has_pose_library_similarity =
            binding.debug_has_pose_library_similarity;
        target.face_recognition_debug_pose_sample_missing_frames =
            binding.debug_pose_sample_missing_frames;
        target.face_recognition_debug_update_reason =
            binding.debug_update_reason;
        target.face_recognition_debug_identity_score =
            binding.debug_identity_score;
        target.face_recognition_debug_sample_similarity =
            binding.debug_sample_similarity;
        target.face_recognition_debug_pose_library_similarity =
            binding.debug_pose_library_similarity;
        target.face_recognition_debug_pose_library_max_similarity =
            binding.debug_pose_library_max_similarity;
      }
    }
  }

 private:
  struct PendingFaceRelink {
    std::string face_id;
    int count = 0;
    int last_frame = 0;
  };

  struct PendingDynamicEnroll {
    std::vector<std::vector<float>> features;
    std::vector<FaceFeatureQuality> qualities;
    std::vector<int> frames;
    int last_frame = 0;
  };

  struct AdaFaceHandle {
    ~AdaFaceHandle() {
      if (handle != nullptr) {
        adaface_rknn_destroy(handle);
      }
    }
    void** out() { return &handle; }
    void* get() const { return handle; }
    void* handle = nullptr;
  };

  void load_feature_library(const std::string& dir) {
    if (dir.empty()) {
      return;
    }
    for (const std::string& path : list_dir_paths(dir)) {
      if (path.size() < 4 || path.substr(path.size() - 4) != ".npy") {
        continue;
      }
      if (basename_without_ext(path) == "_preview") {
        continue;
      }
      std::vector<std::vector<float>> rows;
      std::string error;
      if (!parse_npy_float32(path, &rows, &error) || rows.empty()) {
        std::cerr << "[FaceRecCpp] skip feature " << path << ": " << error << "\n";
        continue;
      }
      FaceIdentity id;
      id.name = basename_without_ext(path);
      id.dynamic = false;
      for (std::vector<float>& feat : rows) {
        if (static_cast<int>(feat.size()) == feature_dim_) {
          id.features.push_back(std::move(feat));
        }
      }
      if (!id.features.empty()) {
        update_next_dynamic_id(id.name);
        identities_.push_back(std::move(id));
      }
    }
  }

  static void normalize_feature(std::vector<float>& feature) {
    double sum = 0.0;
    for (float v : feature) {
      sum += static_cast<double>(v) * v;
    }
    const double norm = std::sqrt(sum);
    if (norm <= 1.0e-12) {
      return;
    }
    const float inv = static_cast<float>(1.0 / norm);
    for (float& v : feature) {
      v *= inv;
    }
  }

	  bool quality_pass(const float* det, std::string* reason) const {
	    const float w = det[2] - det[0];
	    const float h = det[3] - det[1];
    if (!std::isfinite(w) || !std::isfinite(h) ||
        std::min(w, h) < cfg_.face_rec_min_box_size) {
      if (reason != nullptr) {
        *reason = "small_bbox";
      }
      return false;
    }
    const float* left_eye = det + 5 + 0 * 3;
    const float* right_eye = det + 5 + 1 * 3;
    const float* nose = det + 5 + 2 * 3;
    if (!valid_keypoint(left_eye) || !valid_keypoint(right_eye) || !valid_keypoint(nose)) {
      if (reason != nullptr) {
        *reason = "invalid_keypoints";
      }
      return false;
    }
    if (std::min(left_eye[2], std::min(right_eye[2], nose[2])) < cfg_.face_rec_min_keypoint_conf) {
      if (reason != nullptr) {
        *reason = "low_keypoint_conf";
      }
      return false;
    }
    const float dx = right_eye[0] - left_eye[0];
    const float dy = right_eye[1] - left_eye[1];
    const float eye_dist = std::sqrt(dx * dx + dy * dy);
    if (cfg_.face_rec_min_eye_dist > 0.0f && eye_dist < cfg_.face_rec_min_eye_dist) {
      if (reason != nullptr) {
        *reason = "small_eye_dist";
      }
      return false;
	    }
	    return true;
	  }

  bool estimate_face_yaw_deg(const float* det, float* yaw_deg) const {
    if (det == nullptr || yaw_deg == nullptr) {
      return false;
    }
    const float* left_eye = det + 5 + 0 * 3;
    const float* right_eye = det + 5 + 1 * 3;
    const float* nose = det + 5 + 2 * 3;
    if (!valid_keypoint(left_eye) || !valid_keypoint(right_eye) ||
        !valid_keypoint(nose)) {
      return false;
    }
    const double eye_dx = static_cast<double>(right_eye[0] - left_eye[0]);
    if (std::abs(eye_dx) < 1.0e-6) {
      return false;
    }
    const double eye_mid_x =
        (static_cast<double>(left_eye[0]) + static_cast<double>(right_eye[0])) * 0.5;
    const double nose_lateral = static_cast<double>(nose[0]) - eye_mid_x;
    const double nose_norm = nose_lateral / (std::abs(eye_dx) * 0.5);
    const double tan_yaw = std::abs(nose_norm) / 0.6;
    *yaw_deg = static_cast<float>(std::atan(tan_yaw) * 180.0 / kPi);
    return std::isfinite(*yaw_deg);
  }

	  FaceFeatureQuality feature_quality_for_det(const float* det) const {
	    FaceFeatureQuality quality;
	    const float w = det[2] - det[0];
	    const float h = det[3] - det[1];
	    quality.min_side = std::isfinite(w) && std::isfinite(h) ? std::min(w, h) : 0.0f;
	    quality.det_score = det[4];
	    const float* left_eye = det + 5 + 0 * 3;
	    const float* right_eye = det + 5 + 1 * 3;
	    const float* nose = det + 5 + 2 * 3;
	    if (valid_keypoint(left_eye) && valid_keypoint(right_eye) && valid_keypoint(nose)) {
	      quality.min_keypoint_conf =
	          std::min(left_eye[2], std::min(right_eye[2], nose[2]));
	    }
    quality.yaw_valid = estimate_face_yaw_deg(det, &quality.yaw_deg);
	    quality.sample_ok =
	        quality.min_side >= cfg_.face_rec_update_min_box_size &&
	        quality.min_keypoint_conf >= cfg_.face_rec_update_min_keypoint_conf &&
	        quality.det_score >= cfg_.face_rec_update_min_score;
	    quality.primary_ok =
	        quality.sample_ok &&
	        quality.min_side >= cfg_.face_rec_primary_min_box_size &&
	        quality.min_keypoint_conf >= cfg_.face_rec_primary_min_keypoint_conf &&
	        quality.det_score >= cfg_.face_rec_primary_min_score &&
          quality.yaw_valid &&
          quality.yaw_deg <= cfg_.face_rec_primary_max_yaw_deg;
	    const float side_den = std::max(1.0f, cfg_.face_rec_primary_min_box_size);
	    const float side_score = std::min(1.0f, quality.min_side / side_den);
	    quality.quality_score =
	        0.45f * side_score +
	        0.35f * quality.min_keypoint_conf +
	        0.20f * quality.det_score;
	    return quality;
	  }

  void save_debug_crops(const Image& source,
                        int frame_index,
                        int public_id,
                        int display_id,
                        const float* det,
                        const std::vector<float>& selected_input) {
    if (!cfg_.face_rec_debug_crops) {
      return;
    }
    std::vector<float> eye_input;
    if (extract_eye_face_tensor(source, det, &eye_input)) {
      save_debug_crop_tensor(frame_index, public_id, display_id, det, "eye", eye_input);
    }
    std::vector<float> five_point_input;
    if (extract_five_point_face_tensor(source, det, &five_point_input)) {
      save_debug_crop_tensor(
          frame_index, public_id, display_id, det, "five_point", five_point_input);
    }
    if (eye_input.empty() && five_point_input.empty()) {
      save_debug_crop_tensor(
          frame_index, public_id, display_id, det, "selected", selected_input);
    }
  }

  void save_debug_crop_tensor(int frame_index,
                              int public_id,
                              int display_id,
                              const float* det,
                              const std::string& mode,
                              const std::vector<float>& input) {
    if (!cfg_.face_rec_debug_crops || input_h_ <= 0 || input_w_ <= 0 ||
        input_c_ != 3 ||
        input.size() != static_cast<size_t>(input_c_) * input_h_ * input_w_) {
      return;
    }
    try {
      if (!ensure_dir_exists(cfg_.face_rec_debug_crop_dir)) {
        if (debug_crop_errors_ < 3) {
          std::cerr << "[FaceRecCpp] cannot create debug crop dir: "
                    << cfg_.face_rec_debug_crop_dir << "\n";
        }
        debug_crop_errors_ += 1;
        return;
      }
      if (!debug_crop_encoder_) {
        debug_crop_encoder_.reset(new TurboJpegEncoder());
      }
      Image crop;
      crop.width = input_w_;
      crop.height = input_h_;
      crop.bgr.assign(static_cast<size_t>(crop.width) * crop.height * 3, 0);
      for (int y = 0; y < input_h_; ++y) {
        for (int x = 0; x < input_w_; ++x) {
          const size_t out_idx = (static_cast<size_t>(y) * input_w_ + x) * 3;
          for (int c = 0; c < input_c_; ++c) {
            const size_t in_idx =
                (static_cast<size_t>(c) * input_h_ + y) * input_w_ + x;
            const float v = std::max(-1.0f, std::min(1.0f, input[in_idx]));
            crop.bgr[out_idx + static_cast<size_t>(c)] =
                static_cast<uint8_t>(std::max(0, std::min(255,
                    static_cast<int>(std::lround((v * 0.5f + 0.5f) * 255.0f)))));
          }
        }
      }
      std::ostringstream name;
      name << "frame_" << std::setw(6) << std::setfill('0') << frame_index
           << std::setfill(' ')
           << "_" << mode
           << "_public_" << public_id
           << "_display_" << display_id
           << "_score_" << std::fixed << std::setprecision(3)
           << (det != nullptr ? det[4] : 0.0f)
           << "_size_" << std::setprecision(0)
           << (det != nullptr ? std::max(0.0f, det[2] - det[0]) : 0.0f)
           << "x"
           << (det != nullptr ? std::max(0.0f, det[3] - det[1]) : 0.0f)
           << ".jpg";
      const std::string path = join_path(cfg_.face_rec_debug_crop_dir, name.str());
      const std::vector<uint8_t> jpeg = debug_crop_encoder_->encode_bgr(crop, 95);
      std::ofstream out(path.c_str(), std::ios::binary | std::ios::out | std::ios::trunc);
      if (!out) {
        throw std::runtime_error("cannot open " + path);
      }
      out.write(reinterpret_cast<const char*>(jpeg.data()),
                static_cast<std::streamsize>(jpeg.size()));
    } catch (const std::exception& e) {
      if (debug_crop_errors_ < 3) {
        std::cerr << "[FaceRecCpp] save debug crop failed: " << e.what() << "\n";
      }
      debug_crop_errors_ += 1;
    }
	  }

  bool extract_aligned_face_tensor(const Image& source,
                                   const float* det,
                                   std::vector<float>* out_nchw) const {
    if (cfg_.face_rec_align_mode == "five_point") {
      return extract_five_point_face_tensor(source, det, out_nchw);
    }
    return extract_eye_face_tensor(source, det, out_nchw);
  }

  bool face_sampling_ready(const Image& source, std::vector<float>* out_nchw) const {
    if (out_nchw == nullptr || source.width <= 0 || source.height <= 0 ||
        source.bgr.size() != static_cast<size_t>(source.width) * source.height * 3 ||
        base_map_x_ == nullptr || base_map_y_ == nullptr ||
        meta_.base_map_width <= 0 || meta_.base_map_height <= 0 ||
        base_map_x_->size() != static_cast<size_t>(meta_.base_map_width) * meta_.base_map_height ||
        base_map_y_->size() != static_cast<size_t>(meta_.base_map_width) * meta_.base_map_height) {
      return false;
    }
    return true;
  }

  bool extract_eye_face_tensor(const Image& source,
                               const float* det,
                               std::vector<float>* out_nchw) const {
    if (!face_sampling_ready(source, out_nchw)) {
      return false;
    }

    const float* left_eye = det + 5 + 0 * 3;
    const float* right_eye = det + 5 + 1 * 3;
    const float dx = right_eye[0] - left_eye[0];
    const float dy = right_eye[1] - left_eye[1];
    const float eye_dist = std::sqrt(dx * dx + dy * dy);
    if (eye_dist <= 0.0f) {
      return false;
    }
    const float eye_mid_x = (left_eye[0] + right_eye[0]) * 0.5f;
    const float eye_mid_y = (left_eye[1] + right_eye[1]) * 0.5f;
    const float angle = std::atan2(dy, dx);
    const float cos_a = std::cos(angle);
    const float sin_a = std::sin(angle);
    const float side = std::max(32.0f, eye_dist * 3.2f);
    const float crop_x1 = eye_mid_x - side * 0.5f;
    const float crop_y1 = eye_mid_y + 0.55f * eye_dist - side * 0.45f;

    out_nchw->assign(static_cast<size_t>(input_c_) * input_h_ * input_w_, 0.0f);
    for (int oy = 0; oy < input_h_; ++oy) {
      for (int ox = 0; ox < input_w_; ++ox) {
        const float rx = crop_x1 + (static_cast<float>(ox) + 0.5f) * side / input_w_;
        const float ry = crop_y1 + (static_cast<float>(oy) + 0.5f) * side / input_h_;
        const float ddx = rx - eye_mid_x;
        const float ddy = ry - eye_mid_y;
        const float px = eye_mid_x + cos_a * ddx - sin_a * ddy;
        const float py = eye_mid_y + sin_a * ddx + cos_a * ddy;
        float bgr[3] = {0.0f, 0.0f, 0.0f};
        sample_panorama_bgr(source, px, py, bgr);
        for (int c = 0; c < input_c_; ++c) {
          const float normalized = (bgr[c] / 255.0f - 0.5f) / 0.5f;
          (*out_nchw)[(static_cast<size_t>(c) * input_h_ + oy) * input_w_ + ox] = normalized;
        }
      }
    }
    return true;
  }

  bool extract_five_point_face_tensor(const Image& source,
                                      const float* det,
                                      std::vector<float>* out_nchw) const {
    if (!face_sampling_ready(source, out_nchw)) {
      return false;
    }
    static const float kArcFaceTemplate[5][2] = {
        {38.2946f, 51.6963f},
        {73.5318f, 51.5014f},
        {56.0252f, 71.7366f},
        {41.5493f, 92.3655f},
        {70.7299f, 92.2041f},
    };
    std::array<std::array<float, 2>, 5> src{};
    for (int i = 0; i < 5; ++i) {
      const float* kp = det + 5 + i * 3;
      if (!valid_keypoint(kp)) {
        return false;
      }
      src[static_cast<size_t>(i)][0] = kp[0];
      src[static_cast<size_t>(i)][1] = kp[1];
    }

    float mean_src_x = 0.0f;
    float mean_src_y = 0.0f;
    float mean_dst_x = 0.0f;
    float mean_dst_y = 0.0f;
    for (int i = 0; i < 5; ++i) {
      mean_src_x += src[static_cast<size_t>(i)][0];
      mean_src_y += src[static_cast<size_t>(i)][1];
      mean_dst_x += kArcFaceTemplate[i][0];
      mean_dst_y += kArcFaceTemplate[i][1];
    }
    mean_src_x /= 5.0f;
    mean_src_y /= 5.0f;
    mean_dst_x /= 5.0f;
    mean_dst_y /= 5.0f;

    float denom = 0.0f;
    float a_num = 0.0f;
    float b_num = 0.0f;
    for (int i = 0; i < 5; ++i) {
      const float dx = kArcFaceTemplate[i][0] - mean_dst_x;
      const float dy = kArcFaceTemplate[i][1] - mean_dst_y;
      const float sx = src[static_cast<size_t>(i)][0] - mean_src_x;
      const float sy = src[static_cast<size_t>(i)][1] - mean_src_y;
      denom += dx * dx + dy * dy;
      a_num += sx * dx + sy * dy;
      b_num += sy * dx - sx * dy;
    }
    if (denom <= 1.0e-6f) {
      return false;
    }
    const float a = a_num / denom;
    const float b = b_num / denom;
    const float scale = std::sqrt(a * a + b * b);
    if (!std::isfinite(scale) || scale <= 1.0e-3f) {
      return false;
    }
    const float tx = mean_src_x - a * mean_dst_x + b * mean_dst_y;
    const float ty = mean_src_y - b * mean_dst_x - a * mean_dst_y;
    const float cx = (static_cast<float>(input_w_) - 1.0f) * 0.5f;
    const float cy = (static_cast<float>(input_h_) - 1.0f) * 0.5f;
    const float crop_scale = cfg_.face_rec_five_point_scale;

    out_nchw->assign(static_cast<size_t>(input_c_) * input_h_ * input_w_, 0.0f);
    for (int oy = 0; oy < input_h_; ++oy) {
      for (int ox = 0; ox < input_w_; ++ox) {
        const float x = cx + ((static_cast<float>(ox) + 0.5f) - cx) * crop_scale;
        const float y = cy + ((static_cast<float>(oy) + 0.5f) - cy) * crop_scale;
        const float px = a * x - b * y + tx;
        const float py = b * x + a * y + ty;
        float bgr[3] = {0.0f, 0.0f, 0.0f};
        sample_panorama_bgr(source, px, py, bgr);
        for (int c = 0; c < input_c_; ++c) {
          const float normalized = (bgr[c] / 255.0f - 0.5f) / 0.5f;
          (*out_nchw)[(static_cast<size_t>(c) * input_h_ + oy) * input_w_ + ox] = normalized;
        }
      }
    }
    return true;
  }

  void sample_panorama_bgr(const Image& source, float pano_x, float pano_y, float* bgr) const {
    if (bgr == nullptr) {
      return;
    }
    if (!std::isfinite(pano_x) || !std::isfinite(pano_y) ||
        meta_.base_map_width <= 0 || meta_.base_map_height <= 0) {
      return;
    }
    const int map_w = meta_.base_map_width;
    int px = static_cast<int>(std::lround(pano_x));
    if (cfg_.wrap_boundary_geometry) {
      // Fisheye panorama is circular horizontally: sample past one edge from the other edge.
      px %= map_w;
      if (px < 0) {
        px += map_w;
      }
    } else {
      px = std::max(0, std::min(px, map_w - 1));
    }
    const int py = std::max(0, std::min(static_cast<int>(std::lround(pano_y)), meta_.base_map_height - 1));
    const size_t map_idx = static_cast<size_t>(py) * meta_.base_map_width + px;
    const float sx = (*base_map_x_)[map_idx];
    const float sy = (*base_map_y_)[map_idx];
    if (!std::isfinite(sx) || !std::isfinite(sy)) {
      return;
    }
    sample_source_bilinear(source, sx, sy, bgr);
  }

  static void sample_source_bilinear(const Image& source, float x, float y, float* bgr) {
    if (x < 0.0f || y < 0.0f || x > source.width - 1 || y > source.height - 1) {
      return;
    }
    const int x0 = std::max(0, std::min(static_cast<int>(std::floor(x)), source.width - 1));
    const int y0 = std::max(0, std::min(static_cast<int>(std::floor(y)), source.height - 1));
    const int x1 = std::min(x0 + 1, source.width - 1);
    const int y1 = std::min(y0 + 1, source.height - 1);
    const float tx = x - static_cast<float>(x0);
    const float ty = y - static_cast<float>(y0);
    for (int c = 0; c < 3; ++c) {
      const float v00 = source.bgr[(static_cast<size_t>(y0) * source.width + x0) * 3 + c];
      const float v01 = source.bgr[(static_cast<size_t>(y0) * source.width + x1) * 3 + c];
      const float v10 = source.bgr[(static_cast<size_t>(y1) * source.width + x0) * 3 + c];
      const float v11 = source.bgr[(static_cast<size_t>(y1) * source.width + x1) * 3 + c];
      bgr[c] = (1.0f - ty) * ((1.0f - tx) * v00 + tx * v01) +
               ty * ((1.0f - tx) * v10 + tx * v11);
    }
  }

	  float identity_score(const std::string& name, const std::vector<float>& feature) const {
	    float best = 0.0f;
	    for (const FaceIdentity& id : identities_) {
	      if (id.name != name) {
	        continue;
	      }
	      best = std::max(best, identity_score_for(id, feature));
	    }
	    return best;
	  }

	  float identity_score_for(const FaceIdentity& id, const std::vector<float>& feature) const {
	    float best = 0.0f;
	    for (const auto& old : id.features) {
	      if (old.size() == feature.size()) {
	        best = std::max(best, dot(old, feature));
	      }
	    }
	    return best;
	  }

	  float best_other_identity_score(const std::string& name,
	                                  const std::vector<float>& feature) const {
	    float best = 0.0f;
	    for (const FaceIdentity& id : identities_) {
	      if (id.name == name) {
	        continue;
	      }
	      best = std::max(best, identity_score_for(id, feature));
	    }
	    return best;
	  }

  void match_identity(const std::vector<float>& feature,
                      std::string* best_name,
                      float* best_score,
                      float* second_score) const {
    if (best_name != nullptr) {
      best_name->clear();
    }
    if (best_score != nullptr) {
      *best_score = 0.0f;
    }
    if (second_score != nullptr) {
      *second_score = 0.0f;
    }
    std::string best;
	    float best_val = 0.0f;
	    float second_val = 0.0f;
	    for (const FaceIdentity& id : identities_) {
	      const float score = identity_score_for(id, feature);
	      if (score > best_val) {
	        second_val = best_val;
	        best_val = score;
        best = id.name;
      } else if (score > second_val) {
        second_val = score;
      }
    }
    if (best_name != nullptr) {
      *best_name = best;
    }
    if (best_score != nullptr) {
      *best_score = best_val;
    }
    if (second_score != nullptr) {
      *second_score = second_val;
    }
  }

  void match_identity_excluding(const std::vector<float>& feature,
                                const std::string& excluded_name,
                                std::string* best_name,
                                float* best_score,
                                float* second_score) const {
    if (best_name != nullptr) {
      best_name->clear();
    }
    if (best_score != nullptr) {
      *best_score = 0.0f;
    }
    if (second_score != nullptr) {
      *second_score = 0.0f;
    }
    std::string best;
    float best_val = 0.0f;
    float second_val = 0.0f;
    for (const FaceIdentity& id : identities_) {
      if (id.name == excluded_name) {
        continue;
      }
      const float score = identity_score_for(id, feature);
      if (score > best_val) {
        second_val = best_val;
        best_val = score;
        best = id.name;
      } else if (score > second_val) {
        second_val = score;
      }
    }
    if (best_name != nullptr) {
      *best_name = best;
    }
    if (best_score != nullptr) {
      *best_score = best_val;
    }
    if (second_score != nullptr) {
      *second_score = second_val;
    }
  }

  static bool parse_dynamic_face_number(const std::string& name, int* out) {
    if (out != nullptr) {
      *out = 0;
    }
    if (name.size() <= 4 || name.substr(0, 4) != "face") {
      return false;
    }
    char* end = nullptr;
    const long value = std::strtol(name.c_str() + 4, &end, 10);
    if (end != name.c_str() + name.size() || value <= 0) {
      return false;
    }
    if (out != nullptr) {
      *out = static_cast<int>(value);
    }
    return true;
  }

  size_t identity_feature_count(const std::string& name) const {
    for (const FaceIdentity& id : identities_) {
      if (id.name == name) {
        return id.features.size();
      }
    }
    return 0;
  }

  bool relink_target_is_older(const std::string& current_name,
                              const std::string& candidate_name) const {
    if (candidate_name.empty() || candidate_name == current_name) {
      return false;
    }
    if (!is_dynamic_identity(current_name)) {
      return false;
    }
    if (!is_dynamic_identity(candidate_name)) {
      return true;
    }
    int current_id = 0;
    int candidate_id = 0;
    if (!parse_dynamic_face_number(current_name, &current_id) ||
        !parse_dynamic_face_number(candidate_name, &candidate_id)) {
      return false;
    }
    return candidate_id > 0 && current_id > 0 && candidate_id < current_id;
  }

  bool find_relink_candidate(const std::string& current_face_id,
                             const std::vector<float>& feature,
                             const FaceFeatureQuality& quality,
                             std::string* relink_name,
                             float* relink_score,
                             float* relink_second_score) const {
    if (relink_name != nullptr) {
      relink_name->clear();
    }
    if (relink_score != nullptr) {
      *relink_score = 0.0f;
    }
    if (relink_second_score != nullptr) {
      *relink_second_score = 0.0f;
    }
    if (!cfg_.face_rec_relink || current_face_id.empty() || !quality.sample_ok ||
        !is_dynamic_identity(current_face_id) ||
        identity_feature_count(current_face_id) >
            static_cast<size_t>(cfg_.face_rec_relink_max_current_samples)) {
      return false;
    }
    std::string best_name;
    float best_score = 0.0f;
    float second_score = 0.0f;
    match_identity_excluding(feature, current_face_id, &best_name, &best_score, &second_score);
    if (!relink_target_is_older(current_face_id, best_name)) {
      return false;
    }
    if (best_score < cfg_.face_rec_relink_threshold ||
        best_score < second_score + cfg_.face_rec_relink_margin) {
      return false;
    }
    if (relink_name != nullptr) {
      *relink_name = best_name;
    }
    if (relink_score != nullptr) {
      *relink_score = best_score;
    }
    if (relink_second_score != nullptr) {
      *relink_second_score = second_score;
    }
    return true;
  }

  static float dot(const std::vector<float>& a, const std::vector<float>& b) {
    const size_t n = std::min(a.size(), b.size());
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
      sum += static_cast<double>(a[i]) * b[i];
    }
    return static_cast<float>(sum);
  }

  FaceUpdateDecision maybe_update_identity(const std::string& name,
                                           const std::vector<float>& feature,
                                           const FaceFeatureQuality& quality,
                                           bool locked_track_sample,
                                           int frame_id) {
    FaceUpdateDecision decision;
    decision.sample_similarity =
        locked_track_sample ? cfg_.face_rec_locked_sample_similarity
                            : cfg_.face_rec_pose_sample_similarity;
    decision.pose_library_max_similarity =
        cfg_.face_rec_pose_library_max_similarity;
    for (FaceIdentity& id : identities_) {
      if (id.name != name || !id.dynamic) {
        continue;
      }
      decision.best_existing = identity_score_for(id, feature);
      std::vector<const std::vector<float>*> pose_refs;
      if (id.features.size() > 1) {
        for (size_t i = 1; i < id.features.size(); ++i) {
          pose_refs.push_back(&id.features[i]);
        }
      } else if (!id.features.empty()) {
        pose_refs.push_back(&id.features[0]);
      }
      for (const std::vector<float>* old_feature : pose_refs) {
        if (old_feature == nullptr || old_feature->size() != feature.size()) {
          continue;
        }
        const float score = dot(*old_feature, feature);
        if (!decision.has_pose_library_similarity ||
            score > decision.pose_library_similarity) {
          decision.pose_library_similarity = score;
          decision.has_pose_library_similarity = true;
        }
      }

      const bool locked_pose_quality_ok =
          locked_track_sample &&
          quality.min_side >= cfg_.face_rec_update_min_box_size &&
          quality.det_score >= cfg_.face_rec_update_min_score &&
          quality.min_keypoint_conf >= cfg_.face_rec_min_keypoint_conf;
      decision.locked_pose_quality_ok = locked_pose_quality_ok;
      decision.pose_sample_quality_ok = quality.sample_ok || locked_pose_quality_ok;
      decision.pose_sample_missing_frames =
          (id.last_pose_sample_frame > 0 && frame_id > 0)
              ? frame_id - id.last_pose_sample_frame
              : -1;

      const bool pose_library_ok =
          !decision.has_pose_library_similarity ||
          decision.pose_library_similarity >= cfg_.face_rec_pose_sample_similarity;
      const bool pose_diversity_ok =
          !decision.has_pose_library_similarity ||
          decision.pose_library_similarity <= cfg_.face_rec_pose_library_max_similarity;
      const bool pose_interval_ok =
          cfg_.face_rec_pose_sample_interval <= 0 ||
          decision.pose_sample_missing_frames < 0 ||
          decision.pose_sample_missing_frames >= cfg_.face_rec_pose_sample_interval;
      const bool can_update_primary =
          quality.primary_ok && decision.best_existing >= cfg_.face_rec_update_similarity;
      const bool can_add_sample =
          decision.pose_sample_quality_ok &&
          decision.best_existing >= decision.sample_similarity &&
          decision.best_existing < 1.0f - cfg_.face_rec_min_sample_diversity &&
          pose_library_ok &&
          pose_diversity_ok &&
          pose_interval_ok;
      if (!can_update_primary && !can_add_sample) {
        if (!decision.pose_sample_quality_ok) {
          decision.reason = "sample_not_ok";
        } else if (decision.best_existing < decision.sample_similarity) {
          decision.reason = "sample_low_identity_similarity";
        } else if (decision.best_existing >= 1.0f - cfg_.face_rec_min_sample_diversity) {
          decision.reason = "sample_too_similar_to_existing";
        } else if (!pose_library_ok) {
          decision.reason = "sample_low_pose_library_similarity";
        } else if (!pose_diversity_ok) {
          decision.reason = "sample_too_similar_to_pose_library";
        } else if (!pose_interval_ok) {
          decision.reason = "sample_interval";
        } else {
          decision.reason = "primary_not_ok";
        }
        return decision;
      }

      decision.best_other = best_other_identity_score(name, feature);
      if (decision.best_existing < decision.best_other + cfg_.face_rec_margin) {
        decision.reason = "sample_ambiguous_with_other_faceid";
        return decision;
      }

      bool primary_updated = false;
      bool sample_updated = false;
      if (can_update_primary) {
        update_primary_feature(id, feature, quality.quality_score);
        primary_updated = true;
      }
      if (can_add_sample) {
        id.features.push_back(feature);
        trim_identity_samples(&id);
        if (frame_id > 0) {
          id.last_pose_sample_frame = frame_id;
        }
        sample_updated = true;
      }
      if (!primary_updated && !sample_updated) {
        decision.reason = "sample_rejected_by_diversity";
        return decision;
      }
      save_dynamic_identity(id);
      if (primary_updated && sample_updated) {
        decision.kind = "primary_sample";
        decision.reason = "primary_sample_updated";
      } else if (primary_updated) {
        decision.kind = "primary";
        decision.reason = "primary_updated";
      } else {
        decision.kind = "sample";
        decision.reason = "sample_updated";
      }
      return decision;
    }
    decision.reason = "identity_not_found";
    return decision;
  }

  void apply_update_debug(const FaceUpdateDecision& update,
                          FaceRecognitionBinding* binding) const {
    if (binding == nullptr) {
      return;
    }
    binding->debug_update_reason = update.reason;
    binding->debug_identity_score = update.best_existing;
    binding->debug_sample_similarity = update.sample_similarity;
    binding->debug_pose_library_similarity = update.pose_library_similarity;
    binding->debug_pose_library_max_similarity = update.pose_library_max_similarity;
    binding->debug_has_pose_library_similarity = update.has_pose_library_similarity;
    binding->debug_pose_sample_missing_frames = update.pose_sample_missing_frames;
    binding->debug_pose_sample_quality_ok = update.pose_sample_quality_ok;
    binding->debug_locked_pose_quality_ok = update.locked_pose_quality_ok;
  }

  bool merge_dynamic_identity_into(const std::string& from_name,
                                   const std::string& to_name) {
    if (from_name.empty() || to_name.empty() || from_name == to_name ||
        !is_dynamic_identity(from_name)) {
      return false;
    }
    int from_index = -1;
    int to_index = -1;
    for (size_t i = 0; i < identities_.size(); ++i) {
      if (identities_[i].name == from_name) {
        from_index = static_cast<int>(i);
      } else if (identities_[i].name == to_name) {
        to_index = static_cast<int>(i);
      }
    }
    if (from_index < 0 || to_index < 0) {
      return false;
    }
    FaceIdentity& target = identities_[static_cast<size_t>(to_index)];
    const FaceIdentity& source = identities_[static_cast<size_t>(from_index)];
    if (target.dynamic) {
      for (const std::vector<float>& feature : source.features) {
        if (static_cast<int>(feature.size()) != feature_dim_) {
          continue;
        }
        const float best_existing = identity_score_for(target, feature);
        if (best_existing < 1.0f - cfg_.face_rec_min_sample_diversity) {
          target.features.push_back(feature);
        }
      }
      trim_identity_samples(&target);
      save_dynamic_identity(target);
    }
    identities_.erase(identities_.begin() + from_index);
    for (auto& item : bindings_) {
      if (item.second.face_id == from_name) {
        item.second.face_id = to_name;
        item.second.dynamic = is_dynamic_identity(to_name);
      }
    }
    for (auto& item : current_) {
      if (item.second.face_id == from_name) {
        item.second.face_id = to_name;
        item.second.dynamic = is_dynamic_identity(to_name);
      }
    }
    const std::string old_path = join_path(cfg_.face_rec_dynamic_library_dir, from_name + ".npy");
    unlink(old_path.c_str());
    return true;
  }

  bool pending_dynamic_enroll_compatible(const PendingDynamicEnroll& pending,
                                         const std::vector<float>& feature) const {
    if (pending.features.empty()) {
      return true;
    }
    for (const std::vector<float>& old_feature : pending.features) {
      if (old_feature.size() != feature.size() ||
          dot(old_feature, feature) < cfg_.face_rec_threshold) {
        return false;
      }
    }
    return true;
  }

  static bool pending_dynamic_enroll_has_primary(
      const PendingDynamicEnroll& pending) {
    for (const FaceFeatureQuality& quality : pending.qualities) {
      if (quality.primary_ok) {
        return true;
      }
    }
    return false;
  }

  std::string update_pending_dynamic_enroll(int binding_id,
                                            const std::vector<float>& feature,
                                            const FaceFeatureQuality& quality,
                                            int frame_id,
                                            int* pending_hits,
                                            bool* pending_reset) {
    if (pending_hits != nullptr) {
      *pending_hits = 0;
    }
    if (pending_reset != nullptr) {
      *pending_reset = false;
    }
    if (binding_id <= 0 || feature.empty()) {
      return "";
    }
    if (cfg_.face_rec_dynamic_enroll_confirm_frames <= 1) {
      if (pending_hits != nullptr) {
        *pending_hits = 1;
      }
      return create_dynamic_identity_from_samples(
          std::vector<std::vector<float>>(1, feature),
          std::vector<FaceFeatureQuality>(1, quality),
          std::vector<int>(1, frame_id));
    }

    PendingDynamicEnroll& pending = pending_dynamic_enroll_[binding_id];
    if (!pending_dynamic_enroll_compatible(pending, feature)) {
      pending.features.clear();
      pending.qualities.clear();
      pending.frames.clear();
      if (pending_reset != nullptr) {
        *pending_reset = true;
      }
    }
    pending.features.push_back(feature);
    pending.qualities.push_back(quality);
    pending.frames.push_back(frame_id);
    pending.last_frame = frame_id;
    if (pending_hits != nullptr) {
      *pending_hits = static_cast<int>(pending.features.size());
    }
    if (static_cast<int>(pending.features.size()) <
        cfg_.face_rec_dynamic_enroll_confirm_frames) {
      return "";
    }
    if (!pending_dynamic_enroll_has_primary(pending)) {
      return "";
    }

    const std::string name =
        create_dynamic_identity_from_samples(
            pending.features, pending.qualities, pending.frames);
    pending_dynamic_enroll_.erase(binding_id);
    return name;
  }

	  std::string create_dynamic_identity(const std::vector<float>& feature,
	                                      const FaceFeatureQuality& quality) {
	    return create_dynamic_identity_from_samples(
	        std::vector<std::vector<float>>(1, feature),
	        std::vector<FaceFeatureQuality>(1, quality),
	        std::vector<int>());
	  }

  std::string create_dynamic_identity_from_samples(
      const std::vector<std::vector<float>>& features,
      const std::vector<FaceFeatureQuality>& qualities,
      const std::vector<int>& frames) {
    if (features.empty()) {
      return "";
    }
    size_t primary_idx = 0;
    float best_quality = -1.0f;
    for (size_t i = 0; i < features.size(); ++i) {
      const float q = i < qualities.size() ? qualities[i].quality_score : 0.0f;
      const bool primary_ok = i < qualities.size() && qualities[i].primary_ok;
      const bool current_primary =
          primary_idx < qualities.size() && qualities[primary_idx].primary_ok;
      if ((primary_ok && !current_primary) ||
          (primary_ok == current_primary && q > best_quality)) {
        best_quality = q;
        primary_idx = i;
      }
    }

    FaceIdentity id;
    id.name = "face" + std::to_string(next_dynamic_id_++);
    id.dynamic = true;
    id.primary_quality = std::max(0.0f, best_quality);
    id.features.push_back(features[primary_idx]);
    const int primary_frame =
        primary_idx < frames.size() ? frames[primary_idx] : static_cast<int>(primary_idx);
    std::vector<size_t> candidates;
    candidates.reserve(features.size());
    for (size_t i = 0; i < features.size(); ++i) {
      if (i == primary_idx || static_cast<int>(features[i].size()) != feature_dim_) {
        continue;
      }
      if (i < qualities.size() && !qualities[i].sample_ok) {
        continue;
      }
      candidates.push_back(i);
    }
    std::sort(candidates.begin(), candidates.end(), [&](size_t a, size_t b) {
      const float qa = a < qualities.size() ? qualities[a].quality_score : 0.0f;
      const float qb = b < qualities.size() ? qualities[b].quality_score : 0.0f;
      if (qa != qb) {
        return qa > qb;
      }
      const int fa = a < frames.size() ? frames[a] : static_cast<int>(a);
      const int fb = b < frames.size() ? frames[b] : static_cast<int>(b);
      return fa < fb;
    });
    std::vector<int> selected_frames;
    selected_frames.push_back(primary_frame);
    const int seed_limit = std::min(
        std::max(0, cfg_.face_rec_pending_seed_samples),
        std::max(0, cfg_.face_rec_max_samples_per_id - 1));
    int selected_seed_count = 0;
    for (size_t idx : candidates) {
      if (selected_seed_count >= seed_limit) {
        break;
      }
      const int frame = idx < frames.size() ? frames[idx] : static_cast<int>(idx);
      if (cfg_.face_rec_pose_sample_interval > 0) {
        bool too_close_in_time = false;
        for (int old_frame : selected_frames) {
          if (std::abs(frame - old_frame) < cfg_.face_rec_pose_sample_interval) {
            too_close_in_time = true;
            break;
          }
        }
        if (too_close_in_time) {
          continue;
        }
      }
      float best_pose = -2.0f;
      for (const std::vector<float>& old_feature : id.features) {
        if (old_feature.size() == features[idx].size()) {
          best_pose = std::max(best_pose, dot(old_feature, features[idx]));
        }
      }
      if (best_pose < cfg_.face_rec_pose_sample_similarity ||
          best_pose > cfg_.face_rec_pose_library_max_similarity ||
          best_pose >= 1.0f - cfg_.face_rec_min_sample_diversity) {
        continue;
      }
      id.features.push_back(features[idx]);
      selected_frames.push_back(frame);
      selected_seed_count += 1;
    }
    if (!selected_frames.empty()) {
      id.last_pose_sample_frame =
          *std::max_element(selected_frames.begin(), selected_frames.end());
    }
    trim_identity_samples(&id);
    identities_.push_back(id);
    save_dynamic_identity(identities_.back());
    return identities_.back().name;
  }

		  void update_primary_feature(FaceIdentity& id,
	                              const std::vector<float>& feature,
	                              float quality_score) {
	    if (feature.empty()) {
	      return;
	    }
	    if (id.features.empty() || id.features[0].size() != feature.size()) {
	      if (id.features.empty()) {
	        id.features.push_back(feature);
	      } else {
	        id.features[0] = feature;
	      }
	      id.primary_quality = quality_score;
	      return;
	    }
	    if (id.primary_quality <= 0.0f || quality_score > id.primary_quality + 0.2f) {
	      id.features[0] = feature;
	      id.primary_quality = std::max(id.primary_quality, quality_score);
	      return;
	    }
	    const float alpha = cfg_.face_rec_primary_ema_alpha;
	    const float keep = 1.0f - alpha;
	    for (size_t i = 0; i < feature.size(); ++i) {
	      id.features[0][i] = keep * id.features[0][i] + alpha * feature[i];
	    }
	    normalize_feature(id.features[0]);
	    id.primary_quality = std::max(id.primary_quality, quality_score);
	  }

	  void trim_identity_samples(FaceIdentity* id) const {
	    if (id == nullptr || id->features.empty()) {
	      return;
	    }
	    const size_t max_samples =
	        static_cast<size_t>(std::max(1, cfg_.face_rec_max_samples_per_id));
	    if (id->features.size() <= max_samples) {
	      return;
	    }
	    if (max_samples == 1) {
	      id->features.resize(1);
	      return;
	    }
	    const size_t remove_count = id->features.size() - max_samples;
	    const size_t erase_end =
	        std::min(id->features.size(), static_cast<size_t>(1) + remove_count);
	    id->features.erase(id->features.begin() + 1, id->features.begin() + erase_end);
	  }

  void save_dynamic_identity(const FaceIdentity& id) const {
    if (!cfg_.face_rec_dynamic_library || !id.dynamic || id.features.empty()) {
      return;
    }
    mkdir(cfg_.face_rec_dynamic_library_dir.c_str(), 0775);
    write_npy_float32_2d(
        join_path(cfg_.face_rec_dynamic_library_dir, id.name + ".npy"),
        id.features,
        feature_dim_);
  }

  bool is_dynamic_identity(const std::string& name) const {
    for (const FaceIdentity& id : identities_) {
      if (id.name == name) {
        return id.dynamic;
      }
    }
    return false;
  }

  void update_next_dynamic_id(const std::string& name) {
    if (name.size() <= 4 || name.substr(0, 4) != "face") {
      return;
    }
    char* end = nullptr;
    const long value = std::strtol(name.c_str() + 4, &end, 10);
    if (end != name.c_str() + name.size() || value <= 0) {
      return;
    }
    next_dynamic_id_ = std::max(next_dynamic_id_, static_cast<int>(value) + 1);
  }

  const Config& cfg_;
  const MapMeta& meta_;
  const std::vector<float>* base_map_x_ = nullptr;
  const std::vector<float>* base_map_y_ = nullptr;
  std::array<AdaFaceHandle, 3> adafaces_;
  int adaface_runner_count_ = 0;
  int input_h_ = 0;
  int input_w_ = 0;
  int input_c_ = 0;
  int feature_dim_ = 0;
  int next_dynamic_id_ = 1;
  std::vector<FaceIdentity> identities_;
  std::map<int, FaceRecognitionBinding> bindings_;
  std::map<int, FaceRecognitionBinding> current_;
  std::map<int, int> last_attempt_frame_;
  std::map<int, int> last_active_frame_;
  std::set<int> binding_reappeared_after_gap_;
  std::map<int, int> reappeared_probe_until_frame_;
	  std::map<int, PendingFaceRelink> pending_face_relink_;
  std::map<int, PendingDynamicEnroll> pending_dynamic_enroll_;
	  std::map<int, DisplayFaceMemory> display_face_memory_;
  std::set<std::pair<int, std::string>> rejected_tentative_faces_;
  std::unique_ptr<TurboJpegEncoder> debug_crop_encoder_;
  std::thread async_worker_;
  std::mutex async_mutex_;
  std::condition_variable async_cv_;
  std::deque<FaceRecInferTask> async_queue_;
  std::deque<FaceRecInferTask> async_done_;
  std::set<int> async_pending_bindings_;
  bool async_stop_ = false;
  int debug_crop_errors_ = 0;
};
