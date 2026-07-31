class MeetEyeRuntime {
 public:
  explicit MeetEyeRuntime(const Config& cfg)
      : cfg_(cfg),
        meta_(load_map_meta(cfg.map_dir)),
        map_x_(read_floats(join_path(cfg.map_dir, "map_x.bin"))),
        map_y_(read_floats(join_path(cfg.map_dir, "map_y.bin"))) {
    cfg_.projection_mode = meta_.projection_mode;
    if (is_wide_angle_projection(meta_.projection_mode)) {
      if (meta_.num_slices != 1) {
        throw std::runtime_error("wide-angle projection currently requires exactly 1 slice");
      }
      cfg_.wrap_boundary_geometry = false;
      cfg_.boundary_recover = false;
      cfg_.final_boundary_dedup = false;
      cfg_.bound_input = false;
      cfg_.staging_copy_input = false;
      cfg_.staging_pipeline = false;
    } else if (meta_.num_slices != 3) {
      throw std::runtime_error("fisheye panorama projection requires exactly 3 slices");
    }
    const size_t expected_map_values =
        static_cast<size_t>(meta_.num_slices) * meta_.roi_h * meta_.roi_w;
    if (map_x_.size() != expected_map_values || map_y_.size() != expected_map_values) {
      throw std::runtime_error("map_x/map_y size mismatch with meta.txt");
    }
    const std::string base_map_x_path = join_path(cfg.map_dir, "base_map_x.bin");
    const std::string base_map_y_path = join_path(cfg.map_dir, "base_map_y.bin");
    if (file_exists(base_map_x_path) || file_exists(base_map_y_path)) {
      if (!file_exists(base_map_x_path) || !file_exists(base_map_y_path)) {
        throw std::runtime_error("base_map_x.bin/base_map_y.bin must exist together");
      }
      if (meta_.base_map_width <= 0 || meta_.base_map_height <= 0) {
        throw std::runtime_error("base map files exist but base_map_width/base_map_height are missing in meta.txt");
      }
      base_map_x_ = read_floats(base_map_x_path);
      base_map_y_ = read_floats(base_map_y_path);
      const size_t expected_base_values =
          static_cast<size_t>(meta_.base_map_width) * static_cast<size_t>(meta_.base_map_height);
      if (base_map_x_.size() != expected_base_values || base_map_y_.size() != expected_base_values) {
        throw std::runtime_error("base_map_x/base_map_y size mismatch with meta.txt");
      }
    }

    roi_rects_.resize(static_cast<size_t>(meta_.num_slices) * 4);
    slice_shapes_.resize(static_cast<size_t>(meta_.num_slices) * 2);
    gains_.resize(static_cast<size_t>(meta_.num_slices));
    pads_.resize(static_cast<size_t>(meta_.num_slices) * 2);
    starts_.resize(static_cast<size_t>(meta_.num_slices));
    wraps_.resize(static_cast<size_t>(meta_.num_slices));

    for (int i = 0; i < meta_.num_slices; ++i) {
      const SliceMeta& s = meta_.slices[static_cast<size_t>(i)];
      roi_rects_[static_cast<size_t>(i) * 4 + 0] = s.left;
      roi_rects_[static_cast<size_t>(i) * 4 + 1] = s.top;
      roi_rects_[static_cast<size_t>(i) * 4 + 2] = s.new_width;
      roi_rects_[static_cast<size_t>(i) * 4 + 3] = s.new_height;
      slice_shapes_[static_cast<size_t>(i) * 2 + 0] = s.slice_height;
      slice_shapes_[static_cast<size_t>(i) * 2 + 1] = s.slice_width;
      gains_[static_cast<size_t>(i)] = s.gain;
      pads_[static_cast<size_t>(i) * 2 + 0] = s.left;
      pads_[static_cast<size_t>(i) * 2 + 1] = s.top;
      starts_[static_cast<size_t>(i)] = static_cast<float>(s.start_x);
      wraps_[static_cast<size_t>(i)] = s.wrap_around;
    }

    char err[4096] = {0};
    int ret = face_rknn_parallel_create(cfg_.model_path.c_str(), rknn_.out(), err, sizeof(err));
    if (ret != 0) {
      throw std::runtime_error(std::string("face_rknn_parallel_create failed: ") + err);
    }

    ret = face_rknn_parallel_get_shape(
        rknn_.get(), &input_h_, &input_w_, &input_c_, &channels_, &anchors_);
    if (ret != 0) {
      throw std::runtime_error("face_rknn_parallel_get_shape failed");
    }
    if (input_h_ != meta_.imgsz || input_w_ != meta_.imgsz || input_c_ != 3) {
      std::ostringstream oss;
      oss << "model input shape " << input_w_ << "x" << input_h_ << "x" << input_c_
          << " does not match map imgsz=" << meta_.imgsz;
      throw std::runtime_error(oss.str());
    }
  }

  const MapMeta& meta() const { return meta_; }
  bool supports_staging_pipeline() const {
    return meta_.num_slices == 3 && !is_wide_angle_projection(meta_.projection_mode);
  }
  bool supports_wide_angle_yolo_pipeline() const {
    return meta_.num_slices == 1 &&
           is_wide_angle_projection(meta_.projection_mode) &&
           cfg_.wide_angle_yolo_workers > 1;
  }
  int wide_angle_yolo_worker_count() const {
    if (!supports_wide_angle_yolo_pipeline()) {
      return 1;
    }
    return std::max(1, std::min(cfg_.wide_angle_yolo_workers, 2));
  }
  int channels() const { return channels_; }
  int anchors() const { return anchors_; }
  const std::vector<float>* base_map_x() const {
    return base_map_x_.empty() ? nullptr : &base_map_x_;
  }
  const std::vector<float>* base_map_y() const {
    return base_map_y_.empty() ? nullptr : &base_map_y_;
  }

  PreparedStagingFrame prepare_staging_frame(PreparedStagingFrame frame) {
    const Image& image = frame.image;
    if (meta_.img_width > 0 && meta_.img_height > 0 &&
        (image.width != meta_.img_width || image.height != meta_.img_height)) {
      std::ostringstream oss;
      oss << "image shape " << image.width << "x" << image.height
          << " does not match exported map source " << meta_.img_width << "x" << meta_.img_height;
      throw std::runtime_error(oss.str());
    }

    frame.valid = true;
    frame.result = FrameResult();
    frame.result.frame_index = frame.frame_index;
    frame.result.image_path = frame.image_path;
    frame.result.frame_width = image.width;
    frame.result.frame_height = image.height;
    frame.result.detections.assign(
        static_cast<size_t>(cfg_.max_output_dets) * kDetectionFields, 0.0f);

    double t0 = steady_seconds();
    ensure_opencl(image);
    double t1 = steady_seconds();
    frame.result.profile_ms[kProfileOpenclEnsure] = (t1 - t0) * 1000.0;

    frame.original_width =
        meta_.slices.empty() ? static_cast<float>(meta_.process_width)
                             : static_cast<float>(meta_.slices[0].original_width);

    if (!prepare_bound_inputs(frame.result)) {
      throw std::runtime_error("staging pipeline requires RKNN bound input memory");
    }

    ensure_staging_buffers();
    char err[4096] = {0};
    t0 = steady_seconds();
    const int ret = ds_opencl_fused_run_split(
        opencl_.get(),
        image.bgr.data(),
        staging_ptrs_.data(),
        meta_.num_slices,
        frame.result.remap_timings,
        err,
        sizeof(err));
    t1 = steady_seconds();
    frame.result.profile_ms[kProfileOpenclRunOuter] = (t1 - t0) * 1000.0;
    if (ret != 0) {
      throw std::runtime_error(std::string("ds_opencl_fused_run_split(staging) failed: ") + err);
    }
    return frame;
  }

  std::future<PreparedStagingFrame> start_staging_inference(PreparedStagingFrame frame) {
    if (!frame.valid) {
      throw std::runtime_error("cannot infer invalid staging frame");
    }
    copy_staging_to_bound_inputs(frame.result);
    return std::async(std::launch::async, [this, frame = std::move(frame)]() mutable {
      run_bound_inference(frame.original_width, false, frame.result);
      frame.result.detection_count =
          std::max(0, std::min(frame.result.detection_count, cfg_.max_output_dets));
      return frame;
    });
  }

  PreparedStagingFrame prepare_wide_angle_frame(PreparedStagingFrame frame) {
    if (meta_.num_slices != 1 || !is_wide_angle_projection(meta_.projection_mode)) {
      throw std::runtime_error("wide-angle frame pipeline requires a single wide-angle slice");
    }
    const Image& image = frame.image;
    if (meta_.img_width > 0 && meta_.img_height > 0 &&
        (image.width != meta_.img_width || image.height != meta_.img_height)) {
      std::ostringstream oss;
      oss << "image shape " << image.width << "x" << image.height
          << " does not match exported map source " << meta_.img_width << "x" << meta_.img_height;
      throw std::runtime_error(oss.str());
    }

    frame.valid = true;
    frame.result = FrameResult();
    frame.result.frame_index = frame.frame_index;
    frame.result.image_path = frame.image_path;
    frame.result.frame_width = image.width;
    frame.result.frame_height = image.height;
    frame.result.detections.assign(
        static_cast<size_t>(cfg_.max_output_dets) * kDetectionFields, 0.0f);

    double t0 = steady_seconds();
    ensure_opencl(image);
    double t1 = steady_seconds();
    frame.result.profile_ms[kProfileOpenclEnsure] = (t1 - t0) * 1000.0;

    t0 = steady_seconds();
    frame.rknn_inputs.assign(
        static_cast<size_t>(meta_.num_slices) * meta_.imgsz * meta_.imgsz * 3,
        114);
    t1 = steady_seconds();
    frame.result.profile_ms[kProfileRknnInputAlloc] = (t1 - t0) * 1000.0;

    char err[4096] = {0};
    t0 = steady_seconds();
    const int ret = ds_opencl_fused_run(
        opencl_.get(),
        image.bgr.data(),
        frame.rknn_inputs.data(),
        frame.result.remap_timings,
        err,
        sizeof(err));
    t1 = steady_seconds();
    frame.result.profile_ms[kProfileOpenclRunOuter] = (t1 - t0) * 1000.0;
    if (ret != 0) {
      throw std::runtime_error(std::string("ds_opencl_fused_run(wide-angle) failed: ") + err);
    }
    return frame;
  }

  std::future<PreparedStagingFrame> start_wide_angle_inference(PreparedStagingFrame frame,
                                                               int worker_index) {
    if (!frame.valid || frame.rknn_inputs.empty()) {
      throw std::runtime_error("cannot infer invalid wide-angle frame");
    }
    worker_index = std::max(0, std::min(worker_index, wide_angle_yolo_worker_count() - 1));
    return std::async(std::launch::async, [this, frame = std::move(frame), worker_index]() mutable {
      run_wide_angle_single_inference(frame.rknn_inputs.data(), worker_index, frame.result);
      frame.result.detection_count =
          std::max(0, std::min(frame.result.detection_count, cfg_.max_output_dets));
      return frame;
    });
  }

  FrameResult process(const Image& image, const std::string& image_path, int frame_index) {
    if (meta_.img_width > 0 && meta_.img_height > 0 &&
        (image.width != meta_.img_width || image.height != meta_.img_height)) {
      std::ostringstream oss;
      oss << "image shape " << image.width << "x" << image.height
          << " does not match exported map source " << meta_.img_width << "x" << meta_.img_height;
      throw std::runtime_error(oss.str());
    }

    FrameResult result;
    result.frame_index = frame_index;
    result.image_path = image_path;
    result.frame_width = image.width;
    result.frame_height = image.height;
    result.detections.assign(static_cast<size_t>(cfg_.max_output_dets) * kDetectionFields, 0.0f);

    double t0 = steady_seconds();
    ensure_opencl(image);
    double t1 = steady_seconds();
    result.profile_ms[kProfileOpenclEnsure] = (t1 - t0) * 1000.0;

    const float original_width =
        meta_.slices.empty() ? static_cast<float>(meta_.process_width)
                             : static_cast<float>(meta_.slices[0].original_width);

    char err[4096] = {0};
    int ret = 0;
    if (try_process_bound(image, original_width, result)) {
      result.detection_count = std::max(0, std::min(result.detection_count, cfg_.max_output_dets));
      return result;
    }

    t0 = steady_seconds();
    std::vector<uint8_t> rknn_inputs(
        static_cast<size_t>(meta_.num_slices) * meta_.imgsz * meta_.imgsz * 3,
        114);
    t1 = steady_seconds();
    result.profile_ms[kProfileRknnInputAlloc] = (t1 - t0) * 1000.0;

    t0 = steady_seconds();
    ret = ds_opencl_fused_run(
        opencl_.get(),
        image.bgr.data(),
        rknn_inputs.data(),
        result.remap_timings,
        err,
        sizeof(err));
    if (ret != 0) {
      throw std::runtime_error(std::string("ds_opencl_fused_run failed: ") + err);
    }
    t1 = steady_seconds();
    result.profile_ms[kProfileOpenclRunOuter] = (t1 - t0) * 1000.0;

    if (meta_.num_slices == 1) {
      run_wide_angle_single_inference(rknn_inputs.data(), 0, result);
    } else {
      t0 = steady_seconds();
      ret = face_rknn_parallel_infer_merged(
          rknn_.get(),
          rknn_inputs.data(),
          meta_.num_slices,
          input_h_,
          input_w_,
          input_c_,
          slice_shapes_.data(),
          gains_.data(),
          pads_.data(),
          starts_.data(),
          wraps_.data(),
          meta_.num_slices,
          original_width,
          cfg_.overlap_ratio,
          cfg_.merge_iou_threshold,
          cfg_.nms_iou_threshold,
          cfg_.conf_threshold,
          cfg_.decode_iou_threshold,
          cfg_.max_det,
          cfg_.max_nms,
          result.detections.data(),
          cfg_.max_output_dets,
          &result.detection_count,
          result.merge_stats,
          result.rknn_timings,
          err,
          sizeof(err));
      if (ret != 0) {
        throw std::runtime_error(
            std::string("face_rknn_parallel_infer_merged failed: ") + err);
      }
      t1 = steady_seconds();
      result.profile_ms[kProfileRknnTotalOuter] = (t1 - t0) * 1000.0;
    }

    result.detection_count = std::max(0, std::min(result.detection_count, cfg_.max_output_dets));
    return result;
  }

 private:
  void run_wide_angle_single_inference(const uint8_t* input,
                                       int worker_index,
                                       FrameResult& result) {
    if (meta_.num_slices != 1) {
      throw std::runtime_error("single-image inference requires exactly one slice");
    }
    const SliceMeta& slice = meta_.slices[0];
    char err[4096] = {0};
    const double t0 = steady_seconds();
    const int ret = face_rknn_parallel_infer_merged_single_core(
        rknn_.get(),
        worker_index,
        input,
        input_h_,
        input_w_,
        input_c_,
        slice.slice_height,
        slice.slice_width,
        slice.gain,
        slice.left,
        slice.top,
        cfg_.nms_iou_threshold,
        cfg_.conf_threshold,
        cfg_.max_det,
        cfg_.max_nms,
        result.detections.data(),
        cfg_.max_output_dets,
        &result.detection_count,
        result.merge_stats,
        result.rknn_timings,
        err,
        sizeof(err));
    const double t1 = steady_seconds();
    result.profile_ms[kProfileRknnTotalOuter] = (t1 - t0) * 1000.0;
    if (ret != 0) {
      std::ostringstream oss;
      oss << "face_rknn_parallel_infer_merged_single_core[" << worker_index << "] failed: "
          << err;
      throw std::runtime_error(oss.str());
    }
  }

  void disable_bound_once(const std::string& reason) {
    bound_disabled_ = true;
    bound_imported_ = false;
    if (!bound_warned_) {
      std::cerr << "[board_cpp] bound input disabled, fallback to CPU input path: "
                << reason << "\n";
      bound_warned_ = true;
    }
  }

  void disable_zero_copy_once(const std::string& reason) {
    bound_import_failed_ = true;
    bound_imported_ = false;
    if (!bound_import_warned_) {
      std::cerr << "[board_cpp] RKNN/OpenCL zero-copy disabled, fallback to bound-copy path: "
                << reason << "\n";
      bound_import_warned_ = true;
    }
  }

  bool prepare_bound_inputs(FrameResult& result) {
    if (!supports_staging_pipeline() || !cfg_.bound_input || bound_disabled_) {
      return false;
    }

    char err[4096] = {0};
    int ret = 0;
    if (!bound_prepared_) {
      const double t0 = steady_seconds();
      ret = face_rknn_parallel_prepare_bound_inputs(rknn_.get(), err, sizeof(err));
      const double t1 = steady_seconds();
      result.profile_ms[kProfileBoundPrepare] = (t1 - t0) * 1000.0;
      if (ret != 0) {
        disable_bound_once(std::string("face_rknn_parallel_prepare_bound_inputs failed: ") + err);
        return false;
      }
      bound_prepared_ = true;
    }

    if (!bound_ptrs_ready_) {
      std::fill(bound_ptrs_, bound_ptrs_ + 3, nullptr);
      std::fill(bound_ptr_sizes_, bound_ptr_sizes_ + 3, 0);
      err[0] = '\0';
      ret = face_rknn_parallel_get_bound_input_ptrs(
          rknn_.get(), bound_ptrs_, bound_ptr_sizes_, 3, err, sizeof(err));
      if (ret != meta_.num_slices) {
        std::ostringstream oss;
        oss << "face_rknn_parallel_get_bound_input_ptrs returned " << ret
            << ", expected " << meta_.num_slices;
        if (err[0] != '\0') {
          oss << ": " << err;
        }
        throw std::runtime_error(oss.str());
      }
      const uint64_t slice_bytes =
          static_cast<uint64_t>(meta_.imgsz) * static_cast<uint64_t>(meta_.imgsz) * 3ULL;
      for (int i = 0; i < meta_.num_slices; ++i) {
        if (bound_ptrs_[i] == nullptr || bound_ptr_sizes_[i] < slice_bytes) {
          std::ostringstream oss;
          oss << "invalid bound input ptr/size for slice " << i
              << ": ptr=" << static_cast<const void*>(bound_ptrs_[i])
              << " size=" << bound_ptr_sizes_[i];
          throw std::runtime_error(oss.str());
        }
      }
      bound_ptrs_ready_ = true;
    }

    if (bound_imported_ || bound_import_failed_) {
      return true;
    }

    if (cfg_.staging_copy_input) {
      bound_imported_ = false;
      bound_import_failed_ = true;
      return true;
    }

    std::fill(bound_fds_, bound_fds_ + 3, -1);
    std::fill(bound_sizes_, bound_sizes_ + 3, 0);
    const double t0 = steady_seconds();
    ret = face_rknn_parallel_get_bound_input_fds(
        rknn_.get(), bound_fds_, bound_sizes_, 3, err, sizeof(err));
    if (ret != meta_.num_slices) {
      const double t1 = steady_seconds();
      result.profile_ms[kProfileBoundImport] = (t1 - t0) * 1000.0;
      std::ostringstream oss;
      oss << "face_rknn_parallel_get_bound_input_fds returned " << ret
          << ", expected " << meta_.num_slices;
      if (err[0] != '\0') {
        oss << ": " << err;
      }
      disable_zero_copy_once(oss.str());
      return true;
    }
    for (int i = 0; i < meta_.num_slices; ++i) {
      if (bound_fds_[i] < 0 || bound_sizes_[i] == 0) {
        const double t1 = steady_seconds();
        result.profile_ms[kProfileBoundImport] = (t1 - t0) * 1000.0;
        std::ostringstream oss;
        oss << "invalid bound input fd/size for slice " << i
            << ": fd=" << bound_fds_[i] << " size=" << bound_sizes_[i];
        disable_zero_copy_once(oss.str());
        return true;
      }
    }

    err[0] = '\0';
    ret = ds_opencl_fused_import_output_fds(
        opencl_.get(), bound_fds_, bound_sizes_, meta_.num_slices, err, sizeof(err));
    const double t1 = steady_seconds();
    result.profile_ms[kProfileBoundImport] = (t1 - t0) * 1000.0;
    if (ret != 0) {
      disable_zero_copy_once(std::string("ds_opencl_fused_import_output_fds failed: ") + err);
      return true;
    }

    bound_imported_ = true;
    return true;
  }

  void ensure_staging_buffers() {
    const size_t slice_bytes =
        static_cast<size_t>(meta_.imgsz) * static_cast<size_t>(meta_.imgsz) * 3U;
    if (staging_buffers_.size() == static_cast<size_t>(meta_.num_slices) &&
        staging_ptrs_.size() == static_cast<size_t>(meta_.num_slices)) {
      bool ready = true;
      for (int i = 0; i < meta_.num_slices; ++i) {
        ready = ready &&
                staging_buffers_[static_cast<size_t>(i)].size() == slice_bytes &&
                staging_ptrs_[static_cast<size_t>(i)] ==
                    staging_buffers_[static_cast<size_t>(i)].data();
      }
      if (ready) {
        return;
      }
    }

    staging_buffers_.assign(static_cast<size_t>(meta_.num_slices), std::vector<uint8_t>());
    staging_ptrs_.assign(static_cast<size_t>(meta_.num_slices), nullptr);
    for (int i = 0; i < meta_.num_slices; ++i) {
      staging_buffers_[static_cast<size_t>(i)].assign(slice_bytes, 114);
      staging_ptrs_[static_cast<size_t>(i)] = staging_buffers_[static_cast<size_t>(i)].data();
    }
  }

  void copy_staging_to_bound_inputs(FrameResult& result) {
    const size_t slice_bytes =
        static_cast<size_t>(meta_.imgsz) * static_cast<size_t>(meta_.imgsz) * 3U;
    const double t0 = steady_seconds();
    for (int i = 0; i < meta_.num_slices; ++i) {
      if (bound_ptrs_[i] == nullptr || bound_ptr_sizes_[i] < slice_bytes ||
          staging_ptrs_[static_cast<size_t>(i)] == nullptr) {
        std::ostringstream oss;
        oss << "invalid staging/bound input ptr for slice " << i;
        throw std::runtime_error(oss.str());
      }
      std::memcpy(bound_ptrs_[i], staging_ptrs_[static_cast<size_t>(i)], slice_bytes);
    }
    const double t1 = steady_seconds();
    result.profile_ms[kProfileStagingCopy] = (t1 - t0) * 1000.0;
  }

  bool run_bound_inference(float original_width, bool external_device_input, FrameResult& result) {
    char err[4096] = {0};
    double t0 = steady_seconds();
    int ret = 0;
    if (external_device_input) {
      ret = face_rknn_parallel_infer_merged_bound_external(
          rknn_.get(),
          meta_.num_slices,
          input_h_,
          input_w_,
          input_c_,
          slice_shapes_.data(),
          gains_.data(),
          pads_.data(),
          starts_.data(),
          wraps_.data(),
          meta_.num_slices,
          original_width,
          cfg_.overlap_ratio,
          cfg_.merge_iou_threshold,
          cfg_.nms_iou_threshold,
          cfg_.conf_threshold,
          cfg_.decode_iou_threshold,
          cfg_.max_det,
          cfg_.max_nms,
          1,
          result.detections.data(),
          cfg_.max_output_dets,
          &result.detection_count,
          result.merge_stats,
          result.rknn_timings,
          err,
          sizeof(err));
    } else {
      ret = face_rknn_parallel_infer_merged_bound(
          rknn_.get(),
          meta_.num_slices,
          input_h_,
          input_w_,
          input_c_,
          slice_shapes_.data(),
          gains_.data(),
          pads_.data(),
          starts_.data(),
          wraps_.data(),
          meta_.num_slices,
          original_width,
          cfg_.overlap_ratio,
          cfg_.merge_iou_threshold,
          cfg_.nms_iou_threshold,
          cfg_.conf_threshold,
          cfg_.decode_iou_threshold,
          cfg_.max_det,
          cfg_.max_nms,
          result.detections.data(),
          cfg_.max_output_dets,
          &result.detection_count,
          result.merge_stats,
          result.rknn_timings,
          err,
          sizeof(err));
    }
    const double t1 = steady_seconds();
    result.profile_ms[kProfileRknnTotalOuter] = (t1 - t0) * 1000.0;
    if (ret != 0) {
      throw std::runtime_error(
          std::string(external_device_input
                          ? "face_rknn_parallel_infer_merged_bound_external failed: "
                          : "face_rknn_parallel_infer_merged_bound failed: ") +
          err);
    }
    return true;
  }

  bool try_process_bound(const Image& image, float original_width, FrameResult& result) {
    if (!prepare_bound_inputs(result)) {
      return false;
    }

    char err[4096] = {0};
    double t0 = steady_seconds();
    int ret = 0;
    if (cfg_.staging_copy_input) {
      ensure_staging_buffers();
      ret = ds_opencl_fused_run_split(
          opencl_.get(),
          image.bgr.data(),
          staging_ptrs_.data(),
          meta_.num_slices,
          result.remap_timings,
          err,
          sizeof(err));
      const double t1 = steady_seconds();
      result.profile_ms[kProfileOpenclRunOuter] = (t1 - t0) * 1000.0;
      if (ret != 0) {
        throw std::runtime_error(
            std::string("ds_opencl_fused_run_split(staging) failed: ") + err);
      }
      copy_staging_to_bound_inputs(result);
      return run_bound_inference(original_width, false, result);
    }

    if (bound_imported_) {
      ret = ds_opencl_fused_run_imported(
          opencl_.get(),
          image.bgr.data(),
          result.remap_timings,
          err,
          sizeof(err));
      if (ret != 0) {
        disable_zero_copy_once(std::string("ds_opencl_fused_run_imported failed: ") + err);
      }
    }
    if (!bound_imported_) {
      ret = ds_opencl_fused_run_split(
          opencl_.get(),
          image.bgr.data(),
          bound_ptrs_,
          meta_.num_slices,
          result.remap_timings,
          err,
          sizeof(err));
    } else {
      ret = 0;
    }
    double t1 = steady_seconds();
    result.profile_ms[kProfileOpenclRunOuter] = (t1 - t0) * 1000.0;
    if (ret != 0) {
      throw std::runtime_error(
          std::string("ds_opencl_fused_run_split(bound) failed: ") + err);
    }

    if (bound_imported_) {
      try {
        return run_bound_inference(original_width, true, result);
      } catch (const std::exception& exc) {
        disable_zero_copy_once(exc.what());
        t0 = steady_seconds();
        err[0] = '\0';
        ret = ds_opencl_fused_run_split(
            opencl_.get(),
            image.bgr.data(),
            bound_ptrs_,
            meta_.num_slices,
            result.remap_timings,
            err,
            sizeof(err));
        t1 = steady_seconds();
        result.profile_ms[kProfileOpenclRunOuter] += (t1 - t0) * 1000.0;
        if (ret != 0) {
          throw std::runtime_error(
              std::string("ds_opencl_fused_run_split(bound) failed after zero-copy fallback: ") +
              err);
        }
      }
    }
    return run_bound_inference(original_width, false, result);
  }

  void ensure_opencl(const Image& image) {
    if (opencl_.get() != nullptr && opencl_width_ == image.width && opencl_height_ == image.height) {
      return;
    }
    opencl_.reset();
    bound_imported_ = false;
    bound_import_failed_ = false;

    char err[4096] = {0};
    const int ret = ds_opencl_fused_create(
        map_x_.data(),
        map_y_.data(),
        roi_rects_.data(),
        image.width,
        image.height,
        image.width * 3,
        meta_.num_slices,
        meta_.roi_h,
        meta_.roi_w,
        meta_.imgsz,
        opencl_.out(),
        err,
        sizeof(err));
    if (ret != 0) {
      throw std::runtime_error(std::string("ds_opencl_fused_create failed: ") + err);
    }
    opencl_width_ = image.width;
    opencl_height_ = image.height;
  }

  Config cfg_;
  MapMeta meta_;
  std::vector<float> map_x_;
  std::vector<float> map_y_;
  std::vector<float> base_map_x_;
  std::vector<float> base_map_y_;
  std::vector<int> roi_rects_;
  std::vector<int> slice_shapes_;
  std::vector<float> gains_;
  std::vector<int> pads_;
  std::vector<float> starts_;
  std::vector<int> wraps_;
  RknnHandle rknn_;
  OpenCLHandle opencl_;
  int opencl_width_ = 0;
  int opencl_height_ = 0;
  int input_h_ = 0;
  int input_w_ = 0;
  int input_c_ = 0;
  int channels_ = 0;
  int anchors_ = 0;
  bool bound_disabled_ = false;
  bool bound_prepared_ = false;
  bool bound_imported_ = false;
  bool bound_import_failed_ = false;
  bool bound_import_warned_ = false;
  bool bound_ptrs_ready_ = false;
  bool bound_warned_ = false;
  int bound_fds_[3] = {-1, -1, -1};
  uint64_t bound_sizes_[3] = {0, 0, 0};
  uint8_t* bound_ptrs_[3] = {nullptr, nullptr, nullptr};
  uint64_t bound_ptr_sizes_[3] = {0, 0, 0};
  std::vector<std::vector<uint8_t>> staging_buffers_;
  std::vector<uint8_t*> staging_ptrs_;
};
