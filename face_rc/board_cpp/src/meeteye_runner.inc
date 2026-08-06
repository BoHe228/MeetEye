static void write_outputs(const Config& cfg,
                          const MeetEyeRuntime& runtime,
                          AngleAndDistanceRuntime& angle_runtime,
                          FaceRecognitionRuntime* face_rec,
                          std::ofstream& jsonl,
                          std::ofstream& debug_jsonl,
                          WebSocketServer* ws_server,
                          BoardWebUiServer* webui_server,
                          AsyncWebUiFramePublisher* webui_frame_publisher,
                          SystemLoadSampler* system_sampler,
                          const Image* source_image,
                          FrameResult& result,
                          const FrameRateStats& fps,
                          int frame_id,
                          bool first_stdout) {
  const double write_start_s = steady_seconds();
  double t0 = steady_seconds();
  std::vector<TargetInfo> targets = angle_runtime.build_targets(result);
  if (face_rec != nullptr) {
    face_rec->annotate_targets(&targets);
  }
  double t1 = steady_seconds();
  result.profile_ms[kProfileAngleTargets] = (t1 - t0) * 1000.0;

  t0 = steady_seconds();
  const std::string payload = build_output_json(cfg, targets, frame_id, result, fps);
  t1 = steady_seconds();
  result.profile_ms[kProfileBuildPayload] = (t1 - t0) * 1000.0;

  if (!cfg.no_stdout_json) {
    t0 = steady_seconds();
    if (!first_stdout) {
      std::cout << "\n";
    }
    if (cfg.stdout_debug_json) {
      print_result_json(std::cout, runtime.meta(), runtime.channels(), runtime.anchors(), result, fps);
    } else {
      std::cout << payload << "\n";
    }
    t1 = steady_seconds();
    result.profile_ms[kProfileStdout] = (t1 - t0) * 1000.0;
  }
  if (jsonl) {
    t0 = steady_seconds();
    jsonl << payload << "\n";
    t1 = steady_seconds();
    result.profile_ms[kProfileJsonl] = (t1 - t0) * 1000.0;
  }
  if (ws_server != nullptr) {
    t0 = steady_seconds();
    ws_server->publish(payload);
    t1 = steady_seconds();
    result.profile_ms[kProfileWebSocket] = (t1 - t0) * 1000.0;
  }
  if (webui_server != nullptr) {
    t0 = steady_seconds();
    webui_server->publish_json(payload);
    if (system_sampler != nullptr) {
      webui_server->publish_system(system_sampler->latest_json());
    }
    if (webui_frame_publisher != nullptr && source_image != nullptr) {
      webui_frame_publisher->submit(*source_image, result, targets, fps);
    }
    t1 = steady_seconds();
    result.profile_ms[kProfileWebSocket] += (t1 - t0) * 1000.0;
  }
  if (debug_jsonl) {
    t0 = steady_seconds();
    print_result_jsonl(debug_jsonl,
                       runtime.meta(),
                       runtime.channels(),
                       runtime.anchors(),
                       result,
                       fps,
                       cfg.face_rec_enabled ? &targets : nullptr);
    t1 = steady_seconds();
    result.profile_ms[kProfileDebugJsonl] = (t1 - t0) * 1000.0;
  }
  result.profile_ms[kProfileWriteOutputs] = (steady_seconds() - write_start_s) * 1000.0;
}

static void finish_and_write_frame(const Config& cfg,
                                   const MeetEyeRuntime& runtime,
                                   AngleAndDistanceRuntime& angle_runtime,
                                   NativeHybridSortTracker* tracker,
                                   FaceRecognitionRuntime* face_rec,
                                   std::ofstream& jsonl,
                                   std::ofstream& debug_jsonl,
                                   WebSocketServer* ws_server,
                                   BoardWebUiServer* webui_server,
                                   AsyncWebUiFramePublisher* webui_frame_publisher,
                                   SystemLoadSampler* system_sampler,
                                   PreparedStagingFrame& frame,
                                   int& frame_id,
                                   double run_start_s,
                                   ProfileSummary& profile_summary,
                                   bool& first_stdout) {
  FrameResult& result = frame.result;
  result.profile_ms[kProfileFileRead] = frame.file_read_ms;
  result.profile_ms[kProfileCameraRead] = frame.camera_read_ms;
  result.profile_ms[kProfileJpegDecode] = frame.jpeg_decode_ms;
  result.profile_ms[kProfileDecodeWait] = frame.decode_wait_ms;
  result.profile_ms[kProfileDecode] =
      frame.file_read_ms + frame.camera_read_ms + frame.jpeg_decode_ms;

  double t0 = steady_seconds();
  if (tracker != nullptr) {
    tracker->update(result);
    const double t1 = steady_seconds();
    result.profile_ms[kProfileTrackerOuter] = (t1 - t0) * 1000.0;
  } else {
    result.raw_detection_count = result.detection_count;
    result.track_ids.assign(static_cast<size_t>(result.detection_count), -1);
  }

  frame_id += 1;

  if (face_rec != nullptr) {
    const double t_face0 = steady_seconds();
    face_rec->process_frame(frame.image, result, frame_id);
    const double t_face1 = steady_seconds();
    result.profile_ms[kProfileFaceRec] = (t_face1 - t_face0) * 1000.0;
  }

  const double now_s = steady_seconds();
  FrameRateStats fps;
  fps.frame_ms = (now_s - frame.frame_start_s) * 1000.0;
  fps.instant_fps = fps.frame_ms > 0.0 ? 1000.0 / fps.frame_ms : 0.0;
  fps.elapsed_s = std::max(0.0, now_s - run_start_s);
  fps.frames = frame_id;
  fps.average_fps = fps.elapsed_s > 0.0 ? static_cast<double>(fps.frames) / fps.elapsed_s : 0.0;
  result.profile_ms[kProfileFrameTotal] = fps.frame_ms;

  if (cfg.print_profile_summary) {
    profile_summary.add(result, fps);
    if (cfg.profile_interval > 0 && frame_id % cfg.profile_interval == 0) {
      profile_summary.print(std::cerr, "running");
    }
  }

  write_outputs(
      cfg,
      runtime,
      angle_runtime,
      face_rec,
      jsonl,
      debug_jsonl,
      ws_server,
      webui_server,
      webui_frame_publisher,
      system_sampler,
      &frame.image,
      result,
      fps,
      frame_id,
      first_stdout);
  first_stdout = false;
}

static int run(Config cfg) {
  if (cfg.output_jsonl_path.empty() && !cfg.no_output_jsonl) {
    cfg.output_jsonl_path = default_output_jsonl_path();
  }

  MeetEyeRuntime runtime(cfg);
  cfg.projection_mode = runtime.meta().projection_mode;
  apply_projection_mode_defaults(cfg);
  AngleAndDistanceRuntime angle_runtime(cfg, runtime.meta(), runtime.base_map_x(), runtime.base_map_y());
  std::unique_ptr<NativeHybridSortTracker> tracker;
  if (cfg.tracker_enabled) {
    tracker.reset(new NativeHybridSortTracker(cfg, runtime.meta()));
  }
  std::unique_ptr<FaceRecognitionRuntime> face_rec;
  if (cfg.face_rec_enabled) {
    std::function<bool(const Image&, std::array<float, kDetectionFields>*, std::string*)>
        known_photo_detector =
            [&runtime](const Image& image,
                       std::array<float, kDetectionFields>* det,
                       std::string* error) {
              return runtime.detect_known_face_photo(image, det, error);
            };
    face_rec.reset(new FaceRecognitionRuntime(
        cfg, runtime.meta(), runtime.base_map_x(), runtime.base_map_y(), known_photo_detector));
  }

  std::ofstream jsonl;
  if (!cfg.output_jsonl_path.empty()) {
    jsonl.open(cfg.output_jsonl_path.c_str(), std::ios::out | std::ios::trunc);
    if (!jsonl) {
      throw std::runtime_error("cannot open output jsonl: " + cfg.output_jsonl_path);
    }
  }
  std::ofstream debug_jsonl;
  if (!cfg.debug_jsonl_path.empty()) {
    debug_jsonl.open(cfg.debug_jsonl_path.c_str(), std::ios::out | std::ios::trunc);
    if (!debug_jsonl) {
      throw std::runtime_error("cannot open debug jsonl: " + cfg.debug_jsonl_path);
    }
  }
  std::unique_ptr<WebSocketServer> ws_server;
  if (cfg.ws_json) {
    ws_server.reset(new WebSocketServer(cfg.ws_host, cfg.ws_port, cfg.ws_path));
    ws_server->start();
  }
  std::unique_ptr<BoardWebUiServer> webui_server;
  std::unique_ptr<AsyncWebUiFramePublisher> webui_frame_publisher;
  if (cfg.webui) {
    webui_server.reset(new BoardWebUiServer(cfg.webui_host, cfg.webui_port));
    webui_server->start();
    webui_frame_publisher.reset(
        new AsyncWebUiFramePublisher(cfg, runtime, webui_server.get()));
    webui_frame_publisher->start();
  }

  std::string system_profile_path;
  std::unique_ptr<SystemLoadSampler> system_sampler;
  if (cfg.profile_system_load || cfg.webui) {
    if (cfg.profile_system_load) {
      system_profile_path = system_profile_path_for(cfg);
    }
    system_sampler.reset(new SystemLoadSampler(cfg.system_load_interval_ms, system_profile_path));
    system_sampler->start();
    std::cerr << "[system-load] background sampler enabled: interval="
              << cfg.system_load_interval_ms << "ms\n";
    if (!system_profile_path.empty()) {
      std::cerr << "[system-load] profile JSONL: " << system_profile_path << "\n";
    }
  }

  bool first_stdout = true;
  int frame_id = 0;
  const double run_start_s = steady_seconds();
  ProfileSummary profile_summary;
  TurboJpegDecoder jpeg_decoder;
  std::unique_ptr<ImageDecodePrefetcher> image_prefetcher;
  if (cfg.decode_prefetch && !cfg.image_paths.empty()) {
    image_prefetcher.reset(new ImageDecodePrefetcher(cfg.image_paths));
    std::cerr << "[decode-prefetch] image-list decode pipeline enabled: queue=3\n";
  }
  if (cfg.staging_pipeline && runtime.supports_staging_pipeline() && !cfg.image_paths.empty()) {
    auto load_prepared_image = [&](size_t i) -> PreparedStagingFrame {
      PreparedStagingFrame frame;
      frame.frame_start_s = steady_seconds();
      double t0 = steady_seconds();
      double t1 = t0;
      if (image_prefetcher) {
        t0 = steady_seconds();
        DecodedFrame decoded = image_prefetcher->pop();
        t1 = steady_seconds();
        frame.decode_wait_ms = (t1 - t0) * 1000.0;
        frame.image_path = decoded.image_path;
        frame.image = std::move(decoded.image);
        frame.file_read_ms = decoded.file_read_ms;
        frame.jpeg_decode_ms = decoded.jpeg_decode_ms;
        frame.frame_index = static_cast<int>(i);
      } else {
        frame.image_path = cfg.image_paths[i];
        frame.frame_index = static_cast<int>(i);
        t0 = steady_seconds();
        const std::vector<uint8_t> jpeg = read_bytes(frame.image_path);
        t1 = steady_seconds();
        frame.file_read_ms = (t1 - t0) * 1000.0;
        t0 = steady_seconds();
        frame.image = jpeg_decoder.decode_buffer(jpeg.data(), jpeg.size());
        t1 = steady_seconds();
        frame.jpeg_decode_ms = (t1 - t0) * 1000.0;
      }
      return runtime.prepare_staging_frame(std::move(frame));
    };

    PreparedStagingFrame pending = load_prepared_image(0);
    for (size_t i = 1; i < cfg.image_paths.size(); ++i) {
      std::future<PreparedStagingFrame> current_future =
          runtime.start_staging_inference(std::move(pending));
      PreparedStagingFrame next = load_prepared_image(i);
      PreparedStagingFrame done = current_future.get();
      finish_and_write_frame(
          cfg,
          runtime,
          angle_runtime,
          tracker.get(),
          face_rec.get(),
          jsonl,
          debug_jsonl,
          ws_server.get(),
          webui_server.get(),
          webui_frame_publisher.get(),
          system_sampler.get(),
          done,
          frame_id,
          run_start_s,
          profile_summary,
          first_stdout);
      pending = std::move(next);
    }
    std::future<PreparedStagingFrame> current_future =
        runtime.start_staging_inference(std::move(pending));
    PreparedStagingFrame done = current_future.get();
    finish_and_write_frame(
        cfg,
        runtime,
        angle_runtime,
        tracker.get(),
        face_rec.get(),
        jsonl,
        debug_jsonl,
        ws_server.get(),
        webui_server.get(),
        webui_frame_publisher.get(),
        system_sampler.get(),
        done,
        frame_id,
        run_start_s,
        profile_summary,
        first_stdout);
  } else {
    if (runtime.supports_wide_angle_yolo_pipeline() && cfg.image_paths.size() > 1) {
      struct WideAngleInFlightFrame {
        int worker_index = 0;
        std::future<PreparedStagingFrame> future;
      };
      const int worker_count = std::min(
          runtime.wide_angle_yolo_worker_count(),
          static_cast<int>(cfg.image_paths.size()));
      std::cerr << "[wide-angle-yolo-pipeline] image-list YOLO workers="
                << worker_count << " core_masks=1,2";
      if (face_rec) {
        std::cerr << " face_rec_core_mask=4";
      }
      std::cerr << "\n";

      auto load_wide_angle_frame = [&](size_t i) -> PreparedStagingFrame {
        PreparedStagingFrame frame;
        frame.frame_start_s = steady_seconds();
        double t0 = steady_seconds();
        double t1 = t0;
        if (image_prefetcher) {
          t0 = steady_seconds();
          DecodedFrame decoded = image_prefetcher->pop();
          t1 = steady_seconds();
          frame.decode_wait_ms = (t1 - t0) * 1000.0;
          frame.image_path = decoded.image_path;
          frame.image = std::move(decoded.image);
          frame.file_read_ms = decoded.file_read_ms;
          frame.jpeg_decode_ms = decoded.jpeg_decode_ms;
          frame.frame_index = static_cast<int>(decoded.index);
        } else {
          frame.image_path = cfg.image_paths[i];
          frame.frame_index = static_cast<int>(i);
          t0 = steady_seconds();
          const std::vector<uint8_t> jpeg = read_bytes(frame.image_path);
          t1 = steady_seconds();
          frame.file_read_ms = (t1 - t0) * 1000.0;
          t0 = steady_seconds();
          frame.image = jpeg_decoder.decode_buffer(jpeg.data(), jpeg.size());
          t1 = steady_seconds();
          frame.jpeg_decode_ms = (t1 - t0) * 1000.0;
        }
        return runtime.prepare_wide_angle_frame(std::move(frame));
      };

      size_t next_image_idx = 0;
      std::deque<WideAngleInFlightFrame> in_flight;
      auto submit_next_wide_angle = [&](int worker_index) {
        if (next_image_idx >= cfg.image_paths.size()) {
          return;
        }
        PreparedStagingFrame prepared = load_wide_angle_frame(next_image_idx);
        next_image_idx += 1;
        WideAngleInFlightFrame item;
        item.worker_index = worker_index;
        item.future = runtime.start_wide_angle_inference(std::move(prepared), worker_index);
        in_flight.push_back(std::move(item));
      };

      for (int worker = 0; worker < worker_count; ++worker) {
        submit_next_wide_angle(worker);
      }
      while (!in_flight.empty()) {
        WideAngleInFlightFrame current = std::move(in_flight.front());
        in_flight.pop_front();
        PreparedStagingFrame done = current.future.get();
        submit_next_wide_angle(current.worker_index);
        finish_and_write_frame(
            cfg,
            runtime,
            angle_runtime,
            tracker.get(),
            face_rec.get(),
            jsonl,
            debug_jsonl,
            ws_server.get(),
            webui_server.get(),
            webui_frame_publisher.get(),
            system_sampler.get(),
            done,
            frame_id,
            run_start_s,
            profile_summary,
            first_stdout);
      }
    } else {
      for (size_t i = 0; i < cfg.image_paths.size(); ++i) {
        const double frame_start_s = steady_seconds();
        std::string image_path;
        Image image;
        double file_read_ms = 0.0;
        double jpeg_decode_ms = 0.0;
        double decode_wait_ms = 0.0;
        double t0 = steady_seconds();
        double t1 = t0;
        if (image_prefetcher) {
          t0 = steady_seconds();
          DecodedFrame decoded = image_prefetcher->pop();
          t1 = steady_seconds();
          decode_wait_ms = (t1 - t0) * 1000.0;
          image_path = decoded.image_path;
          image = std::move(decoded.image);
          file_read_ms = decoded.file_read_ms;
          jpeg_decode_ms = decoded.jpeg_decode_ms;
        } else {
          image_path = cfg.image_paths[i];
          t0 = steady_seconds();
          const std::vector<uint8_t> jpeg = read_bytes(image_path);
          t1 = steady_seconds();
          file_read_ms = (t1 - t0) * 1000.0;
          t0 = steady_seconds();
          image = jpeg_decoder.decode_buffer(jpeg.data(), jpeg.size());
          t1 = steady_seconds();
          jpeg_decode_ms = (t1 - t0) * 1000.0;
        }
        FrameResult result = runtime.process(image, image_path, static_cast<int>(i));
        result.profile_ms[kProfileFileRead] = file_read_ms;
        result.profile_ms[kProfileJpegDecode] = jpeg_decode_ms;
        result.profile_ms[kProfileDecodeWait] = decode_wait_ms;
        result.profile_ms[kProfileDecode] = file_read_ms + jpeg_decode_ms;
        if (tracker) {
          t0 = steady_seconds();
          tracker->update(result);
          t1 = steady_seconds();
          result.profile_ms[kProfileTrackerOuter] = (t1 - t0) * 1000.0;
        } else {
          result.raw_detection_count = result.detection_count;
          result.track_ids.assign(static_cast<size_t>(result.detection_count), -1);
        }

        frame_id += 1;
        if (face_rec) {
          t0 = steady_seconds();
          face_rec->process_frame(image, result, frame_id);
          t1 = steady_seconds();
          result.profile_ms[kProfileFaceRec] = (t1 - t0) * 1000.0;
        }
        const double now_s = steady_seconds();
        FrameRateStats fps;
        fps.frame_ms = (now_s - frame_start_s) * 1000.0;
        fps.instant_fps = fps.frame_ms > 0.0 ? 1000.0 / fps.frame_ms : 0.0;
        fps.elapsed_s = std::max(0.0, now_s - run_start_s);
        fps.frames = frame_id;
        fps.average_fps =
            fps.elapsed_s > 0.0 ? static_cast<double>(fps.frames) / fps.elapsed_s : 0.0;
        result.profile_ms[kProfileFrameTotal] = fps.frame_ms;
        if (cfg.print_profile_summary) {
          profile_summary.add(result, fps);
          if (cfg.profile_interval > 0 && frame_id % cfg.profile_interval == 0) {
            profile_summary.print(std::cerr, "running");
          }
        }
        write_outputs(
            cfg,
            runtime,
            angle_runtime,
            face_rec.get(),
            jsonl,
            debug_jsonl,
            ws_server.get(),
            webui_server.get(),
            webui_frame_publisher.get(),
            system_sampler.get(),
            &image,
            result,
            fps,
            frame_id,
            first_stdout);
        first_stdout = false;
      }
    }
  }

  if (!cfg.camera_device.empty()) {
    if (cfg.camera_prefetch) {
      CameraDecodePrefetcher camera_prefetcher(cfg);
      if (runtime.supports_wide_angle_yolo_pipeline()) {
        struct WideAngleInFlightFrame {
          int worker_index = 0;
          std::future<PreparedStagingFrame> future;
        };
        int worker_count = runtime.wide_angle_yolo_worker_count();
        if (cfg.max_frames > 0) {
          worker_count = std::min(worker_count, cfg.max_frames);
        }
        std::cerr << "[wide-angle-yolo-pipeline] camera YOLO workers="
                  << worker_count << " core_masks=1,2";
        if (face_rec) {
          std::cerr << " face_rec_core_mask=4";
        }
        std::cerr << "\n";

        auto load_wide_angle_camera = [&]() -> PreparedStagingFrame {
          PreparedStagingFrame frame;
          frame.frame_start_s = steady_seconds();
          const double t0 = steady_seconds();
          DecodedFrame decoded = camera_prefetcher.pop();
          const double t1 = steady_seconds();
          frame.decode_wait_ms = (t1 - t0) * 1000.0;
          frame.image_path = decoded.image_path;
          frame.image = std::move(decoded.image);
          frame.frame_index = static_cast<int>(decoded.index);
          frame.camera_read_ms = decoded.camera_read_ms;
          frame.jpeg_decode_ms = decoded.jpeg_decode_ms;
          return runtime.prepare_wide_angle_frame(std::move(frame));
        };

        int submitted_frames = 0;
        std::deque<WideAngleInFlightFrame> in_flight;
        auto can_submit_next = [&]() -> bool {
          return cfg.max_frames == 0 || submitted_frames < cfg.max_frames;
        };
        auto submit_next_wide_angle_camera = [&](int worker_index) {
          if (!can_submit_next()) {
            return;
          }
          PreparedStagingFrame prepared = load_wide_angle_camera();
          submitted_frames += 1;
          WideAngleInFlightFrame item;
          item.worker_index = worker_index;
          item.future =
              runtime.start_wide_angle_inference(std::move(prepared), worker_index);
          in_flight.push_back(std::move(item));
        };

        for (int worker = 0; worker < worker_count; ++worker) {
          submit_next_wide_angle_camera(worker);
        }
        while (!in_flight.empty()) {
          WideAngleInFlightFrame current = std::move(in_flight.front());
          in_flight.pop_front();
          PreparedStagingFrame done = current.future.get();
          submit_next_wide_angle_camera(current.worker_index);
          finish_and_write_frame(
              cfg,
              runtime,
              angle_runtime,
              tracker.get(),
              face_rec.get(),
              jsonl,
              debug_jsonl,
              ws_server.get(),
              webui_server.get(),
              webui_frame_publisher.get(),
              system_sampler.get(),
              done,
              frame_id,
              run_start_s,
              profile_summary,
              first_stdout);
        }
      } else if (cfg.staging_pipeline && runtime.supports_staging_pipeline()) {
        auto load_prepared_camera = [&]() -> PreparedStagingFrame {
          PreparedStagingFrame frame;
          frame.frame_start_s = steady_seconds();
          const double t0 = steady_seconds();
          DecodedFrame decoded = camera_prefetcher.pop();
          const double t1 = steady_seconds();
          frame.decode_wait_ms = (t1 - t0) * 1000.0;
          frame.image_path = decoded.image_path;
          frame.image = std::move(decoded.image);
          frame.frame_index = static_cast<int>(decoded.index);
          frame.camera_read_ms = decoded.camera_read_ms;
          frame.jpeg_decode_ms = decoded.jpeg_decode_ms;
          return runtime.prepare_staging_frame(std::move(frame));
        };

        PreparedStagingFrame pending = load_prepared_camera();
        for (int i = 1; cfg.max_frames == 0 || i < cfg.max_frames; ++i) {
          std::future<PreparedStagingFrame> current_future =
              runtime.start_staging_inference(std::move(pending));
          PreparedStagingFrame next = load_prepared_camera();
          PreparedStagingFrame done = current_future.get();
          finish_and_write_frame(
              cfg,
              runtime,
              angle_runtime,
              tracker.get(),
              face_rec.get(),
              jsonl,
              debug_jsonl,
              ws_server.get(),
              webui_server.get(),
              webui_frame_publisher.get(),
              system_sampler.get(),
              done,
              frame_id,
              run_start_s,
              profile_summary,
              first_stdout);
          pending = std::move(next);
        }
        if (cfg.max_frames > 0) {
          std::future<PreparedStagingFrame> current_future =
              runtime.start_staging_inference(std::move(pending));
          PreparedStagingFrame done = current_future.get();
          finish_and_write_frame(
              cfg,
              runtime,
              angle_runtime,
              tracker.get(),
              face_rec.get(),
              jsonl,
              debug_jsonl,
              ws_server.get(),
              webui_server.get(),
              webui_frame_publisher.get(),
              system_sampler.get(),
              done,
              frame_id,
              run_start_s,
              profile_summary,
              first_stdout);
        }
      } else {
      for (int i = 0; cfg.max_frames == 0 || i < cfg.max_frames; ++i) {
        const double frame_start_s = steady_seconds();
        double t0 = steady_seconds();
        DecodedFrame decoded = camera_prefetcher.pop();
        double t1 = steady_seconds();
        const double decode_wait_ms = (t1 - t0) * 1000.0;

        FrameResult result =
            runtime.process(decoded.image, decoded.image_path, static_cast<int>(decoded.index));
        result.profile_ms[kProfileCameraRead] = decoded.camera_read_ms;
        result.profile_ms[kProfileJpegDecode] = decoded.jpeg_decode_ms;
        result.profile_ms[kProfileDecodeWait] = decode_wait_ms;
        result.profile_ms[kProfileDecode] = decoded.camera_read_ms + decoded.jpeg_decode_ms;
        if (tracker) {
          t0 = steady_seconds();
          tracker->update(result);
          t1 = steady_seconds();
          result.profile_ms[kProfileTrackerOuter] = (t1 - t0) * 1000.0;
        } else {
          result.raw_detection_count = result.detection_count;
          result.track_ids.assign(static_cast<size_t>(result.detection_count), -1);
        }

        frame_id += 1;
        if (face_rec) {
          t0 = steady_seconds();
          face_rec->process_frame(decoded.image, result, frame_id);
          t1 = steady_seconds();
          result.profile_ms[kProfileFaceRec] = (t1 - t0) * 1000.0;
        }
        const double now_s = steady_seconds();
        FrameRateStats fps;
        fps.frame_ms = (now_s - frame_start_s) * 1000.0;
        fps.instant_fps = fps.frame_ms > 0.0 ? 1000.0 / fps.frame_ms : 0.0;
        fps.elapsed_s = std::max(0.0, now_s - run_start_s);
        fps.frames = frame_id;
        fps.average_fps =
            fps.elapsed_s > 0.0 ? static_cast<double>(fps.frames) / fps.elapsed_s : 0.0;
        result.profile_ms[kProfileFrameTotal] = fps.frame_ms;
        if (cfg.print_profile_summary) {
          profile_summary.add(result, fps);
          if (cfg.profile_interval > 0 && frame_id % cfg.profile_interval == 0) {
            profile_summary.print(std::cerr, "running");
          }
        }
        write_outputs(
            cfg,
            runtime,
            angle_runtime,
            face_rec.get(),
            jsonl,
            debug_jsonl,
            ws_server.get(),
            webui_server.get(),
            webui_frame_publisher.get(),
            system_sampler.get(),
            &decoded.image,
            result,
            fps,
            frame_id,
            first_stdout);
        first_stdout = false;
      }
      }
    } else {
      V4L2MjpegCamera camera(
          cfg.camera_device,
          cfg.camera_width,
          cfg.camera_height,
          cfg.camera_fps,
          cfg.camera_buffers,
          cfg.camera_timeout_ms);
      camera.open();

      for (int i = 0; cfg.max_frames == 0 || i < cfg.max_frames; ++i) {
        const double frame_start_s = steady_seconds();
        double t0 = steady_seconds();
        V4L2MjpegCamera::Frame frame = camera.read_frame();
        double t1 = steady_seconds();
        const double camera_read_ms = (t1 - t0) * 1000.0;
        Image image;
        double jpeg_decode_ms = 0.0;
        try {
          t0 = steady_seconds();
          image = jpeg_decoder.decode_buffer(frame.data, frame.size);
          t1 = steady_seconds();
          jpeg_decode_ms = (t1 - t0) * 1000.0;
        } catch (...) {
          camera.release_frame(frame);
          throw;
        }
        camera.release_frame(frame);

        std::ostringstream frame_name;
        frame_name << cfg.camera_device << "#" << i;
        FrameResult result = runtime.process(image, frame_name.str(), i);
        result.profile_ms[kProfileCameraRead] = camera_read_ms;
        result.profile_ms[kProfileJpegDecode] = jpeg_decode_ms;
        result.profile_ms[kProfileDecode] = camera_read_ms + jpeg_decode_ms;
        if (tracker) {
          t0 = steady_seconds();
          tracker->update(result);
          t1 = steady_seconds();
          result.profile_ms[kProfileTrackerOuter] = (t1 - t0) * 1000.0;
        } else {
          result.raw_detection_count = result.detection_count;
          result.track_ids.assign(static_cast<size_t>(result.detection_count), -1);
        }

        frame_id += 1;
        if (face_rec) {
          t0 = steady_seconds();
          face_rec->process_frame(image, result, frame_id);
          t1 = steady_seconds();
          result.profile_ms[kProfileFaceRec] = (t1 - t0) * 1000.0;
        }
        const double now_s = steady_seconds();
        FrameRateStats fps;
        fps.frame_ms = (now_s - frame_start_s) * 1000.0;
        fps.instant_fps = fps.frame_ms > 0.0 ? 1000.0 / fps.frame_ms : 0.0;
        fps.elapsed_s = std::max(0.0, now_s - run_start_s);
        fps.frames = frame_id;
        fps.average_fps =
            fps.elapsed_s > 0.0 ? static_cast<double>(fps.frames) / fps.elapsed_s : 0.0;
        result.profile_ms[kProfileFrameTotal] = fps.frame_ms;
        if (cfg.print_profile_summary) {
          profile_summary.add(result, fps);
          if (cfg.profile_interval > 0 && frame_id % cfg.profile_interval == 0) {
            profile_summary.print(std::cerr, "running");
          }
        }
        write_outputs(
            cfg,
            runtime,
            angle_runtime,
            face_rec.get(),
            jsonl,
            debug_jsonl,
            ws_server.get(),
            webui_server.get(),
            webui_frame_publisher.get(),
            system_sampler.get(),
            &image,
            result,
            fps,
            frame_id,
            first_stdout);
        first_stdout = false;
      }
    }
  }

  if (cfg.print_profile_summary) {
    profile_summary.print(std::cerr, "final");
  }
  if (system_sampler) {
    system_sampler->stop();
    const double elapsed_s = std::max(0.0, steady_seconds() - run_start_s);
    write_system_load_summary(
        system_profile_path,
        !cfg.debug_jsonl_path.empty() ? cfg.debug_jsonl_path : cfg.output_jsonl_path,
        elapsed_s,
        frame_id);
  }

  return 0;
}
