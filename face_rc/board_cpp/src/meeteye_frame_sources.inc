static std::vector<uint8_t> read_bytes(const std::string& path) {
  std::ifstream f(path.c_str(), std::ios::binary);
  if (!f) {
    throw std::runtime_error("cannot open file: " + path);
  }
  f.seekg(0, std::ios::end);
  const std::streamoff size = f.tellg();
  if (size <= 0) {
    throw std::runtime_error("empty file: " + path);
  }
  f.seekg(0, std::ios::beg);
  std::vector<uint8_t> data(static_cast<size_t>(size));
  f.read(reinterpret_cast<char*>(data.data()), size);
  if (!f) {
    throw std::runtime_error("failed to read file: " + path);
  }
  return data;
}

static std::vector<float> read_floats(const std::string& path) {
  const std::vector<uint8_t> bytes = read_bytes(path);
  if (bytes.size() % sizeof(float) != 0) {
    throw std::runtime_error("float file size is not aligned: " + path);
  }
  std::vector<float> values(bytes.size() / sizeof(float));
  std::memcpy(values.data(), bytes.data(), bytes.size());
  return values;
}

static std::vector<std::string> read_image_list(const std::string& path) {
  std::ifstream f(path.c_str());
  if (!f) {
    throw std::runtime_error("cannot open image list: " + path);
  }
  std::vector<std::string> paths;
  std::string line;
  while (std::getline(f, line)) {
    line = trim(line);
    if (line.empty() || line[0] == '#') {
      continue;
    }
    paths.push_back(line);
  }
  return paths;
}

static int xioctl(int fd, unsigned long request, void* arg) {
  int ret = 0;
  do {
    ret = ioctl(fd, request, arg);
  } while (ret == -1 && errno == EINTR);
  return ret;
}

static int get_int(const std::map<std::string, std::string>& meta,
                   const std::string& key,
                   int default_value = 0) {
  const auto it = meta.find(key);
  if (it == meta.end() || it->second.empty()) {
    return default_value;
  }
  return std::stoi(it->second);
}

static float get_float(const std::map<std::string, std::string>& meta,
                       const std::string& key,
                       float default_value = 0.0f) {
  const auto it = meta.find(key);
  if (it == meta.end() || it->second.empty()) {
    return default_value;
  }
  return std::stof(it->second);
}

static double get_double(const std::map<std::string, std::string>& meta,
                         const std::string& key,
                         double default_value = 0.0) {
  const auto it = meta.find(key);
  if (it == meta.end() || it->second.empty()) {
    return default_value;
  }
  return std::stod(it->second);
}

static std::string get_string(const std::map<std::string, std::string>& meta,
                              const std::string& key,
                              const std::string& default_value = std::string()) {
  const auto it = meta.find(key);
  if (it == meta.end() || it->second.empty()) {
    return default_value;
  }
  return it->second;
}

static std::map<std::string, std::string> load_meta_kv(const std::string& path) {
  std::ifstream f(path.c_str());
  if (!f) {
    throw std::runtime_error("cannot open meta file: " + path);
  }
  std::map<std::string, std::string> meta;
  std::string line;
  while (std::getline(f, line)) {
    line = trim(line);
    if (line.empty() || line[0] == '#') {
      continue;
    }
    const size_t eq = line.find('=');
    if (eq == std::string::npos) {
      continue;
    }
    meta[trim(line.substr(0, eq))] = trim(line.substr(eq + 1));
  }
  return meta;
}

static MapMeta load_map_meta(const std::string& map_dir) {
  const auto kv = load_meta_kv(join_path(map_dir, "meta.txt"));
  MapMeta meta;
  meta.projection_mode =
      normalize_projection_mode(get_string(kv, "projection_mode", meta.projection_mode));
  meta.num_slices = get_int(kv, "num_slices");
  meta.roi_h = get_int(kv, "roi_h");
  meta.roi_w = get_int(kv, "roi_w");
  meta.imgsz = get_int(kv, "imgsz");
  meta.process_width = get_int(kv, "process_width");
  meta.process_height = get_int(kv, "process_height");
  meta.img_width = get_int(kv, "img_width");
  meta.img_height = get_int(kv, "img_height");
  meta.center_x = get_int(kv, "center_x");
  meta.center_y = get_int(kv, "center_y");
  meta.radius = get_int(kv, "radius");
  meta.base_map_width = get_int(kv, "base_map_width");
  meta.base_map_height = get_int(kv, "base_map_height");
  meta.wide_fx = get_double(kv, "wide_fx", meta.wide_fx);
  meta.wide_fy = get_double(kv, "wide_fy", meta.wide_fy);
  meta.wide_cx = get_double(kv, "wide_cx", meta.wide_cx);
  meta.wide_cy = get_double(kv, "wide_cy", meta.wide_cy);

  if (meta.num_slices <= 0 || meta.roi_h <= 0 || meta.roi_w <= 0 || meta.imgsz <= 0) {
    throw std::runtime_error("invalid map meta shape in " + join_path(map_dir, "meta.txt"));
  }
  if (is_wide_angle_projection(meta.projection_mode) &&
      (meta.wide_fx <= 0.0 || meta.wide_fy <= 0.0 ||
       meta.process_width <= 0 || meta.process_height <= 0)) {
    throw std::runtime_error(
        "wide-angle map meta requires wide_fx/wide_fy/process_width/process_height in " +
        join_path(map_dir, "meta.txt"));
  }
  meta.slices.resize(static_cast<size_t>(meta.num_slices));
  for (int i = 0; i < meta.num_slices; ++i) {
    const std::string prefix = "slice" + std::to_string(i) + ".";
    SliceMeta& s = meta.slices[static_cast<size_t>(i)];
    s.start_x = get_int(kv, prefix + "start_x");
    s.slice_width = get_int(kv, prefix + "slice_width");
    s.slice_height = get_int(kv, prefix + "slice_height");
    s.original_width = get_int(kv, prefix + "original_width");
    s.original_height = get_int(kv, prefix + "original_height");
    s.wrap_around = get_int(kv, prefix + "wrap_around");
    s.gain = get_float(kv, prefix + "gain", 1.0f);
    s.left = get_int(kv, prefix + "left");
    s.top = get_int(kv, prefix + "top");
    s.new_width = get_int(kv, prefix + "new_width", meta.imgsz);
    s.new_height = get_int(kv, prefix + "new_height", meta.imgsz);
    if (s.slice_width <= 0 || s.slice_height <= 0 || s.gain <= 0.0f) {
      throw std::runtime_error("invalid slice meta: " + prefix);
    }
  }
  return meta;
}

class TurboJpegDecoder {
 public:
  TurboJpegDecoder() {
    handle_ = tjInitDecompress();
    if (!handle_) {
      throw std::runtime_error("tjInitDecompress failed");
    }
  }

  ~TurboJpegDecoder() {
    if (handle_) {
      tjDestroy(handle_);
    }
  }

  Image decode_buffer(const uint8_t* jpeg_data, size_t jpeg_size) {
    Image image;
    int subsamp = 0;
    int colorspace = 0;
    if (tjDecompressHeader3(
            handle_,
            jpeg_data,
            static_cast<unsigned long>(jpeg_size),
            &image.width,
            &image.height,
            &subsamp,
            &colorspace) != 0) {
      throw std::runtime_error("tjDecompressHeader3 failed: " + error_string());
    }
    if (image.width <= 0 || image.height <= 0) {
      throw std::runtime_error("invalid jpeg shape");
    }
    image.bgr.resize(static_cast<size_t>(image.width) * image.height * 3);
    if (tjDecompress2(
            handle_,
            jpeg_data,
            static_cast<unsigned long>(jpeg_size),
            image.bgr.data(),
            image.width,
            0,
            image.height,
            kTurboJpegPixelFormatBgr,
            0) != 0) {
      throw std::runtime_error("tjDecompress2 failed: " + error_string());
    }
    return image;
  }

 private:
  std::string error_string() const {
    const char* err = tjGetErrorStr2(handle_);
    return err ? std::string(err) : std::string("unknown");
  }

  tjhandle handle_ = nullptr;
};

class TurboJpegEncoder {
 public:
  TurboJpegEncoder() {
    handle_ = tjInitCompress();
    if (!handle_) {
      throw std::runtime_error("tjInitCompress failed");
    }
  }

  ~TurboJpegEncoder() {
    if (handle_) {
      tjDestroy(handle_);
    }
  }

  std::vector<uint8_t> encode_bgr(const Image& image, int quality) {
    if (image.width <= 0 || image.height <= 0 ||
        image.bgr.size() != static_cast<size_t>(image.width) * image.height * 3) {
      throw std::runtime_error("invalid image for jpeg encode");
    }
    quality = std::max(1, std::min(100, quality));
    unsigned char* jpeg_buf = nullptr;
    unsigned long jpeg_size = 0;
    const int ret = tjCompress2(
        handle_,
        image.bgr.data(),
        image.width,
        0,
        image.height,
        kTurboJpegPixelFormatBgr,
        &jpeg_buf,
        &jpeg_size,
        kTurboJpegSubsample420,
        quality,
        0);
    if (ret != 0) {
      std::string err = error_string();
      if (jpeg_buf) {
        tjFree(jpeg_buf);
      }
      throw std::runtime_error("tjCompress2 failed: " + err);
    }
    std::vector<uint8_t> out(jpeg_buf, jpeg_buf + jpeg_size);
    tjFree(jpeg_buf);
    return out;
  }

 private:
  std::string error_string() const {
    const char* err = tjGetErrorStr2(handle_);
    return err ? std::string(err) : std::string("unknown");
  }

  tjhandle handle_ = nullptr;
};

struct DecodedFrame {
  size_t index = 0;
  std::string image_path;
  Image image;
  double file_read_ms = 0.0;
  double jpeg_decode_ms = 0.0;
  double camera_read_ms = 0.0;
};

class ImageDecodePrefetcher {
 public:
  explicit ImageDecodePrefetcher(const std::vector<std::string>& paths)
      : paths_(paths) {
    worker_ = std::thread(&ImageDecodePrefetcher::worker_loop, this);
  }

  ~ImageDecodePrefetcher() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stop_ = true;
    }
    cv_.notify_all();
    if (worker_.joinable()) {
      worker_.join();
    }
  }

  DecodedFrame pop() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [&]() { return !queue_.empty() || done_ || error_ != nullptr; });
    if (error_ != nullptr) {
      std::rethrow_exception(error_);
    }
    if (queue_.empty()) {
      throw std::runtime_error("decode prefetch queue ended unexpectedly");
    }
    DecodedFrame frame = std::move(queue_.front());
    queue_.pop_front();
    cv_.notify_all();
    return frame;
  }

 private:
  void worker_loop() {
    try {
      TurboJpegDecoder decoder;
      for (size_t i = 0; i < paths_.size(); ++i) {
        {
          std::unique_lock<std::mutex> lock(mutex_);
          cv_.wait(lock, [&]() { return stop_ || queue_.size() < kMaxQueue; });
          if (stop_) {
            return;
          }
        }

        const std::string& image_path = paths_[i];
        double t0 = steady_seconds();
        const std::vector<uint8_t> jpeg = read_bytes(image_path);
        double t1 = steady_seconds();
        const double file_read_ms = (t1 - t0) * 1000.0;
        t0 = steady_seconds();
        Image image = decoder.decode_buffer(jpeg.data(), jpeg.size());
        t1 = steady_seconds();
        const double jpeg_decode_ms = (t1 - t0) * 1000.0;

        DecodedFrame frame;
        frame.index = i;
        frame.image_path = image_path;
        frame.image = std::move(image);
        frame.file_read_ms = file_read_ms;
        frame.jpeg_decode_ms = jpeg_decode_ms;
        {
          std::lock_guard<std::mutex> lock(mutex_);
          queue_.push_back(std::move(frame));
        }
        cv_.notify_all();
      }
      {
        std::lock_guard<std::mutex> lock(mutex_);
        done_ = true;
      }
      cv_.notify_all();
    } catch (...) {
      {
        std::lock_guard<std::mutex> lock(mutex_);
        error_ = std::current_exception();
        done_ = true;
      }
      cv_.notify_all();
    }
  }

  static constexpr size_t kMaxQueue = 3;
  const std::vector<std::string>& paths_;
  std::thread worker_;
  std::mutex mutex_;
  std::condition_variable cv_;
  std::deque<DecodedFrame> queue_;
  bool stop_ = false;
  bool done_ = false;
  std::exception_ptr error_;
};

class V4L2MjpegCamera {
 public:
  struct Frame {
    const uint8_t* data = nullptr;
    size_t size = 0;
    int index = -1;
  };

  V4L2MjpegCamera(const std::string& device,
                  int width,
                  int height,
                  int fps,
                  int buffer_count,
                  int timeout_ms)
      : device_(device),
        width_(width),
        height_(height),
        fps_(fps),
        buffer_count_(std::max(2, buffer_count)),
        timeout_ms_(timeout_ms) {}

  ~V4L2MjpegCamera() {
    try {
      close();
    } catch (...) {
    }
  }

  void open() {
    fd_ = ::open(device_.c_str(), O_RDWR | O_NONBLOCK, 0);
    if (fd_ < 0) {
      throw std::runtime_error("cannot open camera " + device_ + ": " + std::strerror(errno));
    }

    v4l2_capability cap{};
    if (xioctl(fd_, VIDIOC_QUERYCAP, &cap) < 0) {
      throw std::runtime_error("VIDIOC_QUERYCAP failed: " + std::string(std::strerror(errno)));
    }
    if (!(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE) || !(cap.capabilities & V4L2_CAP_STREAMING)) {
      throw std::runtime_error("camera must support VIDEO_CAPTURE and STREAMING");
    }

    v4l2_format fmt{};
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = static_cast<uint32_t>(width_);
    fmt.fmt.pix.height = static_cast<uint32_t>(height_);
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_MJPEG;
    fmt.fmt.pix.field = V4L2_FIELD_ANY;
    if (xioctl(fd_, VIDIOC_S_FMT, &fmt) < 0) {
      throw std::runtime_error("VIDIOC_S_FMT MJPEG failed: " + std::string(std::strerror(errno)));
    }
    if (fmt.fmt.pix.pixelformat != V4L2_PIX_FMT_MJPEG) {
      throw std::runtime_error("camera did not accept MJPEG format");
    }
    width_ = static_cast<int>(fmt.fmt.pix.width);
    height_ = static_cast<int>(fmt.fmt.pix.height);

    if (fps_ > 0) {
      v4l2_streamparm parm{};
      parm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
      parm.parm.capture.timeperframe.numerator = 1;
      parm.parm.capture.timeperframe.denominator = static_cast<uint32_t>(fps_);
      xioctl(fd_, VIDIOC_S_PARM, &parm);
    }

    v4l2_requestbuffers req{};
    req.count = static_cast<uint32_t>(buffer_count_);
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;
    if (xioctl(fd_, VIDIOC_REQBUFS, &req) < 0) {
      throw std::runtime_error("VIDIOC_REQBUFS failed: " + std::string(std::strerror(errno)));
    }
    if (req.count < 2) {
      throw std::runtime_error("camera returned too few buffers");
    }

    buffers_.resize(req.count);
    for (uint32_t i = 0; i < req.count; ++i) {
      v4l2_buffer buf{};
      buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
      buf.memory = V4L2_MEMORY_MMAP;
      buf.index = i;
      if (xioctl(fd_, VIDIOC_QUERYBUF, &buf) < 0) {
        throw std::runtime_error("VIDIOC_QUERYBUF failed: " + std::string(std::strerror(errno)));
      }
      void* start = mmap(nullptr, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, buf.m.offset);
      if (start == MAP_FAILED) {
        throw std::runtime_error("mmap camera buffer failed: " + std::string(std::strerror(errno)));
      }
      buffers_[i].start = start;
      buffers_[i].length = buf.length;
    }

    for (uint32_t i = 0; i < buffers_.size(); ++i) {
      v4l2_buffer buf{};
      buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
      buf.memory = V4L2_MEMORY_MMAP;
      buf.index = i;
      if (xioctl(fd_, VIDIOC_QBUF, &buf) < 0) {
        throw std::runtime_error("VIDIOC_QBUF failed: " + std::string(std::strerror(errno)));
      }
    }

    v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (xioctl(fd_, VIDIOC_STREAMON, &type) < 0) {
      throw std::runtime_error("VIDIOC_STREAMON failed: " + std::string(std::strerror(errno)));
    }
    streaming_ = true;
  }

  Frame read_frame() {
    for (;;) {
      fd_set fds;
      FD_ZERO(&fds);
      FD_SET(fd_, &fds);
      timeval tv{};
      tv.tv_sec = timeout_ms_ / 1000;
      tv.tv_usec = (timeout_ms_ % 1000) * 1000;
      const int sel = select(fd_ + 1, &fds, nullptr, nullptr, &tv);
      if (sel < 0) {
        if (errno == EINTR) {
          continue;
        }
        throw std::runtime_error("select camera failed: " + std::string(std::strerror(errno)));
      }
      if (sel == 0) {
        throw std::runtime_error("camera read timeout");
      }

      v4l2_buffer buf{};
      buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
      buf.memory = V4L2_MEMORY_MMAP;
      if (xioctl(fd_, VIDIOC_DQBUF, &buf) < 0) {
        if (errno == EAGAIN) {
          continue;
        }
        throw std::runtime_error("VIDIOC_DQBUF failed: " + std::string(std::strerror(errno)));
      }
      if (buf.index >= buffers_.size()) {
        throw std::runtime_error("camera returned invalid buffer index");
      }
      Frame frame;
      frame.data = static_cast<const uint8_t*>(buffers_[buf.index].start);
      frame.size = buf.bytesused;
      frame.index = static_cast<int>(buf.index);
      return frame;
    }
  }

  void release_frame(const Frame& frame) {
    if (frame.index < 0) {
      return;
    }
    v4l2_buffer buf{};
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.index = static_cast<uint32_t>(frame.index);
    if (xioctl(fd_, VIDIOC_QBUF, &buf) < 0) {
      throw std::runtime_error("VIDIOC_QBUF release failed: " + std::string(std::strerror(errno)));
    }
  }

  void close() {
    if (fd_ < 0) {
      return;
    }
    if (streaming_) {
      v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
      xioctl(fd_, VIDIOC_STREAMOFF, &type);
      streaming_ = false;
    }
    for (Buffer& buffer : buffers_) {
      if (buffer.start && buffer.start != MAP_FAILED) {
        munmap(buffer.start, buffer.length);
        buffer.start = nullptr;
        buffer.length = 0;
      }
    }
    buffers_.clear();
    ::close(fd_);
    fd_ = -1;
  }

 private:
  struct Buffer {
    void* start = nullptr;
    size_t length = 0;
  };

  std::string device_;
  int width_ = 0;
  int height_ = 0;
  int fps_ = 0;
  int buffer_count_ = 0;
  int timeout_ms_ = 0;
  int fd_ = -1;
  bool streaming_ = false;
  std::vector<Buffer> buffers_;
};

class CameraDecodePrefetcher {
 public:
  explicit CameraDecodePrefetcher(const Config& cfg)
      : cfg_(cfg) {
    worker_ = std::thread(&CameraDecodePrefetcher::worker_loop, this);
  }

  ~CameraDecodePrefetcher() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stop_ = true;
    }
    cv_.notify_all();
    if (worker_.joinable()) {
      worker_.join();
    }
  }

  DecodedFrame pop() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [&]() { return !queue_.empty() || done_ || error_ != nullptr; });
    if (error_ != nullptr) {
      std::rethrow_exception(error_);
    }
    if (queue_.empty()) {
      throw std::runtime_error("camera decode prefetch queue ended unexpectedly");
    }
    DecodedFrame frame = std::move(queue_.front());
    queue_.pop_front();
    cv_.notify_all();
    return frame;
  }

 private:
  void worker_loop() {
    try {
      TurboJpegDecoder decoder;
      V4L2MjpegCamera camera(
          cfg_.camera_device,
          cfg_.camera_width,
          cfg_.camera_height,
          cfg_.camera_fps,
          cfg_.camera_buffers,
          cfg_.camera_timeout_ms);
      camera.open();

      for (int i = 0; cfg_.max_frames == 0 || i < cfg_.max_frames; ++i) {
        {
          std::unique_lock<std::mutex> lock(mutex_);
          cv_.wait(lock, [&]() { return stop_ || queue_.size() < kMaxQueue; });
          if (stop_) {
            return;
          }
        }

        V4L2MjpegCamera::Frame camera_frame;
        bool frame_acquired = false;
        double t0 = steady_seconds();
        camera_frame = camera.read_frame();
        double t1 = steady_seconds();
        frame_acquired = true;
        const double camera_read_ms = (t1 - t0) * 1000.0;

        Image image;
        double jpeg_decode_ms = 0.0;
        try {
          t0 = steady_seconds();
          image = decoder.decode_buffer(camera_frame.data, camera_frame.size);
          t1 = steady_seconds();
          jpeg_decode_ms = (t1 - t0) * 1000.0;
          camera.release_frame(camera_frame);
          frame_acquired = false;
        } catch (...) {
          if (frame_acquired) {
            try {
              camera.release_frame(camera_frame);
            } catch (...) {
            }
          }
          throw;
        }

        std::ostringstream frame_name;
        frame_name << cfg_.camera_device << "#" << i;
        DecodedFrame decoded;
        decoded.index = static_cast<size_t>(i);
        decoded.image_path = frame_name.str();
        decoded.image = std::move(image);
        decoded.camera_read_ms = camera_read_ms;
        decoded.jpeg_decode_ms = jpeg_decode_ms;
        {
          std::lock_guard<std::mutex> lock(mutex_);
          queue_.push_back(std::move(decoded));
        }
        cv_.notify_all();
      }

      {
        std::lock_guard<std::mutex> lock(mutex_);
        done_ = true;
      }
      cv_.notify_all();
    } catch (...) {
      {
        std::lock_guard<std::mutex> lock(mutex_);
        error_ = std::current_exception();
        done_ = true;
      }
      cv_.notify_all();
    }
  }

static constexpr size_t kMaxQueue = 3;
  Config cfg_;
  std::thread worker_;
  std::mutex mutex_;
  std::condition_variable cv_;
  std::deque<DecodedFrame> queue_;
  bool stop_ = false;
  bool done_ = false;
  std::exception_ptr error_;
};

static int seconds_to_frames(double seconds, double fps) {
  if (!std::isfinite(seconds) || seconds <= 0.0) {
    return 0;
  }
  const double safe_fps = std::max(1.0, fps);
  const double frames = std::round(seconds * safe_fps);
  if (!std::isfinite(frames) || frames <= 0.0) {
    return 1;
  }
  return static_cast<int>(
      std::min(frames, static_cast<double>(std::numeric_limits<int>::max())));
}
