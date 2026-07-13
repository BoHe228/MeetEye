#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <linux/videodev2.h>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/select.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <utility>
#include <vector>

using tjhandle = void*;

namespace {

constexpr int kTurboJpegPixelFormatBgr = 1;

struct Config {
  std::string camera_device = "/dev/video0";
  int camera_width = 1920;
  int camera_height = 1080;
  double camera_fps = 30.0;
  std::string camera_format = "mjpeg";
  int camera_warmup_frames = 10;
  int sample_frames = 30;
  int camera_buffers = 4;
  int camera_timeout_ms = 3000;

  std::string output_map_file = "maps/6.22_2560_yolo_slices_640.npz";
  std::string cpp_output_dir = "maps/6.22_2560_yolo_slices_640_cpp";
  int imgsz = 640;
  int output_width = 2560;
  int output_height = 720;
  int process_width = 2560;
  int crop_divisor = 3;
  int num_slices = 3;
  double slice_overlap = 0.1;
  double vertical_fov = 100.0;

  double center_x = -1.0;
  double center_y = -1.0;
  double radius = -1.0;
  bool no_camera = false;
};

struct GrayFrame {
  int width = 0;
  int height = 0;
  std::vector<uint8_t> gray;
};

struct FisheyeModel {
  int img_width = 0;
  int img_height = 0;
  int center_x = 0;
  int center_y = 0;
  int radius = 0;
  std::string source;
};

struct FloatMap {
  int width = 0;
  int height = 0;
  std::vector<float> data;

  FloatMap() = default;
  FloatMap(int w, int h) : width(w), height(h), data(static_cast<size_t>(w) * h) {}

  float& at(int y, int x) {
    return data[static_cast<size_t>(y) * width + x];
  }
  float at(int y, int x) const {
    return data[static_cast<size_t>(y) * width + x];
  }
};

struct SliceInfo {
  int slice_idx = 0;
  int start_x = 0;
  int actual_start_x = 0;
  int end_x = 0;
  int slice_width = 0;
  int slice_height = 0;
  int original_width = 0;
  int original_height = 0;
  bool wrap_around = false;
};

struct LetterboxInfo {
  double gain = 1.0;
  int left = 0;
  int top = 0;
  int new_width = 0;
  int new_height = 0;
};

static std::string to_lower(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return value;
}

static int xioctl(int fd, unsigned long request, void* arg) {
  int r = 0;
  do {
    r = ioctl(fd, request, arg);
  } while (r < 0 && errno == EINTR);
  return r;
}

static int clamp_int(int value, int lo, int hi) {
  return std::max(lo, std::min(hi, value));
}

static float clamp_float(float value, float lo, float hi) {
  return std::max(lo, std::min(hi, value));
}

static int python_round_int(double value) {
  return static_cast<int>(std::nearbyint(value));
}

static std::string join_path(const std::string& base, const std::string& name) {
  if (base.empty()) {
    return name;
  }
  if (base[base.size() - 1] == '/') {
    return base + name;
  }
  return base + "/" + name;
}

static bool mkdir_p(const std::string& path) {
  if (path.empty()) {
    return true;
  }
  std::string partial;
  size_t start = 0;
  if (path[0] == '/') {
    partial = "/";
    start = 1;
  }
  while (start <= path.size()) {
    size_t slash = path.find('/', start);
    const std::string part = path.substr(start, slash == std::string::npos ? std::string::npos : slash - start);
    if (!part.empty()) {
      if (!partial.empty() && partial[partial.size() - 1] != '/') {
        partial += "/";
      }
      partial += part;
      if (mkdir(partial.c_str(), 0775) < 0 && errno != EEXIST) {
        return false;
      }
    }
    if (slash == std::string::npos) {
      break;
    }
    start = slash + 1;
  }
  return true;
}

static std::string require_value(int argc, char** argv, int* index) {
  if (*index + 1 >= argc) {
    throw std::runtime_error(std::string("missing value for ") + argv[*index]);
  }
  *index += 1;
  return argv[*index];
}

static int parse_int_arg(const std::string& value, const std::string& name) {
  char* end = nullptr;
  const long parsed = std::strtol(value.c_str(), &end, 10);
  if (end == value.c_str() || *end != '\0') {
    throw std::runtime_error("invalid integer for " + name + ": " + value);
  }
  return static_cast<int>(parsed);
}

static double parse_double_arg(const std::string& value, const std::string& name) {
  char* end = nullptr;
  const double parsed = std::strtod(value.c_str(), &end);
  if (end == value.c_str() || *end != '\0' || !std::isfinite(parsed)) {
    throw std::runtime_error("invalid number for " + name + ": " + value);
  }
  return parsed;
}

static void print_usage(const char* argv0) {
  std::cerr
      << "Usage: " << argv0 << " [options]\n"
      << "\n"
      << "Generate board_cpp direct-slice binary maps without Python/OpenCV.\n"
      << "\n"
      << "Camera/model options:\n"
      << "  --camera-device PATH        V4L2 camera, default: /dev/video0\n"
      << "  --camera-width N            capture width, default: 1920\n"
      << "  --camera-height N           capture height, default: 1080\n"
      << "  --camera-fps N              requested FPS, default: 30\n"
      << "  --camera-format mjpeg|yuyv|any, default: mjpeg\n"
      << "  --camera-warmup-frames N    frames to drop before averaging, default: 10\n"
      << "  --sample-frames N           frames to average for simple fisheye detection, default: 30\n"
      << "  --center-x N --center-y N --radius N\n"
      << "                              override fisheye model and skip camera when all are set\n"
      << "  --no-camera                 use camera width/height and explicit/default fisheye model\n"
      << "\n"
      << "Map options:\n"
      << "  --cpp-output-dir DIR        output directory for map_x.bin/map_y.bin/meta.txt\n"
      << "  --output-map-file PATH      source_npz metadata value only; no .npz is written\n"
      << "  --imgsz N                   model input size, default: 640\n"
      << "  --output-width N            base panorama width, default: 2560\n"
      << "  --output-height N           base panorama height, default: 720\n"
      << "  --process-width N           cropped panorama width, default: 2560\n"
      << "  --crop-divisor N            crop top = output_height / N, default: 3\n"
      << "  --num-slices N              number of direct slices, default: 3\n"
      << "  --slice-overlap R           overlap ratio, default: 0.1\n"
      << "  --vertical-fov N            accepted for CLI compatibility, default: 100\n";
}

static Config parse_args(int argc, char** argv) {
  Config cfg;
  for (int i = 1; i < argc; ++i) {
    const std::string opt = argv[i];
    if (opt == "--help" || opt == "-h") {
      print_usage(argv[0]);
      std::exit(0);
    } else if (opt == "--camera-device") {
      cfg.camera_device = require_value(argc, argv, &i);
    } else if (opt == "--camera-width") {
      cfg.camera_width = parse_int_arg(require_value(argc, argv, &i), opt);
    } else if (opt == "--camera-height") {
      cfg.camera_height = parse_int_arg(require_value(argc, argv, &i), opt);
    } else if (opt == "--camera-fps") {
      cfg.camera_fps = parse_double_arg(require_value(argc, argv, &i), opt);
    } else if (opt == "--camera-format") {
      cfg.camera_format = to_lower(require_value(argc, argv, &i));
    } else if (opt == "--camera-warmup-frames") {
      cfg.camera_warmup_frames = parse_int_arg(require_value(argc, argv, &i), opt);
    } else if (opt == "--sample-frames") {
      cfg.sample_frames = parse_int_arg(require_value(argc, argv, &i), opt);
    } else if (opt == "--camera-buffers") {
      cfg.camera_buffers = parse_int_arg(require_value(argc, argv, &i), opt);
    } else if (opt == "--camera-timeout-ms") {
      cfg.camera_timeout_ms = parse_int_arg(require_value(argc, argv, &i), opt);
    } else if (opt == "--output-map-file") {
      cfg.output_map_file = require_value(argc, argv, &i);
    } else if (opt == "--cpp-output-dir") {
      cfg.cpp_output_dir = require_value(argc, argv, &i);
    } else if (opt == "--imgsz") {
      cfg.imgsz = parse_int_arg(require_value(argc, argv, &i), opt);
    } else if (opt == "--output-width") {
      cfg.output_width = parse_int_arg(require_value(argc, argv, &i), opt);
    } else if (opt == "--output-height") {
      cfg.output_height = parse_int_arg(require_value(argc, argv, &i), opt);
    } else if (opt == "--process-width") {
      cfg.process_width = parse_int_arg(require_value(argc, argv, &i), opt);
    } else if (opt == "--crop-divisor") {
      cfg.crop_divisor = parse_int_arg(require_value(argc, argv, &i), opt);
    } else if (opt == "--num-slices") {
      cfg.num_slices = parse_int_arg(require_value(argc, argv, &i), opt);
    } else if (opt == "--slice-overlap") {
      cfg.slice_overlap = parse_double_arg(require_value(argc, argv, &i), opt);
    } else if (opt == "--vertical-fov") {
      cfg.vertical_fov = parse_double_arg(require_value(argc, argv, &i), opt);
    } else if (opt == "--center-x") {
      cfg.center_x = parse_double_arg(require_value(argc, argv, &i), opt);
    } else if (opt == "--center-y") {
      cfg.center_y = parse_double_arg(require_value(argc, argv, &i), opt);
    } else if (opt == "--radius") {
      cfg.radius = parse_double_arg(require_value(argc, argv, &i), opt);
    } else if (opt == "--no-camera") {
      cfg.no_camera = true;
    } else if (opt == "--no-cpp-export") {
      std::cerr << "[generate-board-cpp-map] warning: --no-cpp-export ignored by the C++ generator\n";
    } else {
      throw std::runtime_error("unknown option: " + opt);
    }
  }

  cfg.camera_format = to_lower(cfg.camera_format);
  if (cfg.camera_format != "mjpeg" && cfg.camera_format != "yuyv" && cfg.camera_format != "any") {
    throw std::runtime_error("--camera-format must be mjpeg, yuyv, or any");
  }
  if (cfg.camera_width <= 0 || cfg.camera_height <= 0) {
    throw std::runtime_error("camera width/height must be positive");
  }
  if (cfg.output_width <= 0 || cfg.output_height <= 1 || cfg.imgsz <= 0) {
    throw std::runtime_error("output width/height and imgsz must be positive");
  }
  if (cfg.process_width <= 0) {
    cfg.process_width = cfg.output_width;
  }
  if (cfg.crop_divisor <= 0) {
    cfg.crop_divisor = 1;
  }
  if (cfg.num_slices <= 0) {
    throw std::runtime_error("--num-slices must be positive");
  }
  if (cfg.slice_overlap < 0.0 || cfg.slice_overlap >= 0.5) {
    throw std::runtime_error("--slice-overlap should be in [0, 0.5)");
  }
  cfg.camera_warmup_frames = std::max(0, cfg.camera_warmup_frames);
  cfg.sample_frames = std::max(1, cfg.sample_frames);
  cfg.camera_buffers = std::max(2, cfg.camera_buffers);
  cfg.camera_timeout_ms = std::max(100, cfg.camera_timeout_ms);
  return cfg;
}

static uint32_t pixel_format_from_name(const std::string& fmt) {
  if (fmt == "yuyv") {
    return V4L2_PIX_FMT_YUYV;
  }
  return V4L2_PIX_FMT_MJPEG;
}

static std::string pixel_format_name(uint32_t fmt) {
  if (fmt == V4L2_PIX_FMT_MJPEG) {
    return "mjpeg";
  }
  if (fmt == V4L2_PIX_FMT_YUYV) {
    return "yuyv";
  }
  std::ostringstream oss;
  oss << "0x" << std::hex << fmt;
  return oss.str();
}

class V4L2Camera {
 public:
  struct Frame {
    const uint8_t* data = nullptr;
    size_t size = 0;
    int index = -1;
  };

  explicit V4L2Camera(const Config& cfg)
      : device_(cfg.camera_device),
        width_(cfg.camera_width),
        height_(cfg.camera_height),
        fps_(cfg.camera_fps),
        requested_format_(cfg.camera_format),
        buffer_count_(cfg.camera_buffers),
        timeout_ms_(cfg.camera_timeout_ms) {}

  ~V4L2Camera() {
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

    std::vector<uint32_t> candidates;
    if (requested_format_ == "any") {
      candidates.push_back(V4L2_PIX_FMT_MJPEG);
      candidates.push_back(V4L2_PIX_FMT_YUYV);
    } else {
      candidates.push_back(pixel_format_from_name(requested_format_));
    }

    bool format_ok = false;
    v4l2_format fmt{};
    for (uint32_t candidate : candidates) {
      std::memset(&fmt, 0, sizeof(fmt));
      fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
      fmt.fmt.pix.width = static_cast<uint32_t>(width_);
      fmt.fmt.pix.height = static_cast<uint32_t>(height_);
      fmt.fmt.pix.pixelformat = candidate;
      fmt.fmt.pix.field = V4L2_FIELD_ANY;
      if (xioctl(fd_, VIDIOC_S_FMT, &fmt) < 0) {
        continue;
      }
      if (fmt.fmt.pix.pixelformat == V4L2_PIX_FMT_MJPEG ||
          fmt.fmt.pix.pixelformat == V4L2_PIX_FMT_YUYV) {
        if (requested_format_ == "any" || fmt.fmt.pix.pixelformat == candidate) {
          format_ok = true;
          break;
        }
      }
    }
    if (!format_ok) {
      throw std::runtime_error("camera did not accept MJPEG/YUYV capture format");
    }

    width_ = static_cast<int>(fmt.fmt.pix.width);
    height_ = static_cast<int>(fmt.fmt.pix.height);
    pixel_format_ = fmt.fmt.pix.pixelformat;

    if (fps_ > 0.0) {
      v4l2_streamparm parm{};
      parm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
      parm.parm.capture.timeperframe.numerator = 1;
      parm.parm.capture.timeperframe.denominator = static_cast<uint32_t>(std::max(1.0, fps_));
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
      throw std::runtime_error("camera returned too few mmap buffers");
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

  int width() const { return width_; }
  int height() const { return height_; }
  uint32_t pixel_format() const { return pixel_format_; }

 private:
  struct Buffer {
    void* start = nullptr;
    size_t length = 0;
  };

  std::string device_;
  int width_ = 0;
  int height_ = 0;
  double fps_ = 0.0;
  std::string requested_format_;
  int buffer_count_ = 4;
  int timeout_ms_ = 3000;
  int fd_ = -1;
  uint32_t pixel_format_ = V4L2_PIX_FMT_MJPEG;
  bool streaming_ = false;
  std::vector<Buffer> buffers_;
};

class TurboJpegGrayDecoder {
 public:
  TurboJpegGrayDecoder() {
    const char* candidates[] = {
        "libturbojpeg.so.0",
        "libturbojpeg.so",
        "lib/libturbojpeg.so.0",
        "lib/libturbojpeg.so",
        "face_rc/board_cpp/lib/libturbojpeg.so.0",
        "face_rc/board_cpp/lib/libturbojpeg.so",
        "/usr/lib/aarch64-linux-gnu/libturbojpeg.so.0",
        "/usr/lib/aarch64-linux-gnu/libturbojpeg.so",
        "/usr/lib/libturbojpeg.so.0",
        "/usr/lib/libturbojpeg.so",
        "/lib/aarch64-linux-gnu/libturbojpeg.so.0",
        "/lib/aarch64-linux-gnu/libturbojpeg.so",
    };
    for (const char* candidate : candidates) {
      lib_ = dlopen(candidate, RTLD_LAZY);
      if (lib_) {
        break;
      }
    }
    if (!lib_) {
      throw std::runtime_error("cannot load libturbojpeg.so.0 for MJPEG decoding");
    }
    init_decompress_ = load_symbol<TjInitDecompressFn>("tjInitDecompress");
    destroy_ = load_symbol<TjDestroyFn>("tjDestroy");
    decompress_header3_ = load_symbol<TjDecompressHeader3Fn>("tjDecompressHeader3");
    decompress2_ = load_symbol<TjDecompress2Fn>("tjDecompress2");
    get_error_ = load_symbol<TjGetErrorStr2Fn>("tjGetErrorStr2");

    handle_ = init_decompress_();
    if (!handle_) {
      throw std::runtime_error("tjInitDecompress failed");
    }
  }

  ~TurboJpegGrayDecoder() {
    if (handle_) {
      destroy_(handle_);
      handle_ = nullptr;
    }
    if (lib_) {
      dlclose(lib_);
      lib_ = nullptr;
    }
  }

  GrayFrame decode(const uint8_t* data, size_t size) {
    GrayFrame frame;
    int subsamp = 0;
    int colorspace = 0;
    if (decompress_header3_(handle_,
                            data,
                            static_cast<unsigned long>(size),
                            &frame.width,
                            &frame.height,
                            &subsamp,
                            &colorspace) != 0) {
      throw std::runtime_error("tjDecompressHeader3 failed: " + error_string());
    }
    if (frame.width <= 0 || frame.height <= 0) {
      throw std::runtime_error("invalid MJPEG frame shape");
    }

    std::vector<uint8_t> bgr(static_cast<size_t>(frame.width) * frame.height * 3);
    if (decompress2_(handle_,
                     data,
                     static_cast<unsigned long>(size),
                     bgr.data(),
                     frame.width,
                     0,
                     frame.height,
                     kTurboJpegPixelFormatBgr,
                     0) != 0) {
      throw std::runtime_error("tjDecompress2 failed: " + error_string());
    }

    frame.gray.resize(static_cast<size_t>(frame.width) * frame.height);
    for (size_t i = 0; i < frame.gray.size(); ++i) {
      const uint8_t b = bgr[i * 3 + 0];
      const uint8_t g = bgr[i * 3 + 1];
      const uint8_t r = bgr[i * 3 + 2];
      frame.gray[i] = static_cast<uint8_t>((29 * b + 150 * g + 77 * r) >> 8);
    }
    return frame;
  }

 private:
  using TjInitDecompressFn = tjhandle (*)();
  using TjDestroyFn = int (*)(tjhandle);
  using TjDecompressHeader3Fn = int (*)(tjhandle,
                                        const unsigned char*,
                                        unsigned long,
                                        int*,
                                        int*,
                                        int*,
                                        int*);
  using TjDecompress2Fn = int (*)(tjhandle,
                                  const unsigned char*,
                                  unsigned long,
                                  unsigned char*,
                                  int,
                                  int,
                                  int,
                                  int,
                                  int);
  using TjGetErrorStr2Fn = char* (*)(tjhandle);

  template <typename T>
  T load_symbol(const char* name) {
    dlerror();
    void* sym = dlsym(lib_, name);
    const char* err = dlerror();
    if (err != nullptr || sym == nullptr) {
      throw std::runtime_error(std::string("cannot load TurboJPEG symbol ") + name);
    }
    return reinterpret_cast<T>(sym);
  }

  std::string error_string() const {
    const char* err = get_error_ ? get_error_(handle_) : nullptr;
    return err ? std::string(err) : std::string("unknown");
  }

  void* lib_ = nullptr;
  tjhandle handle_ = nullptr;
  TjInitDecompressFn init_decompress_ = nullptr;
  TjDestroyFn destroy_ = nullptr;
  TjDecompressHeader3Fn decompress_header3_ = nullptr;
  TjDecompress2Fn decompress2_ = nullptr;
  TjGetErrorStr2Fn get_error_ = nullptr;
};

static GrayFrame decode_yuyv_gray(const uint8_t* data, size_t size, int width, int height) {
  const size_t expected = static_cast<size_t>(width) * height * 2;
  if (size < expected) {
    throw std::runtime_error("YUYV frame is smaller than expected");
  }
  GrayFrame frame;
  frame.width = width;
  frame.height = height;
  frame.gray.resize(static_cast<size_t>(width) * height);
  for (size_t i = 0; i < frame.gray.size(); ++i) {
    frame.gray[i] = data[i * 2];
  }
  return frame;
}

static GrayFrame read_average_camera_gray(const Config& cfg) {
  V4L2Camera camera(cfg);
  camera.open();
  std::cerr << "[generate-board-cpp-map] camera opened: "
            << camera.width() << "x" << camera.height()
            << " format=" << pixel_format_name(camera.pixel_format()) << "\n";

  std::unique_ptr<TurboJpegGrayDecoder> jpeg_decoder;
  std::vector<double> acc;
  int avg_w = 0;
  int avg_h = 0;
  int count = 0;
  const int total_frames = cfg.camera_warmup_frames + cfg.sample_frames;
  for (int i = 0; i < total_frames; ++i) {
    V4L2Camera::Frame raw = camera.read_frame();
    try {
      if (i >= cfg.camera_warmup_frames) {
        GrayFrame frame;
        if (camera.pixel_format() == V4L2_PIX_FMT_MJPEG) {
          if (!jpeg_decoder) {
            jpeg_decoder.reset(new TurboJpegGrayDecoder());
          }
          frame = jpeg_decoder->decode(raw.data, raw.size);
        } else if (camera.pixel_format() == V4L2_PIX_FMT_YUYV) {
          frame = decode_yuyv_gray(raw.data, raw.size, camera.width(), camera.height());
        } else {
          throw std::runtime_error("unsupported camera format: " + pixel_format_name(camera.pixel_format()));
        }

        if (count == 0) {
          avg_w = frame.width;
          avg_h = frame.height;
          acc.assign(static_cast<size_t>(avg_w) * avg_h, 0.0);
        }
        if (frame.width != avg_w || frame.height != avg_h) {
          throw std::runtime_error("camera frame size changed during sampling");
        }
        for (size_t k = 0; k < frame.gray.size(); ++k) {
          acc[k] += static_cast<double>(frame.gray[k]);
        }
        ++count;
      }
      camera.release_frame(raw);
    } catch (...) {
      camera.release_frame(raw);
      throw;
    }
  }

  if (count <= 0 || avg_w <= 0 || avg_h <= 0) {
    throw std::runtime_error("cannot read average camera frame");
  }
  GrayFrame avg;
  avg.width = avg_w;
  avg.height = avg_h;
  avg.gray.resize(static_cast<size_t>(avg_w) * avg_h);
  for (size_t k = 0; k < avg.gray.size(); ++k) {
    avg.gray[k] = static_cast<uint8_t>(clamp_int(python_round_int(acc[k] / count), 0, 255));
  }
  std::cerr << "[generate-board-cpp-map] averaged " << count
            << " frame(s), frame=" << avg.width << "x" << avg.height << "\n";
  return avg;
}

static FisheyeModel default_model(int width, int height) {
  FisheyeModel model;
  model.img_width = width;
  model.img_height = height;
  model.center_x = width / 2;
  model.center_y = height / 2;
  model.radius = std::min(width, height) / 2;
  model.source = "default";
  return model;
}

static void clip_model(FisheyeModel* model) {
  const int min_dim = std::min(model->img_width, model->img_height);
  const int max_center_offset = min_dim / 4;
  model->center_x = clamp_int(model->center_x,
                              model->img_width / 2 - max_center_offset,
                              model->img_width / 2 + max_center_offset);
  model->center_y = clamp_int(model->center_y,
                              model->img_height / 2 - max_center_offset,
                              model->img_height / 2 + max_center_offset);
  model->radius = clamp_int(model->radius, min_dim / 3, min_dim / 2 + 50);
}

static FisheyeModel detect_fisheye_simple(const GrayFrame& frame) {
  FisheyeModel best = default_model(frame.width, frame.height);
  double best_radius = -1.0;
  const int thresholds[] = {20, 30, 40, 50};
  const int total = frame.width * frame.height;
  const int min_count = std::max(1, total / 100);
  const int min_box = std::min(frame.width, frame.height) / 3;

  for (int threshold : thresholds) {
    int min_x = frame.width;
    int min_y = frame.height;
    int max_x = -1;
    int max_y = -1;
    int count = 0;
    for (int y = 0; y < frame.height; ++y) {
      const size_t row = static_cast<size_t>(y) * frame.width;
      for (int x = 0; x < frame.width; ++x) {
        if (frame.gray[row + x] > threshold) {
          min_x = std::min(min_x, x);
          min_y = std::min(min_y, y);
          max_x = std::max(max_x, x);
          max_y = std::max(max_y, y);
          ++count;
        }
      }
    }
    if (count < min_count || max_x <= min_x || max_y <= min_y) {
      continue;
    }
    const int box_w = max_x - min_x + 1;
    const int box_h = max_y - min_y + 1;
    if (box_w < min_box || box_h < min_box) {
      continue;
    }
    const double radius = 0.5 * std::max(box_w, box_h);
    if (radius > best_radius) {
      best_radius = radius;
      best.img_width = frame.width;
      best.img_height = frame.height;
      best.center_x = python_round_int((min_x + max_x) * 0.5);
      best.center_y = python_round_int((min_y + max_y) * 0.5);
      best.radius = python_round_int(radius);
      best.source = "threshold";
    }
  }
  clip_model(&best);
  return best;
}

static bool has_full_override(const Config& cfg) {
  return cfg.center_x >= 0.0 && cfg.center_y >= 0.0 && cfg.radius > 0.0;
}

static FisheyeModel build_fisheye_model(const Config& cfg) {
  FisheyeModel model;
  if (cfg.no_camera || has_full_override(cfg)) {
    model = default_model(cfg.camera_width, cfg.camera_height);
    model.source = cfg.no_camera ? "no-camera-default" : "override";
  } else {
    const GrayFrame avg = read_average_camera_gray(cfg);
    model = detect_fisheye_simple(avg);
  }

  if (cfg.center_x >= 0.0) {
    model.center_x = python_round_int(cfg.center_x);
    model.source = "override";
  }
  if (cfg.center_y >= 0.0) {
    model.center_y = python_round_int(cfg.center_y);
    model.source = "override";
  }
  if (cfg.radius > 0.0) {
    model.radius = python_round_int(cfg.radius);
    model.source = "override";
  }
  clip_model(&model);
  return model;
}

static std::pair<FloatMap, FloatMap> create_panorama_maps(const FisheyeModel& model,
                                                          int output_width,
                                                          int output_height) {
  FloatMap map_x(output_width, output_height);
  FloatMap map_y(output_width, output_height);
  const double pi = std::acos(-1.0);
  for (int y = 0; y < output_height; ++y) {
    const double r_ratio = static_cast<double>(y) / static_cast<double>(output_height - 1);
    const double radial = r_ratio * static_cast<double>(model.radius);
    for (int x = 0; x < output_width; ++x) {
      const double angle = -2.0 * pi * static_cast<double>(x) / static_cast<double>(output_width);
      const float sx = static_cast<float>(model.center_x + radial * std::cos(angle));
      const float sy = static_cast<float>(model.center_y + radial * std::sin(angle));
      map_x.at(y, x) = clamp_float(sx, 0.0f, static_cast<float>(model.img_width - 1));
      map_y.at(y, x) = clamp_float(sy, 0.0f, static_cast<float>(model.img_height - 1));
    }
  }
  return std::make_pair(std::move(map_x), std::move(map_y));
}

static float resize_sample_linear(const FloatMap& src, int dst_w, int dst_h, int dst_x, int dst_y) {
  if (src.width == 1 && src.height == 1) {
    return src.data[0];
  }
  const double scale_x = static_cast<double>(src.width) / static_cast<double>(dst_w);
  const double scale_y = static_cast<double>(src.height) / static_cast<double>(dst_h);
  double fx = (static_cast<double>(dst_x) + 0.5) * scale_x - 0.5;
  double fy = (static_cast<double>(dst_y) + 0.5) * scale_y - 0.5;
  int x0 = static_cast<int>(std::floor(fx));
  int y0 = static_cast<int>(std::floor(fy));
  double wx = fx - x0;
  double wy = fy - y0;

  if (x0 < 0) {
    x0 = 0;
    wx = 0.0;
  }
  if (y0 < 0) {
    y0 = 0;
    wy = 0.0;
  }
  if (x0 >= src.width - 1) {
    x0 = std::max(0, src.width - 2);
    wx = 1.0;
  }
  if (y0 >= src.height - 1) {
    y0 = std::max(0, src.height - 2);
    wy = 1.0;
  }
  const int x1 = std::min(src.width - 1, x0 + 1);
  const int y1 = std::min(src.height - 1, y0 + 1);
  const double v00 = src.at(y0, x0);
  const double v01 = src.at(y0, x1);
  const double v10 = src.at(y1, x0);
  const double v11 = src.at(y1, x1);
  const double top = v00 * (1.0 - wx) + v01 * wx;
  const double bottom = v10 * (1.0 - wx) + v11 * wx;
  return static_cast<float>(top * (1.0 - wy) + bottom * wy);
}

static FloatMap crop_and_resize_map(const FloatMap& base,
                                    int crop_top,
                                    int process_width,
                                    int process_height) {
  if (crop_top < 0 || crop_top + process_height > base.height) {
    throw std::runtime_error("invalid crop range for base map");
  }
  FloatMap cropped(base.width, process_height);
  for (int y = 0; y < process_height; ++y) {
    const size_t src = static_cast<size_t>(crop_top + y) * base.width;
    const size_t dst = static_cast<size_t>(y) * base.width;
    std::copy(base.data.begin() + src, base.data.begin() + src + base.width, cropped.data.begin() + dst);
  }
  if (process_width == cropped.width) {
    return cropped;
  }

  FloatMap resized(process_width, process_height);
  for (int y = 0; y < process_height; ++y) {
    for (int x = 0; x < process_width; ++x) {
      resized.at(y, x) = resize_sample_linear(cropped, process_width, process_height, x, y);
    }
  }
  return resized;
}

static std::vector<SliceInfo> build_slice_infos(int width, int height, int num_slices, double overlap_ratio) {
  std::vector<SliceInfo> infos;
  const int base_slice_w = width / num_slices;
  const int overlap_w = static_cast<int>(base_slice_w * overlap_ratio);
  for (int idx = 0; idx < num_slices; ++idx) {
    SliceInfo info;
    info.slice_idx = idx;
    info.start_x = idx * base_slice_w - overlap_w;
    info.end_x = (idx + 1) * base_slice_w + overlap_w;
    if (idx == 0 && info.start_x < 0) {
      info.actual_start_x = width + info.start_x;
      info.slice_width = -info.start_x + info.end_x;
      info.wrap_around = true;
    } else if (idx == num_slices - 1 && info.end_x > width) {
      info.actual_start_x = info.start_x;
      info.slice_width = width - info.start_x + info.end_x - width;
      info.wrap_around = true;
    } else {
      info.actual_start_x = info.start_x;
      info.slice_width = info.end_x - info.start_x;
      info.wrap_around = false;
    }
    info.slice_height = height;
    info.original_width = width;
    info.original_height = height;
    infos.push_back(info);
  }
  return infos;
}

static LetterboxInfo make_letterbox_info(int slice_width, int slice_height, int imgsz) {
  LetterboxInfo info;
  info.gain = std::min(static_cast<double>(imgsz) / slice_height,
                       static_cast<double>(imgsz) / slice_width);
  info.new_width = python_round_int(slice_width * info.gain);
  info.new_height = python_round_int(slice_height * info.gain);
  const int pad_w = imgsz - info.new_width;
  const int pad_h = imgsz - info.new_height;
  info.left = python_round_int(pad_w / 2.0 - 0.1);
  info.top = python_round_int(pad_h / 2.0 - 0.1);
  return info;
}

static float bilinear_constant(const FloatMap& src, float x, float y) {
  if (x < 0.0f || y < 0.0f || x > static_cast<float>(src.width - 1) ||
      y > static_cast<float>(src.height - 1)) {
    return 0.0f;
  }
  const int x0 = static_cast<int>(std::floor(x));
  const int y0 = static_cast<int>(std::floor(y));
  const int x1 = x0 + 1;
  const int y1 = y0 + 1;
  const float wx = x - static_cast<float>(x0);
  const float wy = y - static_cast<float>(y0);

  const auto sample = [&](int sx, int sy) -> float {
    if (sx < 0 || sy < 0 || sx >= src.width || sy >= src.height) {
      return 0.0f;
    }
    return src.at(sy, sx);
  };

  const float v00 = sample(x0, y0);
  const float v01 = sample(x1, y0);
  const float v10 = sample(x0, y1);
  const float v11 = sample(x1, y1);
  const float top = v00 * (1.0f - wx) + v01 * wx;
  const float bottom = v10 * (1.0f - wx) + v11 * wx;
  return top * (1.0f - wy) + bottom * wy;
}

static void build_direct_slice_maps(const FloatMap& process_x,
                                    const FloatMap& process_y,
                                    int imgsz,
                                    int num_slices,
                                    double overlap_ratio,
                                    std::vector<float>* out_x,
                                    std::vector<float>* out_y,
                                    std::vector<SliceInfo>* slice_infos,
                                    std::vector<LetterboxInfo>* letterbox_infos,
                                    int* roi_w,
                                    int* roi_h) {
  *slice_infos = build_slice_infos(process_x.width, process_x.height, num_slices, overlap_ratio);
  letterbox_infos->clear();
  for (const SliceInfo& info : *slice_infos) {
    letterbox_infos->push_back(make_letterbox_info(info.slice_width, info.slice_height, imgsz));
  }

  *roi_w = (*letterbox_infos)[0].new_width;
  *roi_h = (*letterbox_infos)[0].new_height;
  for (const LetterboxInfo& lb : *letterbox_infos) {
    if (lb.new_width != *roi_w || lb.new_height != *roi_h) {
      throw std::runtime_error("slice letterbox sizes differ; cannot write a rectangular C++ map stack");
    }
  }

  out_x->assign(static_cast<size_t>(num_slices) * (*roi_w) * (*roi_h), 0.0f);
  out_y->assign(out_x->size(), 0.0f);

  for (int s = 0; s < num_slices; ++s) {
    const SliceInfo& info = (*slice_infos)[static_cast<size_t>(s)];
    const LetterboxInfo& lb = (*letterbox_infos)[static_cast<size_t>(s)];
    for (int y = 0; y < lb.new_height; ++y) {
      const float pano_y = static_cast<float>(static_cast<double>(y) / lb.gain);
      for (int x = 0; x < lb.new_width; ++x) {
        const float xs = static_cast<float>(static_cast<double>(x) / lb.gain);
        float pano_x = 0.0f;
        if (info.wrap_around && info.start_x < 0) {
          const int right_w = -info.start_x;
          pano_x = xs < right_w
              ? static_cast<float>(process_x.width + info.start_x) + xs
              : xs - static_cast<float>(right_w);
        } else if (info.wrap_around && info.end_x > process_x.width) {
          const int left_w = process_x.width - info.start_x;
          pano_x = xs < left_w
              ? static_cast<float>(info.start_x) + xs
              : xs - static_cast<float>(left_w);
        } else {
          pano_x = static_cast<float>(info.start_x) + xs;
        }

        const size_t idx = (static_cast<size_t>(s) * (*roi_h) + y) * (*roi_w) + x;
        (*out_x)[idx] = bilinear_constant(process_x, pano_x, pano_y);
        (*out_y)[idx] = bilinear_constant(process_y, pano_x, pano_y);
      }
    }
  }
}

static void write_floats(const std::string& path, const std::vector<float>& values) {
  std::ofstream out(path.c_str(), std::ios::binary | std::ios::trunc);
  if (!out) {
    throw std::runtime_error("cannot open output file: " + path);
  }
  out.write(reinterpret_cast<const char*>(values.data()),
            static_cast<std::streamsize>(values.size() * sizeof(float)));
  if (!out) {
    throw std::runtime_error("failed to write output file: " + path);
  }
}

static void write_meta(const std::string& path,
                       const Config& cfg,
                       const FisheyeModel& model,
                       int crop_top,
                       int process_width,
                       int process_height,
                       int roi_w,
                       int roi_h,
                       const std::vector<SliceInfo>& slice_infos,
                       const std::vector<LetterboxInfo>& letterbox_infos) {
  std::ofstream out(path.c_str(), std::ios::out | std::ios::trunc);
  if (!out) {
    throw std::runtime_error("cannot open output file: " + path);
  }
  out << std::setprecision(12);
  out << "metadata_format=direct_slice_cpp_v1\n";
  out << "source_npz=" << cfg.output_map_file << "\n";
  out << "num_slices=" << cfg.num_slices << "\n";
  out << "roi_h=" << roi_h << "\n";
  out << "roi_w=" << roi_w << "\n";
  out << "imgsz=" << cfg.imgsz << "\n";
  out << "slice_overlap=" << cfg.slice_overlap << "\n";
  out << "crop_top=" << crop_top << "\n";
  out << "crop_divisor=" << cfg.crop_divisor << "\n";
  out << "process_width=" << process_width << "\n";
  out << "process_height=" << process_height << "\n";
  out << "base_output_width=" << cfg.output_width << "\n";
  out << "base_output_height=" << cfg.output_height << "\n";
  out << "img_width=" << model.img_width << "\n";
  out << "img_height=" << model.img_height << "\n";
  out << "center_x=" << model.center_x << "\n";
  out << "center_y=" << model.center_y << "\n";
  out << "radius=" << model.radius << "\n";
  out << "base_map_width=" << process_width << "\n";
  out << "base_map_height=" << process_height << "\n";
  out << "fisheye_model_source=" << model.source << "\n";

  for (size_t i = 0; i < slice_infos.size(); ++i) {
    const SliceInfo& s = slice_infos[i];
    const LetterboxInfo& lb = letterbox_infos[i];
    out << "slice" << i << ".slice_idx=" << s.slice_idx << "\n";
    out << "slice" << i << ".start_x=" << s.start_x << "\n";
    out << "slice" << i << ".actual_start_x=" << s.actual_start_x << "\n";
    out << "slice" << i << ".end_x=" << s.end_x << "\n";
    out << "slice" << i << ".slice_width=" << s.slice_width << "\n";
    out << "slice" << i << ".slice_height=" << s.slice_height << "\n";
    out << "slice" << i << ".original_width=" << s.original_width << "\n";
    out << "slice" << i << ".original_height=" << s.original_height << "\n";
    out << "slice" << i << ".wrap_around=" << (s.wrap_around ? 1 : 0) << "\n";
    out << "slice" << i << ".gain=" << lb.gain << "\n";
    out << "slice" << i << ".left=" << lb.left << "\n";
    out << "slice" << i << ".top=" << lb.top << "\n";
    out << "slice" << i << ".new_width=" << lb.new_width << "\n";
    out << "slice" << i << ".new_height=" << lb.new_height << "\n";
  }
}

static int run(int argc, char** argv) {
  const Config cfg = parse_args(argc, argv);
  const FisheyeModel model = build_fisheye_model(cfg);
  std::cerr << "[generate-board-cpp-map] fisheye center=("
            << model.center_x << "," << model.center_y << ") radius=" << model.radius
            << " source=" << model.source << "\n";

  const int crop_top = cfg.output_height / cfg.crop_divisor;
  const int process_height = cfg.output_height - crop_top;
  const int process_width = cfg.process_width > 0 ? cfg.process_width : cfg.output_width;

  auto base_maps = create_panorama_maps(model, cfg.output_width, cfg.output_height);
  FloatMap process_x = crop_and_resize_map(base_maps.first, crop_top, process_width, process_height);
  FloatMap process_y = crop_and_resize_map(base_maps.second, crop_top, process_width, process_height);

  std::vector<float> slice_map_x;
  std::vector<float> slice_map_y;
  std::vector<SliceInfo> slice_infos;
  std::vector<LetterboxInfo> letterbox_infos;
  int roi_w = 0;
  int roi_h = 0;
  build_direct_slice_maps(process_x,
                          process_y,
                          cfg.imgsz,
                          cfg.num_slices,
                          cfg.slice_overlap,
                          &slice_map_x,
                          &slice_map_y,
                          &slice_infos,
                          &letterbox_infos,
                          &roi_w,
                          &roi_h);

  if (!mkdir_p(cfg.cpp_output_dir)) {
    throw std::runtime_error("cannot create output directory: " + cfg.cpp_output_dir);
  }
  write_floats(join_path(cfg.cpp_output_dir, "map_x.bin"), slice_map_x);
  write_floats(join_path(cfg.cpp_output_dir, "map_y.bin"), slice_map_y);
  write_floats(join_path(cfg.cpp_output_dir, "base_map_x.bin"), process_x.data);
  write_floats(join_path(cfg.cpp_output_dir, "base_map_y.bin"), process_y.data);
  write_meta(join_path(cfg.cpp_output_dir, "meta.txt"),
             cfg,
             model,
             crop_top,
             process_width,
             process_height,
             roi_w,
             roi_h,
             slice_infos,
             letterbox_infos);

  std::cerr << "[generate-board-cpp-map] saved board_cpp map: " << cfg.cpp_output_dir << "\n";
  std::cerr << "[generate-board-cpp-map] files: map_x.bin map_y.bin base_map_x.bin base_map_y.bin meta.txt\n";
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return run(argc, argv);
  } catch (const std::exception& e) {
    std::cerr << "[generate-board-cpp-map] error: " << e.what() << "\n";
    return 1;
  }
}
