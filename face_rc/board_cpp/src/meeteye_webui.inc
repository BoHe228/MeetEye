static uint32_t sha1_rotl(uint32_t value, int bits) {
  return (value << bits) | (value >> (32 - bits));
}

static std::array<uint8_t, 20> sha1_digest(const std::string& input) {
  std::vector<uint8_t> data(input.begin(), input.end());
  const uint64_t bit_len = static_cast<uint64_t>(data.size()) * 8ULL;
  data.push_back(0x80);
  while ((data.size() % 64) != 56) {
    data.push_back(0x00);
  }
  for (int i = 7; i >= 0; --i) {
    data.push_back(static_cast<uint8_t>((bit_len >> (i * 8)) & 0xff));
  }

  uint32_t h0 = 0x67452301U;
  uint32_t h1 = 0xefcdab89U;
  uint32_t h2 = 0x98badcfeU;
  uint32_t h3 = 0x10325476U;
  uint32_t h4 = 0xc3d2e1f0U;

  for (size_t offset = 0; offset < data.size(); offset += 64) {
    uint32_t w[80] = {0};
    for (int i = 0; i < 16; ++i) {
      const size_t p = offset + static_cast<size_t>(i) * 4;
      w[i] = (static_cast<uint32_t>(data[p]) << 24) |
             (static_cast<uint32_t>(data[p + 1]) << 16) |
             (static_cast<uint32_t>(data[p + 2]) << 8) |
             static_cast<uint32_t>(data[p + 3]);
    }
    for (int i = 16; i < 80; ++i) {
      w[i] = sha1_rotl(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);
    }

    uint32_t a = h0;
    uint32_t b = h1;
    uint32_t c = h2;
    uint32_t d = h3;
    uint32_t e = h4;
    for (int i = 0; i < 80; ++i) {
      uint32_t f = 0;
      uint32_t k = 0;
      if (i < 20) {
        f = (b & c) | ((~b) & d);
        k = 0x5a827999U;
      } else if (i < 40) {
        f = b ^ c ^ d;
        k = 0x6ed9eba1U;
      } else if (i < 60) {
        f = (b & c) | (b & d) | (c & d);
        k = 0x8f1bbcdcU;
      } else {
        f = b ^ c ^ d;
        k = 0xca62c1d6U;
      }
      const uint32_t temp = sha1_rotl(a, 5) + f + e + k + w[i];
      e = d;
      d = c;
      c = sha1_rotl(b, 30);
      b = a;
      a = temp;
    }
    h0 += a;
    h1 += b;
    h2 += c;
    h3 += d;
    h4 += e;
  }

  std::array<uint8_t, 20> out{{0}};
  const uint32_t words[5] = {h0, h1, h2, h3, h4};
  for (int i = 0; i < 5; ++i) {
    out[static_cast<size_t>(i) * 4 + 0] = static_cast<uint8_t>((words[i] >> 24) & 0xff);
    out[static_cast<size_t>(i) * 4 + 1] = static_cast<uint8_t>((words[i] >> 16) & 0xff);
    out[static_cast<size_t>(i) * 4 + 2] = static_cast<uint8_t>((words[i] >> 8) & 0xff);
    out[static_cast<size_t>(i) * 4 + 3] = static_cast<uint8_t>(words[i] & 0xff);
  }
  return out;
}

static std::string base64_encode(const uint8_t* data, size_t size) {
  static const char* table =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string out;
  out.reserve(((size + 2) / 3) * 4);
  for (size_t i = 0; i < size; i += 3) {
    const uint32_t b0 = data[i];
    const uint32_t b1 = i + 1 < size ? data[i + 1] : 0;
    const uint32_t b2 = i + 2 < size ? data[i + 2] : 0;
    const uint32_t triple = (b0 << 16) | (b1 << 8) | b2;
    out.push_back(table[(triple >> 18) & 0x3f]);
    out.push_back(table[(triple >> 12) & 0x3f]);
    out.push_back(i + 1 < size ? table[(triple >> 6) & 0x3f] : '=');
    out.push_back(i + 2 < size ? table[triple & 0x3f] : '=');
  }
  return out;
}

static std::string websocket_accept_key(const std::string& client_key) {
  const std::string guid = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
  const std::array<uint8_t, 20> digest = sha1_digest(client_key + guid);
  return base64_encode(digest.data(), digest.size());
}

static std::string lower_ascii(std::string value) {
  for (char& ch : value) {
    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  }
  return value;
}

static std::string get_http_header(const std::string& request, const std::string& name) {
  const std::string wanted = lower_ascii(name);
  size_t pos = 0;
  while (pos < request.size()) {
    const size_t end = request.find("\r\n", pos);
    const std::string line =
        request.substr(pos, end == std::string::npos ? std::string::npos : end - pos);
    const size_t colon = line.find(':');
    if (colon != std::string::npos) {
      std::string key = lower_ascii(trim(line.substr(0, colon)));
      if (key == wanted) {
        return trim(line.substr(colon + 1));
      }
    }
    if (end == std::string::npos) {
      break;
    }
    pos = end + 2;
  }
  return "";
}

static bool send_all(int fd, const uint8_t* data, size_t size) {
  size_t sent = 0;
  while (sent < size) {
    const ssize_t n = ::send(fd, data + sent, size - sent, MSG_NOSIGNAL);
    if (n <= 0) {
      if (errno == EINTR) {
        continue;
      }
      return false;
    }
    sent += static_cast<size_t>(n);
  }
  return true;
}

static bool send_all(int fd, const std::string& data) {
  return send_all(fd, reinterpret_cast<const uint8_t*>(data.data()), data.size());
}

static bool set_fd_nonblocking(int fd) {
  const int flags = fcntl(fd, F_GETFL, 0);
  if (flags < 0) {
    return false;
  }
  return fcntl(fd, F_SETFL, flags | O_NONBLOCK) == 0;
}

static void set_socket_send_buffer(int fd, int bytes) {
  setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &bytes, sizeof(bytes));
}

static void set_socket_recv_timeout(int fd, int timeout_ms) {
  timeval tv {};
  tv.tv_sec = timeout_ms / 1000;
  tv.tv_usec = static_cast<suseconds_t>((timeout_ms % 1000) * 1000);
  setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
}

static bool send_all_nonblocking(int fd,
                                 const uint8_t* data,
                                 size_t size,
                                 int timeout_ms = 50) {
  size_t sent = 0;
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
  while (sent < size) {
    const ssize_t n = ::send(fd, data + sent, size - sent,
                             MSG_NOSIGNAL | MSG_DONTWAIT);
    if (n > 0) {
      sent += static_cast<size_t>(n);
      continue;
    }
    if (n < 0 && errno == EINTR) {
      continue;
    }
    if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
      const auto now = std::chrono::steady_clock::now();
      if (now >= deadline) {
        return false;
      }
      const auto remaining_us =
          std::chrono::duration_cast<std::chrono::microseconds>(deadline - now).count();
      timeval tv {};
      tv.tv_sec = 0;
      tv.tv_usec = static_cast<suseconds_t>(
          std::max<int64_t>(1000, std::min<int64_t>(remaining_us, 5000)));
      fd_set wfds;
      FD_ZERO(&wfds);
      FD_SET(fd, &wfds);
      const int ret = select(fd + 1, nullptr, &wfds, nullptr, &tv);
      if (ret > 0) {
        continue;
      }
      if (ret < 0 && errno == EINTR) {
        continue;
      }
      continue;
    }
    return false;
  }
  return true;
}

static bool send_all_nonblocking(int fd,
                                 const std::vector<uint8_t>& data,
                                 int timeout_ms = 50) {
  return send_all_nonblocking(fd, data.data(), data.size(), timeout_ms);
}

static std::vector<uint8_t> make_websocket_frame(const uint8_t* data,
                                                 size_t size,
                                                 uint8_t opcode) {
  std::vector<uint8_t> frame;
  frame.reserve(size + 10);
  frame.push_back(static_cast<uint8_t>(0x80 | (opcode & 0x0f)));
  const uint64_t len = static_cast<uint64_t>(size);
  if (len <= 125) {
    frame.push_back(static_cast<uint8_t>(len));
  } else if (len <= 0xffff) {
    frame.push_back(126);
    frame.push_back(static_cast<uint8_t>((len >> 8) & 0xff));
    frame.push_back(static_cast<uint8_t>(len & 0xff));
  } else {
    frame.push_back(127);
    for (int i = 7; i >= 0; --i) {
      frame.push_back(static_cast<uint8_t>((len >> (i * 8)) & 0xff));
    }
  }
  frame.insert(frame.end(), data, data + size);
  return frame;
}

static int create_tcp_listener(const std::string& host, int port, int backlog) {
  struct addrinfo hints {};
  hints.ai_family = (host.empty() || host == "0.0.0.0") ? AF_INET : AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  hints.ai_flags = AI_PASSIVE;
  struct addrinfo* result = nullptr;
  const std::string port_text = std::to_string(port);
  const char* node = (host.empty() || host == "0.0.0.0") ? nullptr : host.c_str();
  const int gai = getaddrinfo(node, port_text.c_str(), &hints, &result);
  if (gai != 0) {
    throw std::runtime_error(std::string("getaddrinfo failed for bind: ") + gai_strerror(gai));
  }

  int fd = -1;
  for (struct addrinfo* rp = result; rp != nullptr; rp = rp->ai_next) {
    fd = ::socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
    if (fd < 0) {
      continue;
    }
    int one = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    if (::bind(fd, rp->ai_addr, rp->ai_addrlen) == 0 && ::listen(fd, backlog) == 0) {
      break;
    }
    ::close(fd);
    fd = -1;
  }
  freeaddrinfo(result);
  if (fd < 0) {
    throw std::runtime_error("cannot bind socket on " + host + ":" + std::to_string(port));
  }
  return fd;
}

static std::vector<std::string> local_ipv4_addresses() {
  std::vector<std::string> addrs;
  struct ifaddrs* ifaddr = nullptr;
  if (getifaddrs(&ifaddr) != 0) {
    return addrs;
  }
  for (struct ifaddrs* ifa = ifaddr; ifa != nullptr; ifa = ifa->ifa_next) {
    if (ifa->ifa_addr == nullptr || ifa->ifa_addr->sa_family != AF_INET) {
      continue;
    }
    if ((ifa->ifa_flags & IFF_LOOPBACK) != 0) {
      continue;
    }
    char host[INET_ADDRSTRLEN] = {0};
    const sockaddr_in* addr = reinterpret_cast<const sockaddr_in*>(ifa->ifa_addr);
    if (inet_ntop(AF_INET, &addr->sin_addr, host, sizeof(host)) != nullptr) {
      addrs.push_back(host);
    }
  }
  freeifaddrs(ifaddr);
  std::sort(addrs.begin(), addrs.end());
  addrs.erase(std::unique(addrs.begin(), addrs.end()), addrs.end());
  return addrs;
}

class BoardWebUiServer {
 public:
  BoardWebUiServer(const std::string& host, int port)
      : host_(host), port_(port) {}

  ~BoardWebUiServer() { stop(); }

  void start() {
    if (running_.load()) {
      return;
    }
    listen_fd_ = create_tcp_listener(host_, port_, 64);
    running_.store(true);
    sender_thread_ = std::thread(&BoardWebUiServer::sender_loop, this);
    accept_thread_ = std::thread(&BoardWebUiServer::accept_loop, this);
    std::cerr << "[board_cpp-webui] listening: http://" << host_ << ":" << port_ << "/\n";
    if (host_.empty() || host_ == "0.0.0.0") {
      const std::vector<std::string> addrs = local_ipv4_addresses();
      if (addrs.empty()) {
        std::cerr << "[board_cpp-webui] open: http://<board-ip>:" << port_ << "/\n";
      } else {
        for (const std::string& addr : addrs) {
          std::cerr << "[board_cpp-webui] open: http://" << addr << ":" << port_ << "/\n";
        }
      }
    } else {
      std::cerr << "[board_cpp-webui] open: http://" << host_ << ":" << port_ << "/\n";
    }
  }

  void stop() {
    if (!running_.load() && listen_fd_ < 0) {
      return;
    }
    running_.store(false);
    if (listen_fd_ >= 0) {
      ::shutdown(listen_fd_, SHUT_RDWR);
      ::close(listen_fd_);
      listen_fd_ = -1;
    }
    sender_cv_.notify_all();
    if (accept_thread_.joinable()) {
      accept_thread_.join();
    }
    if (sender_thread_.joinable()) {
      sender_thread_.join();
    }
	    std::vector<int> json_clients;
	    std::vector<int> system_clients;
	    std::vector<int> slot_clients;
	    std::vector<int> frame_clients;
	    std::vector<int> mjpeg_clients;
	    {
	      std::lock_guard<std::mutex> lock(mutex_);
	      json_clients.swap(json_clients_);
	      system_clients.swap(system_clients_);
	      slot_clients.swap(slot_clients_);
	      frame_clients.swap(frame_clients_);
	      mjpeg_clients.swap(mjpeg_clients_);
	    }
	    close_all(json_clients);
	    close_all(system_clients);
	    close_all(slot_clients);
	    close_all(frame_clients);
	    close_all(mjpeg_clients);
	  }

	  bool has_frame_clients() const {
	    std::lock_guard<std::mutex> lock(mutex_);
	    return !frame_clients_.empty() || !mjpeg_clients_.empty();
	  }

	  bool has_slot_clients() const {
	    std::lock_guard<std::mutex> lock(mutex_);
	    return !slot_clients_.empty();
	  }

  void publish_json(const std::string& payload) {
    if (!running_.load()) {
      return;
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      latest_json_ = payload;
      json_dirty_ = true;
    }
    sender_cv_.notify_one();
  }

	  void publish_system(const std::string& payload) {
    if (!running_.load() || payload.empty()) {
      return;
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      latest_system_ = payload;
      system_dirty_ = true;
    }
	    sender_cv_.notify_one();
	  }

	  void publish_slots(const std::string& payload) {
	    if (!running_.load() || payload.empty()) {
	      return;
	    }
	    {
	      std::lock_guard<std::mutex> lock(mutex_);
	      latest_slots_ = payload;
	      slots_dirty_ = true;
	    }
	    sender_cv_.notify_one();
	  }

	  void publish_jpeg(const std::vector<uint8_t>& jpeg) {
    if (!running_.load() || jpeg.empty()) {
      return;
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      latest_jpeg_ = jpeg;
      jpeg_dirty_ = true;
    }
    sender_cv_.notify_one();
  }

 private:
  static void close_all(const std::vector<int>& fds) {
    for (int fd : fds) {
      ::shutdown(fd, SHUT_RDWR);
      ::close(fd);
    }
  }

  void remove_failed(std::vector<int>& owned,
                     const std::vector<int>& candidates,
                     const std::vector<uint8_t>& payload,
                     int timeout_ms = 50) {
    std::vector<int> failed;
    for (int fd : candidates) {
      if (!send_all_nonblocking(fd, payload, timeout_ms)) {
        failed.push_back(fd);
      }
    }
    if (failed.empty()) {
      return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<int> kept;
    kept.reserve(owned.size());
    for (int fd : owned) {
      if (std::find(failed.begin(), failed.end(), fd) == failed.end()) {
        kept.push_back(fd);
      } else {
        ::shutdown(fd, SHUT_RDWR);
        ::close(fd);
      }
    }
    owned.swap(kept);
  }

  void sender_loop() {
    while (true) {
	      std::string json_payload;
	      std::string system_payload;
	      std::string slots_payload;
	      std::vector<uint8_t> jpeg_payload;
	      std::vector<int> json_clients;
	      std::vector<int> system_clients;
	      std::vector<int> slot_clients;
	      std::vector<int> frame_clients;
	      std::vector<int> mjpeg_clients;
	      bool send_json = false;
	      bool send_system = false;
	      bool send_slots = false;
	      bool send_jpeg = false;
	      {
	        std::unique_lock<std::mutex> lock(mutex_);
	        sender_cv_.wait(lock, [&] {
	          return !running_.load() || json_dirty_ || system_dirty_ || slots_dirty_ ||
	                 jpeg_dirty_;
	        });
	        if (!running_.load() && !json_dirty_ && !system_dirty_ && !slots_dirty_ &&
	            !jpeg_dirty_) {
	          break;
	        }
        if (json_dirty_) {
          json_payload = latest_json_;
          json_clients = json_clients_;
          json_dirty_ = false;
          send_json = !json_payload.empty() && !json_clients.empty();
        }
        if (system_dirty_) {
          system_payload = latest_system_;
          system_clients = system_clients_;
	          system_dirty_ = false;
	          send_system = !system_payload.empty() && !system_clients.empty();
	        }
	        if (slots_dirty_) {
	          slots_payload = latest_slots_;
	          slot_clients = slot_clients_;
	          slots_dirty_ = false;
	          send_slots = !slots_payload.empty() && !slot_clients.empty();
	        }
	        if (jpeg_dirty_) {
          jpeg_payload = latest_jpeg_;
          frame_clients = frame_clients_;
          mjpeg_clients = mjpeg_clients_;
          jpeg_dirty_ = false;
          send_jpeg = !jpeg_payload.empty() &&
                      (!frame_clients.empty() || !mjpeg_clients.empty());
        }
      }

      if (send_json) {
        std::vector<uint8_t> frame = make_websocket_frame(
            reinterpret_cast<const uint8_t*>(json_payload.data()), json_payload.size(), 0x2);
        remove_failed(json_clients_, json_clients, frame);
      }
	      if (send_system) {
	        std::vector<uint8_t> frame = make_websocket_frame(
	            reinterpret_cast<const uint8_t*>(system_payload.data()), system_payload.size(), 0x2);
	        remove_failed(system_clients_, system_clients, frame);
	      }
	      if (send_slots) {
	        std::vector<uint8_t> frame = make_websocket_frame(
	            reinterpret_cast<const uint8_t*>(slots_payload.data()), slots_payload.size(), 0x2);
	        remove_failed(slot_clients_, slot_clients, frame);
	      }
	      if (send_jpeg) {
        if (!frame_clients.empty()) {
          std::vector<uint8_t> frame = make_websocket_frame(
              jpeg_payload.data(), jpeg_payload.size(), 0x2);
          remove_failed(frame_clients_, frame_clients, frame);
        }
        if (!mjpeg_clients.empty()) {
          std::ostringstream header;
          header << "--frame\r\n"
                 << "Content-Type: image/jpeg\r\n"
                 << "Content-Length: " << jpeg_payload.size() << "\r\n\r\n";
          const std::string header_text = header.str();
          std::vector<uint8_t> chunk(header_text.begin(), header_text.end());
          chunk.insert(chunk.end(), jpeg_payload.begin(), jpeg_payload.end());
          chunk.push_back('\r');
          chunk.push_back('\n');
          remove_failed(mjpeg_clients_, mjpeg_clients, chunk, 200);
        }
      }
    }
  }

  void accept_loop() {
    while (running_.load()) {
      sockaddr_storage addr {};
      socklen_t addr_len = sizeof(addr);
      const int fd = ::accept(listen_fd_, reinterpret_cast<sockaddr*>(&addr), &addr_len);
      if (fd < 0) {
        if (running_.load() && errno != EINTR) {
          std::cerr << "[board_cpp-webui] accept failed: " << std::strerror(errno) << "\n";
        }
        continue;
      }
      set_socket_recv_timeout(fd, 1000);
      handle_client(fd);
    }
  }

  void handle_client(int fd) {
    std::string request;
    char buffer[1024];
    while (request.find("\r\n\r\n") == std::string::npos && request.size() < 8192) {
      const ssize_t n = ::recv(fd, buffer, sizeof(buffer), 0);
      if (n <= 0) {
        ::close(fd);
        return;
      }
      request.append(buffer, buffer + n);
    }

    const size_t first_line_end = request.find("\r\n");
    const std::string first_line =
        first_line_end == std::string::npos ? request : request.substr(0, first_line_end);
    std::istringstream iss(first_line);
    std::string method;
    std::string path;
    std::string version;
    iss >> method >> path >> version;
    if (method != "GET") {
      send_simple_response(fd, "405 Method Not Allowed", "text/plain", "method not allowed");
      ::close(fd);
      return;
    }
    const size_t query_pos = path.find('?');
    if (query_pos != std::string::npos) {
      path = path.substr(0, query_pos);
    }
    const std::string upgrade = get_http_header(request, "Upgrade");
    if (!upgrade.empty()) {
      if (path == "/ws/inference") {
        std::string latest;
        {
          std::lock_guard<std::mutex> lock(mutex_);
          latest = latest_json_;
        }
        accept_websocket(fd, request, json_clients_, latest);
        return;
      }
	      if (path == "/ws/system") {
	        std::string latest;
	        {
	          std::lock_guard<std::mutex> lock(mutex_);
	          latest = latest_system_;
	        }
	        accept_websocket(fd, request, system_clients_, latest);
	        return;
	      }
	      if (path == "/ws/slots") {
	        std::string latest;
	        {
	          std::lock_guard<std::mutex> lock(mutex_);
	          latest = latest_slots_;
	        }
	        std::cerr << "[board_cpp-webui] slot WebSocket client connected\n";
	        accept_websocket(fd, request, slot_clients_, latest);
	        return;
	      }
	      if (path == "/ws/frame") {
	        accept_websocket(fd, request, frame_clients_, std::string());
	        return;
      }
      send_simple_response(fd, "404 Not Found", "text/plain", "not found");
      ::close(fd);
      return;
    }
    if (path == "/" || path == "/index.html") {
      send_simple_response(fd, "200 OK", "text/html; charset=utf-8", index_html());
      ::close(fd);
      return;
    }
    if (path == "/inference/latest") {
      std::string payload;
      {
        std::lock_guard<std::mutex> lock(mutex_);
        payload = latest_json_;
      }
      if (payload.empty()) {
        send_simple_response(fd, "503 Service Unavailable", "application/json",
                             "{\"error\":\"no inference result yet\"}");
      } else {
        send_simple_response(fd, "200 OK", "application/json", payload);
      }
      ::close(fd);
      return;
    }
	    if (path == "/system/latest") {
      std::string payload;
      {
        std::lock_guard<std::mutex> lock(mutex_);
        payload = latest_system_;
      }
      if (payload.empty()) {
        send_simple_response(fd, "503 Service Unavailable", "application/json",
                             "{\"error\":\"no system load sample yet\"}");
      } else {
        send_simple_response(fd, "200 OK", "application/json", payload);
      }
      ::close(fd);
	      return;
	    }
	    if (path == "/slots/latest") {
	      std::string payload;
	      {
	        std::lock_guard<std::mutex> lock(mutex_);
	        payload = latest_slots_;
	      }
	      if (payload.empty()) {
	        send_simple_response(fd, "503 Service Unavailable", "application/json",
	                             "{\"error\":\"no display slot crop yet\"}");
	      } else {
	        send_simple_response(fd, "200 OK", "application/json", payload);
	      }
	      ::close(fd);
	      return;
	    }
	    if (path == "/video/infer") {
      const std::string header =
          "HTTP/1.1 200 OK\r\n"
          "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n"
          "Cache-Control: no-store\r\n"
          "Connection: close\r\n\r\n";
      if (!send_all(fd, header)) {
        ::close(fd);
        return;
      }
      if (!set_fd_nonblocking(fd)) {
        ::close(fd);
        return;
      }
      set_socket_send_buffer(fd, 1024 * 1024);
      {
        std::lock_guard<std::mutex> lock(mutex_);
        mjpeg_clients_.push_back(fd);
      }
      return;
    }
    send_simple_response(fd, "404 Not Found", "text/plain", "not found");
    ::close(fd);
  }

  bool accept_websocket(int fd,
                        const std::string& request,
                        std::vector<int>& client_list,
                        const std::string& latest_payload) {
    const std::string key = get_http_header(request, "Sec-WebSocket-Key");
    if (key.empty()) {
      send_simple_response(fd, "400 Bad Request", "text/plain", "missing websocket key");
      ::close(fd);
      return false;
    }
    std::ostringstream response;
    response << "HTTP/1.1 101 Switching Protocols\r\n"
             << "Upgrade: websocket\r\n"
             << "Connection: Upgrade\r\n"
             << "Sec-WebSocket-Accept: " << websocket_accept_key(key) << "\r\n"
             << "\r\n";
    if (!send_all(fd, response.str())) {
      ::close(fd);
      return false;
    }
    if (!set_fd_nonblocking(fd)) {
      ::close(fd);
      return false;
    }
    set_socket_send_buffer(fd, 1024 * 1024);
    {
      std::lock_guard<std::mutex> lock(mutex_);
      client_list.push_back(fd);
    }
    if (!latest_payload.empty()) {
      std::vector<uint8_t> frame = make_websocket_frame(
          reinterpret_cast<const uint8_t*>(latest_payload.data()), latest_payload.size(), 0x2);
      if (!send_all_nonblocking(fd, frame)) {
        {
          std::lock_guard<std::mutex> lock(mutex_);
          client_list.erase(std::remove(client_list.begin(), client_list.end(), fd),
                            client_list.end());
        }
        ::shutdown(fd, SHUT_RDWR);
        ::close(fd);
      }
    }
    return true;
  }

  static void send_simple_response(int fd,
                                   const std::string& status,
                                   const std::string& content_type,
                                   const std::string& body) {
    std::ostringstream os;
    os << "HTTP/1.1 " << status << "\r\n"
       << "Content-Type: " << content_type << "\r\n"
       << "Content-Length: " << body.size() << "\r\n"
       << "Cache-Control: no-store\r\n"
       << "Connection: close\r\n\r\n"
       << body;
    send_all(fd, os.str());
  }

  static std::string index_html() {
    return R"HTML(<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MeetEye C++ WebUI</title>
  <style>
    :root{
      --bg:#0c1117;
      --panel:#151b23;
      --panel-2:#10161d;
      --panel-3:#0a0f14;
      --line:#273241;
      --line-soft:#1f2937;
      --text:#e8edf3;
      --muted:#8c9aaa;
      --muted-2:#637084;
      --cyan:#42c7e8;
      --green:#45d483;
      --amber:#f0b35a;
      --red:#f06b5f;
      --violet:#a78bfa;
      --shadow:0 14px 32px rgba(0,0,0,.28);
    }
    *{box-sizing:border-box;}
    body{margin:0;background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei",Arial,sans-serif;}
    header{min-height:62px;display:grid;grid-template-columns:auto minmax(0,1fr);gap:16px;align-items:center;padding:10px 16px;background:#111821;border-bottom:1px solid var(--line);}
    .brand{display:flex;align-items:center;gap:10px;min-width:210px;}
    .brand-mark{width:34px;height:34px;border:1px solid #315060;background:#0b1b23;color:#7dd3fc;display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:800;}
    h1{font-size:17px;margin:0;font-weight:700;letter-spacing:0;}
    .brand-sub{margin-top:1px;color:var(--muted);font-size:11px;white-space:nowrap;}
    .top-strip{display:grid;grid-template-columns:repeat(5,minmax(82px,1fr));gap:8px;align-items:stretch;}
    .top-metric{background:var(--panel-2);border:1px solid var(--line);padding:7px 9px;min-height:42px;}
    .top-metric span{display:block;color:var(--muted);font-size:10px;white-space:nowrap;margin-bottom:2px;}
    .top-metric b{display:block;color:var(--text);font-size:15px;line-height:1.15;white-space:nowrap;}
    .status{color:var(--green)!important;}
    main{display:grid;grid-template-columns:minmax(0,1fr) 360px;gap:12px;padding:12px;height:calc(100vh - 63px);}
    .left-pane{display:grid;grid-template-rows:auto auto;gap:12px;min-width:0;min-height:0;align-content:start;}
    .panel{background:var(--panel);border:1px solid var(--line);box-shadow:var(--shadow);}
    .panel-head,.slot-head{height:38px;display:flex;align-items:center;justify-content:space-between;gap:10px;padding:0 12px;border-bottom:1px solid var(--line-soft);}
    h2{font-size:12px;letter-spacing:.02em;margin:0;color:#f3f6fa;font-weight:700;}
    .panel-note,.slot-meta{font-size:10px;color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
    .view-card{min-height:0;display:flex;flex-direction:column;}
    .view{position:relative;width:100%;aspect-ratio:var(--stream-aspect,32/9);background:#030506;display:flex;align-items:center;justify-content:center;overflow:hidden;}
    img{width:100%;height:100%;display:block;object-fit:contain;}
    .slot-panel{background:var(--panel);border:1px solid var(--line);box-shadow:var(--shadow);padding-bottom:8px;}
    .display-slots{display:grid;grid-template-columns:repeat(auto-fit,minmax(172px,1fr));gap:7px;padding:8px 10px 0;}
    .display-slot{min-height:76px;background:var(--panel-2);border:1px solid var(--line-soft);padding:7px;display:grid;grid-template-columns:44px minmax(0,1fr);gap:8px;align-items:center;}
    .display-slot.active{border-color:#2d93aa;background:#0f2029;box-shadow:inset 3px 0 0 var(--cyan);}
    .display-slot.empty{opacity:.62;}
    .slot-face{width:44px;height:44px;background:#0b1118;border:1px solid #334155;overflow:hidden;display:flex;align-items:center;justify-content:center;color:#64748b;font-size:10px;font-weight:800;}
    .slot-face img{width:100%;height:100%;object-fit:cover;}
    .slot-id{display:flex;align-items:center;justify-content:space-between;gap:6px;font-size:12px;font-weight:800;color:#f8fafc;white-space:nowrap;}
    .slot-id span:last-child{font-size:10px;color:#7dd3fc;font-weight:700;overflow:hidden;text-overflow:ellipsis;}
    .slot-info{min-width:0;}
    .slot-row{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:3px 7px;margin-top:5px;}
    .slot-stat{min-width:0;display:grid;grid-template-columns:12px minmax(0,1fr);align-items:baseline;gap:3px;white-space:nowrap;}
    .slot-stat i{font-style:normal;color:#8fa0b4;font-size:9px;line-height:1;text-align:right;}
    .slot-stat b{display:block;min-width:0;font-size:10px;line-height:1.05;font-weight:800;color:#f8fafc;font-variant-numeric:tabular-nums;overflow:hidden;text-overflow:ellipsis;}
    .slot-muted{font-size:9px;color:var(--muted-2);margin-top:3px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
    aside{display:flex;flex-direction:column;gap:12px;overflow:auto;min-height:0;}
    .side-section{background:var(--panel);border:1px solid var(--line);box-shadow:var(--shadow);}
    .side-section .content{padding:10px;}
    .grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px;}
    .card{background:var(--panel-2);border:1px solid var(--line-soft);padding:8px;min-height:58px;}
    .label{font-size:10px;color:var(--muted);margin-bottom:4px;white-space:nowrap;}
    .value{font-size:15px;font-weight:800;color:#f9fafb;white-space:nowrap;}
    .bar{height:4px;background:#273241;margin-top:7px;overflow:hidden;}
    .fill{height:100%;width:0;background:var(--cyan);}
    .warn .fill{background:var(--amber);}
    .hot .fill{background:var(--red);}
    .cores{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:6px;}
    .core{font-size:10px;color:#cbd5e1;background:var(--panel-2);border:1px solid var(--line-soft);padding:5px;text-align:center;}
    .sector-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:6px;}
    .sector{font-size:10px;line-height:1.28;color:var(--muted);background:var(--panel-2);border:1px solid var(--line-soft);padding:6px;text-align:center;min-height:42px;}
    .sector.on{color:#ffd9d4;border-color:#8d3a32;background:#2a1212;box-shadow:inset 3px 0 0 var(--red);}
    details.debug-json{background:var(--panel);border:1px solid var(--line);box-shadow:var(--shadow);}
    details.debug-json summary{height:36px;display:flex;align-items:center;padding:0 12px;cursor:pointer;color:#f3f6fa;font-size:12px;font-weight:700;border-bottom:1px solid var(--line-soft);}
    pre{white-space:pre-wrap;word-break:break-word;font-size:10px;line-height:1.35;color:#cbd5e1;margin:0;padding:10px;max-height:300px;overflow:auto;background:#0b1016;}
    @media(max-width:1100px){main{grid-template-columns:1fr;height:auto;}aside{overflow:visible;}.top-strip{grid-template-columns:repeat(3,minmax(82px,1fr));}.display-slots{grid-template-columns:repeat(2,minmax(0,1fr));}}
    @media(max-width:720px){header{grid-template-columns:1fr;}.top-strip{grid-template-columns:repeat(2,minmax(0,1fr));}main{padding:8px;}.grid{grid-template-columns:1fr;}}
  </style>
</head>
<body>
  <header>
    <div class="brand">
      <div class="brand-mark">ME</div>
      <div>
        <h1>MeetEye Board</h1>
        <div class="brand-sub">RK3588 real-time inference</div>
      </div>
    </div>
    <div class="top-strip">
      <div class="top-metric"><span>连接</span><b class="status" id="status">连接中...</b></div>
      <div class="top-metric"><span>帧号</span><b id="frame">-</b></div>
      <div class="top-metric"><span>目标数</span><b id="count">-</b></div>
      <div class="top-metric"><span>FPS</span><b id="fps">-</b></div>
      <div class="top-metric"><span>JSON / SYS</span><b><span id="ws">-</span> / <span id="sysws">-</span></b></div>
    </div>
	  </header>
	  <main>
	    <section class="left-pane">
	      <section class="panel view-card">
	        <div class="panel-head">
	          <h2>实时画面</h2>
	          <div class="panel-note">WebUI frame stream</div>
	        </div>
	        <section class="view"><img id="infer" alt="等待推理画面"></section>
	      </section>
	      <section class="slot-panel">
	        <div class="slot-head">
	          <h2>显示槽</h2>
	          <div class="slot-meta" id="slotMeta">等待目标</div>
	        </div>
	        <div class="display-slots" id="displaySlots"></div>
	      </section>
	    </section>
	    <aside>
      <section class="side-section">
        <div class="panel-head"><h2>硬件负载</h2><div class="panel-note">200ms sample</div></div>
        <div class="content">
          <div class="grid">
            <div class="card"><div class="label">CPU 总占用</div><div class="value" id="cpu">-</div><div class="bar"><div id="cpu-bar" class="fill"></div></div></div>
            <div class="card"><div class="label">GPU 占用</div><div class="value" id="gpu">-</div><div class="bar"><div id="gpu-bar" class="fill"></div></div></div>
            <div class="card"><div class="label">NPU Core0</div><div class="value" id="npu0">-</div><div class="bar"><div id="npu0-bar" class="fill"></div></div></div>
            <div class="card"><div class="label">NPU Core1</div><div class="value" id="npu1">-</div><div class="bar"><div id="npu1-bar" class="fill"></div></div></div>
            <div class="card"><div class="label">NPU Core2</div><div class="value" id="npu2">-</div><div class="bar"><div id="npu2-bar" class="fill"></div></div></div>
            <div class="card"><div class="label">内存占用</div><div class="value" id="mem">-</div><div class="bar"><div id="mem-bar" class="fill"></div></div></div>
            <div class="card"><div class="label">最高温度</div><div class="value" id="thermal">-</div><div class="bar"><div id="thermal-bar" class="fill"></div></div></div>
            <div class="card"><div class="label">可用内存</div><div class="value" id="memfree">-</div></div>
          </div>
        </div>
      </section>
      <section class="side-section">
        <div class="panel-head"><h2>CPU 核心</h2><div class="panel-note">per-core</div></div>
        <div class="content"><div class="cores" id="cores"></div></div>
      </section>
      <details class="debug-json">
        <summary>推理 JSON</summary>
        <pre id="json">{}</pre>
      </details>
    </aside>
  </main>
  <script>
    const statusEl=document.getElementById('status');
    const wsEl=document.getElementById('ws');
    const frameEl=document.getElementById('frame');
    const countEl=document.getElementById('count');
    const fpsEl=document.getElementById('fps');
    const sysWsEl=document.getElementById('sysws');
	    const jsonEl=document.getElementById('json');
	    const coresEl=document.getElementById('cores');
	    const displaySlotsEl=document.getElementById('displaySlots');
	    const slotMetaEl=document.getElementById('slotMeta');
	    const inferImg=document.getElementById('infer');
	    let inferObjectUrl=null;
		    let configuredSlotCount=8;
		    let latestTargetsBySlot=new Map();
		    let latestFrameId=0;
		    const slotState=new Map();
		    let configuredSlotFps=10;
	    function fmtPct(v){return Number.isFinite(v)?`${v.toFixed(1)}%`:'-';}
	    function fmtFreq(v){return Number.isFinite(v)&&v>0?`${(v/1000000).toFixed(0)}MHz`:'';}
    function fmtDeg(v){
      if(v===null||v===undefined)return '-';
	      const n=Number(v);
	      return Number.isFinite(n)?`${n.toFixed(0)}°`:'-';
	    }
	    function fmtMeters(v){
	      if(v===null||v===undefined)return '-';
	      const n=Number(v);
	      return Number.isFinite(n)?`${n.toFixed(2)}m`:'-';
	    }
	    function escapeHtml(value){
	      return String(value).replace(/[&<>"']/g,ch=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
	    }
	    function targetDisplayId(target,key){
	      const id=Number(target&&target.display_id ? target.display_id : target&&target.id ? target.id : key);
	      return Number.isFinite(id)&&id>0?Math.floor(id):0;
	    }
	    function targetFaceId(target){
	      const rec=target&&target.face_recognition;
	      if(rec&&rec.matched&&rec.known_name)return String(rec.known_name);
	      if(target&&target.known_name)return String(target.known_name);
	      if(target&&target.face_id)return String(target.face_id);
	      if(rec&&rec.matched&&rec.face_id)return String(rec.face_id);
	      return 'null';
	    }
	    function targetPublicId(target){
	      const id=Number(target&&target.public_id);
	      return Number.isFinite(id)&&id>0?Math.floor(id):0;
	    }
		    function targetRawTrackId(target){
		      const id=Number(target&&target.raw_track_id);
		      return Number.isFinite(id)&&id>0?Math.floor(id):0;
		    }
		    function slotFpsLabel(){
		      return Number.isFinite(configuredSlotFps)&&configuredSlotFps>0
		        ? configuredSlotFps.toFixed(configuredSlotFps>=10?0:1)
		        : '10';
		    }
			    function renderDisplaySlots(){
	      const count=Math.max(1,Math.min(32,configuredSlotCount||8));
		      slotMetaEl.textContent=`固定 ${count} 槽，槽图按 ${slotFpsLabel()} FPS 实时低频发送`;
	      const cards=[];
	      for(let slot=1;slot<=count;++slot){
	        const target=latestTargetsBySlot.get(slot);
	        let state=slotState.get(slot);
	        if(!state){state={};slotState.set(slot,state);}
	        const active=!!target;
	        const face=active?targetFaceId(target):(state.faceId||'null');
	        const publicValue=active?targetPublicId(target):(state.publicId||0);
	        const rawValue=active?targetRawTrackId(target):(state.rawTrackId||0);
	        if(active&&state.image){
	          const publicMismatch=state.publicId>0&&publicValue>0&&state.publicId!==publicValue;
	          const rawMismatch=state.rawTrackId>0&&rawValue>0&&state.rawTrackId!==rawValue;
	          if(publicMismatch||rawMismatch){
	            state.image=null;
	            state.faceId='null';
	            state.publicId=publicValue;
	            state.rawTrackId=rawValue;
	          }
	        }
	        const publicId=active&&publicValue>0?`P${publicValue}`:'';
	        const rawId=active&&rawValue>0?`R${rawValue}`:'';
	        const displayLabel=active?targetDisplayId(target,slot):slot;
	        const img=state.image?`<img src="${state.image}" alt="ID${displayLabel}">`:`ID${displayLabel}`;
	        cards.push(`<div class="display-slot ${active?'active':'empty'}">
	          <div class="slot-face">${img}</div>
	          <div class="slot-info">
	            <div class="slot-id"><span>ID${displayLabel}</span><span>${escapeHtml(face)}</span></div>
	            ${active?`<div class="slot-row">
	              <span class="slot-stat"><i>A</i><b>${fmtDeg(target.azimuth)}</b></span>
	              <span class="slot-stat"><i>E</i><b>${fmtDeg(target.elevation)}</b></span>
	              <span class="slot-stat"><i>D</i><b>${fmtMeters(target.distance)}</b></span>
	              <span class="slot-stat"><i>Y</i><b>${fmtDeg(target.face_yaw)}</b></span>
	            </div>
	            <div class="slot-muted">${escapeHtml([publicId,rawId].filter(Boolean).join(' '))}</div>`:
		            `<div class="slot-muted">空闲</div>`}
	          </div>
	        </div>`);
	      }
	      displaySlotsEl.innerHTML=cards.join('');
	    }
		    function updateSlotsFromPayload(obj){
		      if(!obj||!Array.isArray(obj.slots))return;
		      const maxCount=Number(obj.display_id_max_count);
		      if(Number.isFinite(maxCount)&&maxCount>0){
		        configuredSlotCount=Math.floor(maxCount);
		      }
		      const slotFps=Number(obj.slot_fps);
		      if(Number.isFinite(slotFps)&&slotFps>0){
		        configuredSlotFps=slotFps;
		      }
		      const seen=new Set();
		      obj.slots.forEach(slot=>{
		        const id=Number(slot.display_id);
		        if(!Number.isFinite(id)||id<=0)return;
		        seen.add(Math.floor(id));
		        let state=slotState.get(id);
		        if(!state){state={};slotState.set(id,state);}
		        state.image=slot.image||null;
		        state.faceId=slot.face_id||'null';
		        state.publicId=Number(slot.public_id)||0;
		        state.rawTrackId=Number(slot.raw_track_id)||0;
		      });
		      for(const id of Array.from(slotState.keys())){
		        if(id>0&&!seen.has(id)){
		          slotState.delete(id);
		        }
		      }
		      renderDisplaySlots();
		    }
		    function updateDisplaySlotsFromTargets(entries,obj){
	      const maxCount=Number(obj.display_id_max_count);
	      const fallbackLimit=Number(obj.target_output_limit);
	      configuredSlotCount=Number.isFinite(maxCount)&&maxCount>0
	        ? Math.floor(maxCount)
	        : (Number.isFinite(fallbackLimit)&&fallbackLimit>0?Math.floor(fallbackLimit):Math.max(8,entries.length));
	      latestTargetsBySlot=new Map();
	      entries.forEach(([key,target])=>{
	        const slot=targetDisplayId(target,key);
	        if(slot>0&&slot<=configuredSlotCount){
	          latestTargetsBySlot.set(slot,target);
	        }
	      });
	      renderDisplaySlots();
	    }
		    function setMetric(id,barId,value,suffix='%',extra=''){
	      const el=document.getElementById(id);
	      const bar=document.getElementById(barId);
	      if(!Number.isFinite(value)){el.textContent='-'; if(bar)bar.style.width='0%'; return;}
	      el.textContent=`${value.toFixed(1)}${suffix}${extra}`;
	      if(bar)bar.style.width=`${Math.max(0,Math.min(100,value))}%`;
	    }
	    function npuCoreValue(obj,index){
	      if(Array.isArray(obj.npu_core_percent)&&Number.isFinite(Number(obj.npu_core_percent[index]))){
	        return Number(obj.npu_core_percent[index]);
	      }
	      return NaN;
	    }
    function connect(){
      const proto=location.protocol==='https:'?'wss':'ws';
      const ws=new WebSocket(`${proto}://${location.host}/ws/inference`);
      ws.binaryType='arraybuffer';
      ws.onopen=()=>{statusEl.textContent='已连接';wsEl.textContent='online';};
      ws.onclose=()=>{statusEl.textContent='已断开，重连中...';wsEl.textContent='offline';setTimeout(connect,1000);};
      ws.onerror=()=>{statusEl.textContent='连接异常';};
      ws.onmessage=(ev)=>{
        const decode=(data)=> data instanceof ArrayBuffer ? new TextDecoder().decode(data) : data;
        const text=decode(ev.data);
        try{
	          const obj=JSON.parse(text);
	          latestFrameId=Number(obj.frame_id)||latestFrameId;
	          frameEl.textContent=obj.frame_id ?? '-';
	          if(obj.fps && Number.isFinite(obj.fps.average)){fpsEl.textContent=obj.fps.average.toFixed(1);}
	          if(obj.targets){
	            const entries=Object.entries(obj.targets).sort((a,b)=>Number(a[0])-Number(b[0]));
	            countEl.textContent=entries.length;
	            updateDisplaySlotsFromTargets(entries,obj);
          }
	          else if(obj.sectors){
	            latestTargetsBySlot=new Map();
	            renderDisplaySlots();
            countEl.textContent=Object.values(obj.sectors).filter(s=>s.has_target).length;
          }
          jsonEl.textContent=JSON.stringify(obj,null,2);
        }catch(e){jsonEl.textContent=text;}
      };
    }
    let frameImageBusy=false;
    let pendingFrameBuffer=null;
    function pumpFrameImage(){
      if(frameImageBusy || !pendingFrameBuffer)return;
      frameImageBusy=true;
      const buffer=pendingFrameBuffer;
      pendingFrameBuffer=null;
      const nextUrl=URL.createObjectURL(new Blob([buffer],{type:'image/jpeg'}));
	      const prevUrl=inferObjectUrl;
	      inferImg.onload=()=>{
	        if(inferImg.naturalWidth>0&&inferImg.naturalHeight>0){
	          document.documentElement.style.setProperty('--stream-aspect',`${inferImg.naturalWidth}/${inferImg.naturalHeight}`);
	        }
	        if(prevUrl)URL.revokeObjectURL(prevUrl);
	        frameImageBusy=false;
	        pumpFrameImage();
	      };
      inferImg.onerror=()=>{
        URL.revokeObjectURL(nextUrl);
        frameImageBusy=false;
        pumpFrameImage();
      };
      inferObjectUrl=nextUrl;
      inferImg.src=nextUrl;
    }
	    function connectFrame(){
      const proto=location.protocol==='https:'?'wss':'ws';
      const ws=new WebSocket(`${proto}://${location.host}/ws/frame`);
      ws.binaryType='arraybuffer';
      ws.onclose=()=>{setTimeout(connectFrame,1000);};
      ws.onerror=()=>{try{ws.close();}catch(e){};};
      ws.onmessage=(ev)=>{
        if(!(ev.data instanceof ArrayBuffer))return;
        pendingFrameBuffer=ev.data;
        pumpFrameImage();
	      };
	    }
	    function connectSlots(){
	      const proto=location.protocol==='https:'?'wss':'ws';
	      const ws=new WebSocket(`${proto}://${location.host}/ws/slots`);
	      ws.binaryType='arraybuffer';
	      ws.onclose=()=>{setTimeout(connectSlots,1000);};
	      ws.onerror=()=>{try{ws.close();}catch(e){};};
	      ws.onmessage=(ev)=>{
	        const decode=(data)=> data instanceof ArrayBuffer ? new TextDecoder().decode(data) : data;
	        try{updateSlotsFromPayload(JSON.parse(decode(ev.data)));}catch(e){}
	      };
	    }
    function connectSystem(){
      const proto=location.protocol==='https:'?'wss':'ws';
      const ws=new WebSocket(`${proto}://${location.host}/ws/system`);
      ws.binaryType='arraybuffer';
      ws.onopen=()=>{sysWsEl.textContent='online';};
      ws.onclose=()=>{sysWsEl.textContent='offline';setTimeout(connectSystem,1000);};
      ws.onerror=()=>{sysWsEl.textContent='error';};
      ws.onmessage=(ev)=>{
        const decode=(data)=> data instanceof ArrayBuffer ? new TextDecoder().decode(data) : data;
        try{
	          const obj=JSON.parse(decode(ev.data));
	          setMetric('cpu','cpu-bar',obj.cpu_percent);
	          const npuFreq=fmtFreq(obj.npu_freq_hz);
	          setMetric('npu0','npu0-bar',npuCoreValue(obj,0),'%',npuFreq?` ${npuFreq}`:'');
	          setMetric('npu1','npu1-bar',npuCoreValue(obj,1));
	          setMetric('npu2','npu2-bar',npuCoreValue(obj,2));
	          setMetric('gpu','gpu-bar',obj.gpu_percent,'%',fmtFreq(obj.gpu_freq_hz)?` ${fmtFreq(obj.gpu_freq_hz)}`:'');
          setMetric('mem','mem-bar',obj.memory_percent);
          setMetric('thermal','thermal-bar',obj.thermal_max_c,'°C');
          document.getElementById('memfree').textContent=Number.isFinite(obj.memory_available_mb)?`${obj.memory_available_mb.toFixed(0)}MB`:'-';
          if(Array.isArray(obj.cpu_per_core_percent)){
            coresEl.innerHTML=obj.cpu_per_core_percent.map((v,i)=>`<div class="core">C${i} <b>${fmtPct(v)}</b></div>`).join('');
          }
        }catch(e){}
      };
    }
	    connect();
	    connectSystem();
	    connectSlots();
	    connectFrame();
  </script>
</body>
</html>)HTML";
  }

  std::string host_;
  int port_ = 0;
  int listen_fd_ = -1;
  std::atomic<bool> running_{false};
  std::thread accept_thread_;
  std::thread sender_thread_;
  mutable std::mutex mutex_;
	  std::condition_variable sender_cv_;
	  std::vector<int> json_clients_;
	  std::vector<int> system_clients_;
	  std::vector<int> slot_clients_;
	  std::vector<int> frame_clients_;
	  std::vector<int> mjpeg_clients_;
	  std::string latest_json_;
	  std::string latest_system_;
	  std::string latest_slots_;
	  std::vector<uint8_t> latest_jpeg_;
	  bool json_dirty_ = false;
	  bool system_dirty_ = false;
	  bool slots_dirty_ = false;
	  bool jpeg_dirty_ = false;
	};

class AsyncWebUiFramePublisher {
 public:
  AsyncWebUiFramePublisher(const Config& cfg,
                           const MeetEyeRuntime& runtime,
                           BoardWebUiServer* webui_server)
      : cfg_(cfg), runtime_(runtime), webui_server_(webui_server) {}

  ~AsyncWebUiFramePublisher() { stop(); }

  void start() {
    if (running_.load()) {
      return;
    }
    running_.store(true);
    worker_ = std::thread(&AsyncWebUiFramePublisher::run_loop, this);
  }

  void stop() {
    if (!running_.load()) {
      return;
    }
    running_.store(false);
    cv_.notify_all();
    if (worker_.joinable()) {
      worker_.join();
    }
  }

  bool submit(const Image& source,
              const FrameResult& result,
              const std::vector<TargetInfo>& targets,
              const FrameRateStats& fps) {
    if (!running_.load() || webui_server_ == nullptr ||
        (!webui_server_->has_frame_clients() && !webui_server_->has_slot_clients())) {
      return false;
    }
    std::unique_ptr<Job> job(new Job);
    job->source = source;
    job->result = result;
    job->targets = targets;
    job->fps = fps;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      latest_job_ = std::move(job);
    }
    cv_.notify_one();
    return true;
  }

 private:
	  struct Job {
	    Image source;
	    FrameResult result;
	    std::vector<TargetInfo> targets;
	    FrameRateStats fps;
	  };

		  struct DisplaySlotCropState {
		    int display_id = 0;
		    int public_id = 0;
		    int raw_track_id = 0;
		    int frame_id = 0;
		    double front_score = -1.0;
		    std::string face_id;
		    std::string image_data_url;
		  };

	  int display_slot_limit() const {
	    if (cfg_.display_id_max_count > 0) {
	      return std::min(cfg_.display_id_max_count, 32);
	    }
	    if (cfg_.target_output_limit > 0) {
	      return std::min(cfg_.target_output_limit, 32);
	    }
	    return 8;
	  }

	  static std::string target_face_name(const TargetInfo& target) {
	    if (target.has_known_face && !target.known_name.empty()) {
	      return target.known_name;
	    }
	    return target.has_face_recognition && !target.face_id.empty() ? target.face_id : "null";
	  }

	  static double target_bbox_area(const float* det) {
	    if (det == nullptr) {
	      return 0.0;
	    }
	    const double w = static_cast<double>(det[2] - det[0]);
	    const double h = static_cast<double>(det[3] - det[1]);
	    return std::isfinite(w) && std::isfinite(h) && w > 0.0 && h > 0.0 ? w * h : 0.0;
	  }

	  static Image resize_nearest(const Image& src, double scale) {
	    if (scale >= 0.999 || src.width <= 0 || src.height <= 0 ||
	        src.bgr.size() != static_cast<size_t>(src.width) * src.height * 3) {
	      return src;
	    }
	    const int dst_w = std::max(1, static_cast<int>(std::lround(src.width * scale)));
	    const int dst_h = std::max(1, static_cast<int>(std::lround(src.height * scale)));
	    if (dst_w == src.width && dst_h == src.height) {
	      return src;
	    }
	    Image dst;
	    dst.width = dst_w;
	    dst.height = dst_h;
	    dst.bgr.assign(static_cast<size_t>(dst_w) * dst_h * 3, 0);
	    for (int y = 0; y < dst_h; ++y) {
	      const int sy = std::max(0, std::min(src.height - 1,
	          static_cast<int>((static_cast<double>(y) + 0.5) / scale)));
	      for (int x = 0; x < dst_w; ++x) {
	        const int sx = std::max(0, std::min(src.width - 1,
	            static_cast<int>((static_cast<double>(x) + 0.5) / scale)));
	        const size_t src_idx = (static_cast<size_t>(sy) * src.width + sx) * 3;
	        const size_t dst_idx = (static_cast<size_t>(y) * dst_w + x) * 3;
	        dst.bgr[dst_idx + 0] = src.bgr[src_idx + 0];
	        dst.bgr[dst_idx + 1] = src.bgr[src_idx + 1];
	        dst.bgr[dst_idx + 2] = src.bgr[src_idx + 2];
	      }
	    }
	    return dst;
	  }

	  static double target_front_score(const TargetInfo& target, const float* det) {
	    const double area_score = std::min(200.0, std::sqrt(std::max(0.0, target_bbox_area(det))));
	    if (target.has_face_yaw && std::isfinite(target.face_yaw)) {
	      return std::max(0.0, 90.0 - std::abs(target.face_yaw)) * 10.0 + area_score;
	    }
	    return area_score;
	  }

	  bool build_slot_crop(const Image& clean,
	                       const float* det,
	                       Image* crop) const {
	    if (crop == nullptr || det == nullptr || clean.width <= 0 || clean.height <= 0 ||
	        clean.bgr.size() != static_cast<size_t>(clean.width) * clean.height * 3) {
	      return false;
	    }
	    const float x1 = det[0];
	    const float y1 = det[1];
	    const float x2 = det[2];
	    const float y2 = det[3];
	    const float w = x2 - x1;
	    const float h = y2 - y1;
	    if (!std::isfinite(w) || !std::isfinite(h) || w < 4.0f || h < 4.0f) {
	      return false;
	    }
	    const int crop_size = 64;
	    const float cx = (x1 + x2) * 0.5f;
	    const float cy = (y1 + y2) * 0.5f;
	    const float side = std::max(w, h) * 1.42f;
	    if (!std::isfinite(side) || side < 4.0f) {
	      return false;
	    }
	    crop->width = crop_size;
	    crop->height = crop_size;
	    crop->bgr.assign(static_cast<size_t>(crop_size) * crop_size * 3, 18);
	    for (int oy = 0; oy < crop_size; ++oy) {
	      for (int ox = 0; ox < crop_size; ++ox) {
	        float sx_f = cx - side * 0.5f +
	                     (static_cast<float>(ox) + 0.5f) * side / crop_size;
	        float sy_f = cy - side * 0.5f +
	                     (static_cast<float>(oy) + 0.5f) * side / crop_size;
	        int sx = static_cast<int>(std::lround(sx_f));
	        int sy = static_cast<int>(std::lround(sy_f));
	        if (cfg_.wrap_boundary_geometry && clean.width > 1) {
	          sx %= clean.width;
	          if (sx < 0) {
	            sx += clean.width;
	          }
	        } else {
	          sx = std::max(0, std::min(sx, clean.width - 1));
	        }
	        sy = std::max(0, std::min(sy, clean.height - 1));
	        const size_t src_idx = (static_cast<size_t>(sy) * clean.width + sx) * 3;
	        const size_t dst_idx = (static_cast<size_t>(oy) * crop_size + ox) * 3;
	        crop->bgr[dst_idx + 0] = clean.bgr[src_idx + 0];
	        crop->bgr[dst_idx + 1] = clean.bgr[src_idx + 1];
	        crop->bgr[dst_idx + 2] = clean.bgr[src_idx + 2];
	      }
	    }
	    return true;
	  }

	  std::string build_slot_payload() const {
	    std::ostringstream os;
	    os << "{\"display_id_max_count\":" << display_slot_limit()
	       << ",\"slot_fps\":" << std::fixed << std::setprecision(3) << cfg_.webui_slot_fps
	       << ",\"slots\":[";
	    bool first = true;
	    for (const auto& item : slot_crops_) {
	      const DisplaySlotCropState& s = item.second;
	      if (s.display_id <= 0 || s.image_data_url.empty()) {
	        continue;
	      }
	      if (!first) {
	        os << ",";
	      }
	      first = false;
	      os << "{\"display_id\":" << s.display_id
	         << ",\"public_id\":" << s.public_id
	         << ",\"raw_track_id\":" << s.raw_track_id
		         << ",\"frame_id\":" << s.frame_id
		         << ",\"face_id\":\"" << json_escape(s.face_id) << "\""
		         << ",\"front_score\":" << std::fixed << std::setprecision(3)
		         << s.front_score
		         << ",\"image\":\"" << s.image_data_url << "\"}";
	    }
	    os << "]}";
	    return os.str();
	  }

	  void publish_slot_crops(const Image& clean,
		                          const FrameResult& result,
		                          const std::vector<TargetInfo>& targets) {
		    if (webui_server_ == nullptr || !webui_server_->has_slot_clients()) {
		      return;
		    }
		    const auto now = std::chrono::steady_clock::now();
		    const double fps_limit = std::max(0.1, cfg_.webui_slot_fps);
		    const std::chrono::duration<double> min_interval(1.0 / fps_limit);
		    if (last_slot_publish_time_.time_since_epoch().count() != 0 &&
		        now - last_slot_publish_time_ < min_interval) {
		      return;
		    }
		    last_slot_publish_time_ = now;

		    const int limit = display_slot_limit();
		    std::map<int, DisplaySlotCropState> next_slots;
		    const int n = std::min(result.detection_count, static_cast<int>(targets.size()));
		    for (int i = 0; i < n; ++i) {
		      const TargetInfo& target = targets[static_cast<size_t>(i)];
	      if (!target.has_display_id || target.display_id <= 0 || target.display_id > limit) {
	        continue;
	      }
		      if (result.detections.size() < (static_cast<size_t>(i) + 1) * kDetectionFields) {
		        continue;
		      }
		      const float* det = result.detections.data() + static_cast<size_t>(i) * kDetectionFields;
		      const std::string face_id = target_face_name(target);
		      const double front_score = target_front_score(target, det);
		      Image crop;
		      if (!build_slot_crop(clean, det, &crop)) {
		        continue;
	      }
	      try {
		        if (!slot_crop_encoder_) {
		          slot_crop_encoder_.reset(new TurboJpegEncoder());
		        }
		        const std::vector<uint8_t> jpeg = slot_crop_encoder_->encode_bgr(crop, 72);
		        DisplaySlotCropState state;
		        state.display_id = target.display_id;
		        state.public_id = target.public_id;
		        state.raw_track_id = target.raw_track_id;
		        state.frame_id = result.frame_index;
		        state.face_id = face_id;
		        state.front_score = front_score;
		        state.image_data_url =
		            "data:image/jpeg;base64," + base64_encode(jpeg.data(), jpeg.size());
		        next_slots[target.display_id] = std::move(state);
		      } catch (const std::exception& e) {
		        std::cerr << "[board_cpp-webui] slot crop encode failed: " << e.what() << "\n";
		      }
		    }
		    if (next_slots.empty() && slot_crops_.empty()) {
		      return;
		    }
		    slot_crops_.swap(next_slots);
		    webui_server_->publish_slots(build_slot_payload());
		  }

	  void run_loop() {
	    TurboJpegEncoder encoder;
	    while (true) {
      std::unique_ptr<Job> job;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [&] {
          return !running_.load() || latest_job_ != nullptr;
        });
        if (!running_.load() && latest_job_ == nullptr) {
          break;
        }
        job = std::move(latest_job_);
      }
      if (!job || webui_server_ == nullptr) {
        continue;
      }
      const bool has_frame_clients = webui_server_->has_frame_clients();
      const bool has_slot_clients = webui_server_->has_slot_clients();
      if (!has_frame_clients && !has_slot_clients) {
        continue;
      }
	      try {
	        Image annotated = remap_panorama_cpu(
	            job->source, runtime_.meta(), runtime_.base_map_x(), runtime_.base_map_y());
	        if (has_slot_clients) {
	          publish_slot_crops(annotated, job->result, job->targets);
	        }
	        if (!has_frame_clients) {
	          continue;
	        }
	        draw_annotation(annotated, cfg_, job->result, job->targets, job->fps);
	        Image scaled;
	        const Image* encode_image = &annotated;
	        if (cfg_.webui_frame_scale < 0.999) {
	          scaled = resize_nearest(annotated, cfg_.webui_frame_scale);
	          encode_image = &scaled;
	        }
        const std::vector<uint8_t> jpeg =
            encoder.encode_bgr(*encode_image, cfg_.webui_jpeg_quality);
        webui_server_->publish_jpeg(jpeg);
      } catch (const std::exception& e) {
        std::cerr << "[board_cpp-webui] frame publish failed: " << e.what() << "\n";
      } catch (...) {
        std::cerr << "[board_cpp-webui] frame publish failed: unknown exception\n";
      }
    }
  }

  const Config& cfg_;
  const MeetEyeRuntime& runtime_;
  BoardWebUiServer* webui_server_ = nullptr;
  std::atomic<bool> running_{false};
	  std::thread worker_;
	  std::mutex mutex_;
		  std::condition_variable cv_;
		  std::unique_ptr<Job> latest_job_;
		  std::map<int, DisplaySlotCropState> slot_crops_;
		  std::unique_ptr<TurboJpegEncoder> slot_crop_encoder_;
		  std::chrono::steady_clock::time_point last_slot_publish_time_;
		};

class WebSocketServer {
 public:
  WebSocketServer(const std::string& host, int port, const std::string& path)
      : host_(host), port_(port), path_(path.empty() ? "/ws/inference" : path) {}

  ~WebSocketServer() { stop(); }

  void start() {
    if (running_.load()) {
      return;
    }
    listen_fd_ = create_listen_socket();
    running_.store(true);
    sender_thread_ = std::thread(&WebSocketServer::sender_loop, this);
    accept_thread_ = std::thread(&WebSocketServer::accept_loop, this);
    std::cerr << "[board_cpp-ws] JSON WebSocket listening: ws://" << host_ << ":"
              << port_ << path_ << "\n";
  }

  void stop() {
    if (!running_.load() && listen_fd_ < 0) {
      return;
    }
    running_.store(false);
    if (listen_fd_ >= 0) {
      ::shutdown(listen_fd_, SHUT_RDWR);
      ::close(listen_fd_);
      listen_fd_ = -1;
    }
    sender_cv_.notify_all();
    if (accept_thread_.joinable()) {
      accept_thread_.join();
    }
    if (sender_thread_.joinable()) {
      sender_thread_.join();
    }
    std::vector<int> clients;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      clients.swap(clients_);
    }
    for (int fd : clients) {
      ::shutdown(fd, SHUT_RDWR);
      ::close(fd);
    }
  }

  void publish(const std::string& payload) {
    if (!running_.load()) {
      return;
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      latest_payload_ = payload;
      payload_dirty_ = true;
    }
    sender_cv_.notify_one();
  }

 private:
  int create_listen_socket() const {
    struct addrinfo hints {};
    hints.ai_family = (host_.empty() || host_ == "0.0.0.0") ? AF_INET : AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_PASSIVE;
    struct addrinfo* result = nullptr;
    const std::string port = std::to_string(port_);
    const char* node = (host_.empty() || host_ == "0.0.0.0") ? nullptr : host_.c_str();
    const int gai = getaddrinfo(node, port.c_str(), &hints, &result);
    if (gai != 0) {
      throw std::runtime_error(std::string("getaddrinfo failed for websocket bind: ") +
                               gai_strerror(gai));
    }

    int fd = -1;
    for (struct addrinfo* rp = result; rp != nullptr; rp = rp->ai_next) {
      fd = ::socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
      if (fd < 0) {
        continue;
      }
      int one = 1;
      setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
      if (::bind(fd, rp->ai_addr, rp->ai_addrlen) == 0 && ::listen(fd, 16) == 0) {
        break;
      }
      ::close(fd);
      fd = -1;
    }
    freeaddrinfo(result);
    if (fd < 0) {
      throw std::runtime_error("cannot bind websocket server on " + host_ + ":" +
                               std::to_string(port_));
    }
    return fd;
  }

  void accept_loop() {
    while (running_.load()) {
      sockaddr_storage addr {};
      socklen_t addr_len = sizeof(addr);
      const int fd = ::accept(listen_fd_, reinterpret_cast<sockaddr*>(&addr), &addr_len);
      if (fd < 0) {
        if (running_.load() && errno != EINTR) {
          std::cerr << "[board_cpp-ws] accept failed: " << std::strerror(errno) << "\n";
        }
        continue;
      }
      set_socket_recv_timeout(fd, 1000);
      if (!handle_handshake(fd)) {
        ::close(fd);
        continue;
      }
      std::string latest;
      {
        std::lock_guard<std::mutex> lock(mutex_);
        clients_.push_back(fd);
        latest = latest_payload_;
      }
      if (!set_fd_nonblocking(fd)) {
        remove_clients(std::vector<int>{fd});
        continue;
      }
      set_socket_send_buffer(fd, 1024 * 1024);
      if (!latest.empty()) {
        std::vector<uint8_t> frame = make_frame(latest);
        if (!send_all_nonblocking(fd, frame)) {
          remove_clients(std::vector<int>{fd});
        }
      }
    }
  }

  void sender_loop() {
    while (true) {
      std::string payload;
      std::vector<int> clients;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        sender_cv_.wait(lock, [&] {
          return !running_.load() || payload_dirty_;
        });
        if (!running_.load() && !payload_dirty_) {
          break;
        }
        payload = latest_payload_;
        clients = clients_;
        payload_dirty_ = false;
      }
      if (payload.empty() || clients.empty()) {
        continue;
      }
      const std::vector<uint8_t> frame = make_frame(payload);
      std::vector<int> failed;
      for (int fd : clients) {
        if (!send_all_nonblocking(fd, frame)) {
          failed.push_back(fd);
        }
      }
      if (!failed.empty()) {
        remove_clients(failed);
      }
    }
  }

  bool handle_handshake(int fd) const {
    std::string request;
    char buffer[1024];
    while (request.find("\r\n\r\n") == std::string::npos && request.size() < 8192) {
      const ssize_t n = ::recv(fd, buffer, sizeof(buffer), 0);
      if (n <= 0) {
        return false;
      }
      request.append(buffer, buffer + n);
    }
    const size_t first_line_end = request.find("\r\n");
    const std::string first_line =
        first_line_end == std::string::npos ? request : request.substr(0, first_line_end);
    const std::string expected = "GET " + path_ + " ";
    if (first_line.find(expected) != 0) {
      const std::string not_found =
          "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
      send_all(fd, not_found);
      return false;
    }
    const std::string key = get_http_header(request, "Sec-WebSocket-Key");
    if (key.empty()) {
      const std::string bad =
          "HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
      send_all(fd, bad);
      return false;
    }
    std::ostringstream response;
    response << "HTTP/1.1 101 Switching Protocols\r\n"
             << "Upgrade: websocket\r\n"
             << "Connection: Upgrade\r\n"
             << "Sec-WebSocket-Accept: " << websocket_accept_key(key) << "\r\n"
             << "\r\n";
    return send_all(fd, response.str());
  }

  std::vector<uint8_t> make_frame(const std::string& payload) const {
    std::vector<uint8_t> frame;
    frame.reserve(payload.size() + 10);
    frame.push_back(0x82);  // FIN + binary frame, matching Python send_bytes().
    const uint64_t len = static_cast<uint64_t>(payload.size());
    if (len <= 125) {
      frame.push_back(static_cast<uint8_t>(len));
    } else if (len <= 0xffff) {
      frame.push_back(126);
      frame.push_back(static_cast<uint8_t>((len >> 8) & 0xff));
      frame.push_back(static_cast<uint8_t>(len & 0xff));
    } else {
      frame.push_back(127);
      for (int i = 7; i >= 0; --i) {
        frame.push_back(static_cast<uint8_t>((len >> (i * 8)) & 0xff));
      }
    }
    frame.insert(frame.end(), payload.begin(), payload.end());
    return frame;
  }

  void remove_clients(const std::vector<int>& failed) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<int> kept;
    kept.reserve(clients_.size());
    for (int fd : clients_) {
      if (std::find(failed.begin(), failed.end(), fd) == failed.end()) {
        kept.push_back(fd);
      } else {
        ::shutdown(fd, SHUT_RDWR);
        ::close(fd);
      }
    }
    clients_.swap(kept);
  }

  std::string host_;
  int port_ = 0;
  std::string path_;
  int listen_fd_ = -1;
  std::atomic<bool> running_{false};
  std::thread accept_thread_;
  std::thread sender_thread_;
  std::mutex mutex_;
  std::condition_variable sender_cv_;
  std::vector<int> clients_;
  std::string latest_payload_;
  bool payload_dirty_ = false;
};
