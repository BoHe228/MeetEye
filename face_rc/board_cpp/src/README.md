# board_cpp implementation split

`meeteye_cpp_smoke.cpp` is still the only compiled translation unit.
The files in this directory are included by that file in the original source order,
so this split is intended to be behavior-preserving while making future feature work easier.

Current layout:

- `meeteye_types.inc`: config, shared result structs and frame-rate structs.
- `meeteye_profile_system.inc`: profile summary, filesystem helpers and system/NPU load sampling.
- `meeteye_frame_sources.inc`: file/image list loading, TurboJPEG, image prefetch and V4L2 camera capture.
- `meeteye_cli.inc`: command-line help, parsing and config normalization.
- `meeteye_drawing_json.inc`: drawing helpers, debug JSON helpers, target JSON helpers and NPY helpers.
- `meeteye_face_recognition.inc`: AdaFace runtime, crop alignment and FaceID feature library.
- `meeteye_output_payloads.inc`: sector overlay, annotation drawing, target selection and JSON output builders.
- `meeteye_native_tracker.inc`: native library handles and HybridSORT/public/display ID management.
- `meeteye_runtime_pipeline.inc`: map loading, OpenCL/RKNN pipeline and per-frame inference.
- `meeteye_webui.inc`: WebUI HTTP server, WebSocket server and async frame publisher.
- `meeteye_runner.inc`: output dispatch and main run loop.
