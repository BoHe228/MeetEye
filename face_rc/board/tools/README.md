# RK3588 Native Tools And Probes

This directory keeps the native C++ sources used by the RK3588 board runtimes.
The generated binaries and shared libraries under `tools/bin/` are build
artifacts and are intentionally ignored by git.

## Probe: OpenCL Import Of RKNN Input Memory

`probe_opencl_rknn_dmabuf.cpp` checks whether an RKNN input memory fd can be
imported by OpenCL through `cl_arm_import_memory`.

```bash
cd face_rc/board
bash tools/build_probe_opencl_rknn_dmabuf.sh

LD_LIBRARY_PATH=$PWD/tools/bin/lib:$LD_LIBRARY_PATH \
  tools/bin/probe_opencl_rknn_dmabuf \
  ../board_cpp/models/yolov8n-face-640-b1-int8-hybrid-split-kptconf-rk3588.rknn
```

Passing this probe means the C++ runtime can try its zero-copy bound-input path.
If it fails on a target system, `board_cpp` falls back to a bound-copy path and
then to the older CPU input-buffer path.

## Probe: Double RKNN Bound Input Buffers

`probe_rknn_double_bound_input.cpp` checks whether one RKNN context can switch
between two independent input memories, and whether OpenCL can write the next
buffer while RKNN is running the current one.

```bash
cd face_rc/board
bash tools/build_probe_rknn_double_bound_input.sh

LD_LIBRARY_PATH=$PWD/tools/bin/lib:$LD_LIBRARY_PATH \
  tools/bin/probe_rknn_double_bound_input \
  ../board_cpp/models/yolov8n-face-640-b1-int8-hybrid-split-kptconf-rk3588.rknn
```

Expected success ends with:

```text
[result] double RKNN input buffers look feasible for OpenCL/RKNN pipelining.
```

This proves the API and memory ownership are feasible on the tested RK3588
runtime. The production C++ pipeline still needs explicit per-context
double-buffer state before this is used for frame-level scheduling.
