#!/usr/bin/env python3
"""Render a standalone timing dashboard from board_cpp debug JSONL."""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


STAGES = [
    ("frame_total", "Frame total"),
    ("file_read", "File read"),
    ("jpeg_decode", "JPEG decode"),
    ("decode_wait", "Decode wait"),
    ("opencl_ensure", "OpenCL init/ensure"),
    ("opencl_run_outer", "OpenCL slice"),
    ("opencl_upload", "OpenCL upload"),
    ("opencl_kernel", "OpenCL kernel"),
    ("opencl_read", "OpenCL read"),
    ("rknn_total_outer", "RKNN total"),
    ("rknn_run_max", "RKNN run max"),
    ("rknn_output_max", "RKNN output max"),
    ("rknn_decode_max", "RKNN decode max"),
    ("native_merge", "Native merge"),
    ("tracker_outer", "Tracker"),
    ("angle_targets", "Angle targets"),
    ("build_payload", "Build payload"),
    ("bound_prepare", "Bound prepare"),
    ("bound_import", "Bound import"),
    ("jsonl", "JSONL write"),
    ("websocket", "WebSocket"),
    ("debug_jsonl", "Debug JSONL"),
]

PIE_STAGES = [
    ("jpeg_decode", "JPEG decode", "#3b82f6"),
    ("decode_wait", "Decode wait", "#60a5fa"),
    ("opencl_run_outer", "OpenCL slice", "#10b981"),
    ("rknn_total_outer", "RKNN inference", "#f59e0b"),
    ("tracker_outer", "Tracker", "#8b5cf6"),
    ("angle_targets", "Angle", "#ec4899"),
    ("build_payload", "Payload", "#14b8a6"),
    ("bound_prepare", "Bound prepare", "#64748b"),
    ("bound_import", "Bound import", "#94a3b8"),
    ("websocket", "WebSocket", "#ef4444"),
    ("jsonl", "JSONL", "#a855f7"),
]


def percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct
    low = int(math.floor(rank))
    high = int(math.ceil(rank))
    if low == high:
        return ordered[low]
    frac = rank - low
    return ordered[low] * (1.0 - frac) + ordered[high] * frac


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{lineno}: invalid JSON: {exc}") from exc
    return rows


def default_system_profile_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_system_profile.jsonl")


def load_system_rows(path: Optional[Path]) -> List[Dict[str, Any]]:
    if path is None or not path.exists():
        return []
    return read_jsonl(path)


def collect_records(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for i, row in enumerate(rows):
        profile = row.get("profile_ms") or {}
        fps = row.get("fps") or {}
        timings = row.get("timings_ms") or {}
        tracker = row.get("tracker") or {}
        merge = row.get("merge_stats") or {}
        frame_index = int(row.get("frame_index", i))
        detection_count = int(row.get("detection_count", 0) or 0)
        records.append(
            {
                "frame": frame_index,
                "image": str(row.get("image", "")),
                "frame_ms": finite_float(
                    fps.get("frame_ms", profile.get("frame_total", timings.get("frame_total", 0.0)))
                ),
                "instant_fps": finite_float(fps.get("instant", timings.get("instant_fps", 0.0))),
                "average_fps": finite_float(fps.get("average", timings.get("average_fps", 0.0))),
                "detection_count": detection_count,
                "raw_detections": int(tracker.get("raw_detections", detection_count) or 0),
                "tracks": int(tracker.get("tracks", 0) or 0),
                "created": int(tracker.get("created", 0) or 0),
                "matches_first": int(tracker.get("matches_first", 0) or 0),
                "matches_byte": int(tracker.get("matches_byte", 0) or 0),
                "decoded": int(merge.get("decoded", 0) or 0),
                "nms_keep": int(merge.get("nms_keep", 0) or 0),
                "final": int(merge.get("final", detection_count) or 0),
                "profile": {k: finite_float(v) for k, v in profile.items()},
            }
        )
    return records


def values_for(records: Sequence[Dict[str, Any]], key: str) -> List[float]:
    return [finite_float(r.get("profile", {}).get(key, 0.0)) for r in records]


def stat_row(records: Sequence[Dict[str, Any]], warm_records: Sequence[Dict[str, Any]], key: str, label: str) -> Dict[str, Any]:
    values = values_for(records, key)
    warm_values = values_for(warm_records, key)
    return {
        "key": key,
        "label": label,
        "avg": mean(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "max": max(values) if values else 0.0,
        "warm_avg": mean(warm_values),
        "warm_p95": percentile(warm_values, 0.95),
        "warm_max": max(warm_values) if warm_values else 0.0,
    }


def fmt_ms(value: float) -> str:
    return f"{value:.2f} ms"


def fmt_num(value: float) -> str:
    return f"{value:.2f}"


def build_report(input_path: Path, rows: List[Dict[str, Any]], warmup_frames: int) -> Dict[str, Any]:
    records = collect_records(rows)
    if not records:
        raise SystemExit(f"{input_path}: no frames found")

    warm_records = records[min(warmup_frames, len(records)) :] or records
    final_avg_fps = records[-1]["average_fps"]
    frame_ms = [r["frame_ms"] for r in records]
    warm_frame_ms = [r["frame_ms"] for r in warm_records]
    warm_fps_from_frame = 1000.0 / mean(warm_frame_ms) if mean(warm_frame_ms) > 0 else 0.0
    instant_fps = [r["instant_fps"] for r in warm_records if r["instant_fps"] > 0]

    stage_stats = [stat_row(records, warm_records, key, label) for key, label in STAGES]
    pie = []
    for key, label, color in PIE_STAGES:
        value = stat_row(records, warm_records, key, label)["warm_avg"]
        if value > 0.0005:
            pie.append({"key": key, "label": label, "value": value, "color": color})

    outliers = sorted(records, key=lambda r: r["frame_ms"], reverse=True)[:12]
    summary = [
        {"label": "Frames", "value": str(len(records))},
        {"label": "Final average FPS", "value": fmt_num(final_avg_fps)},
        {"label": "Warm FPS from frame time", "value": fmt_num(warm_fps_from_frame)},
        {"label": "Warm instant FPS p50", "value": fmt_num(percentile(instant_fps, 0.50))},
        {"label": "Frame total avg", "value": fmt_ms(mean(frame_ms))},
        {"label": "Warm frame total avg", "value": fmt_ms(mean(warm_frame_ms))},
        {"label": "Warm frame total p95", "value": fmt_ms(percentile(warm_frame_ms, 0.95))},
        {"label": "Frame total max", "value": fmt_ms(max(frame_ms))},
    ]

    return {
        "title": "board_cpp timing profile",
        "source": str(input_path),
        "warmup_frames": warmup_frames,
        "summary": summary,
        "stage_stats": stage_stats,
        "pie": pie,
        "outliers": outliers,
        "records": records,
        "system_source": "",
        "system_records": [],
        "system_summary": [],
    }


def summarize_system_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, str]]:
    def vals(key: str) -> List[float]:
        out: List[float] = []
        for row in rows:
            value = row.get(key)
            if value is None:
                continue
            parsed = finite_float(value, float("nan"))
            if math.isfinite(parsed):
                out.append(parsed)
        return out

    cards: List[Dict[str, str]] = [{"label": "System samples", "value": str(len(rows))}]
    for key, label, suffix in [
        ("cpu_percent", "CPU avg/max", "%"),
        ("memory_percent", "Memory avg/max", "%"),
        ("gpu_percent", "GPU avg/max", "%"),
        ("npu_percent", "NPU avg/max", "%"),
        ("thermal_max_c", "Temp avg/max", "C"),
    ]:
        data = vals(key)
        if data:
            cards.append({"label": label, "value": f"{mean(data):.1f}{suffix} / {max(data):.1f}{suffix}"})
        else:
            cards.append({"label": label, "value": "N/A"})
    return cards


def attach_system_report(report: Dict[str, Any], system_path: Optional[Path], system_rows: List[Dict[str, Any]]) -> None:
    if not system_rows:
        return
    records: List[Dict[str, Any]] = []
    first_ts = finite_float(system_rows[0].get("timestamp", 0.0))
    for i, row in enumerate(system_rows):
        timestamp = finite_float(row.get("timestamp", 0.0))
        records.append(
            {
                "sample": int(row.get("sample_id", i + 1) or (i + 1)),
                "time_s": finite_float(row.get("time_s", timestamp - first_ts)),
                "cpu_percent": row.get("cpu_percent"),
                "memory_percent": row.get("memory_percent"),
                "memory_available_mb": row.get("memory_available_mb"),
                "gpu_percent": row.get("gpu_percent"),
                "gpu_freq_hz": row.get("gpu_freq_hz"),
                "npu_percent": row.get("npu_percent"),
                "npu_freq_hz": row.get("npu_freq_hz"),
                "thermal_max_c": row.get("thermal_max_c"),
            }
        )
    report["system_source"] = str(system_path) if system_path else ""
    report["system_records"] = records
    report["system_summary"] = summarize_system_rows(system_rows)


def render_html(report: Dict[str, Any]) -> str:
    report_json = json.dumps(report, ensure_ascii=False, separators=(",", ":"))
    source = html.escape(report["source"])
    title = html.escape(report["title"])
    return (
        r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>"""
        + title
        + r"""</title>
  <style>
    :root {
      --bg: #f6f7f9;
      --panel: #ffffff;
      --text: #151922;
      --muted: #667085;
      --line: #d9dee8;
      --accent: #2563eb;
      --good: #10b981;
      --warn: #f59e0b;
      --danger: #ef4444;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    header {
      padding: 24px 28px 12px;
      border-bottom: 1px solid var(--line);
      background: #fff;
    }
    h1 { margin: 0 0 6px; font-size: 24px; font-weight: 700; letter-spacing: 0; }
    .source { color: var(--muted); font-size: 13px; }
    main { padding: 20px 28px 32px; max-width: 1500px; margin: 0 auto; }
    .metrics {
      display: grid;
      grid-template-columns: repeat(4, minmax(160px, 1fr));
      gap: 12px;
      margin-bottom: 18px;
    }
    .metric, .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }
    .metric { padding: 14px 16px; min-height: 76px; }
    .metric-label { color: var(--muted); font-size: 12px; margin-bottom: 6px; }
    .metric-value { font-size: 22px; font-weight: 700; white-space: nowrap; }
    .grid {
      display: grid;
      grid-template-columns: minmax(0, 1.4fr) minmax(340px, 0.8fr);
      gap: 16px;
      margin-bottom: 16px;
    }
    .panel { padding: 16px; min-width: 0; }
    .panel h2 { margin: 0 0 4px; font-size: 17px; font-weight: 700; }
    .sub { margin: 0 0 12px; color: var(--muted); font-size: 12px; }
    canvas { width: 100%; height: 310px; display: block; }
    #donutChart { height: 340px; }
    .legend {
      display: flex;
      flex-wrap: wrap;
      gap: 8px 14px;
      margin-top: 8px;
      color: var(--muted);
      font-size: 12px;
    }
    .legend span { display: inline-flex; align-items: center; gap: 6px; }
    .swatch { width: 10px; height: 10px; border-radius: 2px; display: inline-block; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th, td { padding: 8px 10px; border-bottom: 1px solid var(--line); text-align: right; }
    th:first-child, td:first-child { text-align: left; }
    th { color: var(--muted); font-weight: 600; background: #fbfcfe; position: sticky; top: 0; }
    .table-wrap { max-height: 520px; overflow: auto; border: 1px solid var(--line); border-radius: 8px; }
    .note {
      margin-top: 10px;
      color: var(--muted);
      font-size: 12px;
    }
    @media (max-width: 980px) {
      header { padding: 20px 16px 10px; }
      main { padding: 16px; }
      .metrics { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .grid { grid-template-columns: 1fr; }
      canvas { height: 280px; }
    }
    @media (max-width: 520px) {
      .metrics { grid-template-columns: 1fr; }
      .metric-value { font-size: 19px; }
    }
  </style>
</head>
<body>
  <header>
    <h1>board_cpp timing profile</h1>
    <div class="source">Source: <code>"""
        + source
        + r"""</code></div>
  </header>
  <main>
    <section class="metrics" id="summary"></section>
    <section class="metrics" id="systemSummary" style="display:none"></section>

    <section class="grid">
      <div class="panel">
        <h2>Frame and Stage Timeline</h2>
        <p class="sub">Major per-frame timings in milliseconds. First-frame initialization is kept visible.</p>
        <canvas id="timelineChart"></canvas>
        <div class="legend" id="timelineLegend"></div>
      </div>
      <div class="panel">
        <h2>Average Stage Cost</h2>
        <p class="sub">Warm frames only. Decode is worker cost and can overlap with frame processing.</p>
        <canvas id="donutChart"></canvas>
        <div class="legend" id="donutLegend"></div>
      </div>
    </section>

    <section class="grid" id="systemSection" style="display:none">
      <div class="panel">
        <h2>Hardware Load</h2>
        <p class="sub">Background samples from /proc and /sys. GPU/NPU are shown only when the kernel exposes load nodes.</p>
        <canvas id="systemLoadChart"></canvas>
        <div class="legend" id="systemLoadLegend"></div>
      </div>
      <div class="panel">
        <h2>Temperature and Frequency</h2>
        <p class="sub">Thermal max in Celsius and available devfreq current frequencies.</p>
        <canvas id="systemFreqChart"></canvas>
        <div class="legend" id="systemFreqLegend"></div>
      </div>
    </section>

    <section class="grid">
      <div class="panel">
        <h2>FPS</h2>
        <p class="sub">Instant FPS and cumulative average FPS reported by the runtime.</p>
        <canvas id="fpsChart"></canvas>
        <div class="legend" id="fpsLegend"></div>
      </div>
      <div class="panel">
        <h2>Tracker and Detection Counts</h2>
        <p class="sub">Raw detections, active tracks, and final emitted detections per frame.</p>
        <canvas id="countChart"></canvas>
        <div class="legend" id="countLegend"></div>
      </div>
    </section>

    <section class="panel">
      <h2>Stage Statistics</h2>
      <p class="sub">All values are milliseconds. Warm columns skip the configured warmup frames.</p>
      <div class="table-wrap">
        <table id="stageTable"></table>
      </div>
    </section>

    <section class="panel" style="margin-top:16px">
      <h2>Slowest Frames</h2>
      <p class="sub">Useful for spotting initialization spikes and runtime stalls.</p>
      <div class="table-wrap">
        <table id="outlierTable"></table>
      </div>
    </section>
  </main>

  <script>
    const report = __REPORT_JSON__;
    const palette = {
      frame: '#111827',
      rknn: '#f59e0b',
      jpeg: '#3b82f6',
      opencl: '#10b981',
      wait: '#60a5fa',
      tracker: '#8b5cf6',
      fps: '#2563eb',
      avg: '#ef4444',
      raw: '#f97316',
      detections: '#06b6d4'
    };

    function fmt(value, digits = 2) {
      if (!Number.isFinite(value)) return '0.00';
      return value.toFixed(digits);
    }

    function setupCanvas(id) {
      const canvas = document.getElementById(id);
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      canvas.width = Math.max(1, Math.floor(rect.width * dpr));
      canvas.height = Math.max(1, Math.floor(rect.height * dpr));
      const ctx = canvas.getContext('2d');
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      return { canvas, ctx, width: rect.width, height: rect.height };
    }

    function drawLegend(id, lines) {
      const el = document.getElementById(id);
      el.innerHTML = lines.map(line =>
        `<span><i class="swatch" style="background:${line.color}"></i>${line.label}</span>`
      ).join('');
    }

    function niceMax(value) {
      if (value <= 0) return 1;
      const power = Math.pow(10, Math.floor(Math.log10(value)));
      const n = value / power;
      const step = n <= 2 ? 2 : n <= 5 ? 5 : 10;
      return step * power;
    }

    function drawLineChart(id, records, lines, opts = {}) {
      const { ctx, width, height } = setupCanvas(id);
      const pad = { left: 52, right: 18, top: 18, bottom: 34 };
      const plotW = Math.max(1, width - pad.left - pad.right);
      const plotH = Math.max(1, height - pad.top - pad.bottom);
      ctx.clearRect(0, 0, width, height);
      ctx.font = '12px system-ui, sans-serif';
      const maxY = niceMax(Math.max(...records.flatMap(r => lines.map(l => Number(l.value(r)) || 0)), 1));
      const minX = records.length ? records[0].frame : 0;
      const maxX = records.length ? records[records.length - 1].frame : 1;
      const xRange = Math.max(1, maxX - minX);

      ctx.strokeStyle = '#e5e7eb';
      ctx.lineWidth = 1;
      ctx.fillStyle = '#667085';
      for (let i = 0; i <= 4; i++) {
        const y = pad.top + plotH * i / 4;
        const value = maxY * (1 - i / 4);
        ctx.beginPath();
        ctx.moveTo(pad.left, y);
        ctx.lineTo(width - pad.right, y);
        ctx.stroke();
        ctx.fillText(fmt(value, opts.digits ?? 0), 8, y + 4);
      }

      ctx.strokeStyle = '#cbd5e1';
      ctx.beginPath();
      ctx.moveTo(pad.left, pad.top);
      ctx.lineTo(pad.left, pad.top + plotH);
      ctx.lineTo(width - pad.right, pad.top + plotH);
      ctx.stroke();

      for (const line of lines) {
        ctx.strokeStyle = line.color;
        ctx.lineWidth = line.width || 2;
        ctx.beginPath();
        records.forEach((r, idx) => {
          const x = pad.left + ((r.frame - minX) / xRange) * plotW;
          const y = pad.top + (1 - ((Number(line.value(r)) || 0) / maxY)) * plotH;
          if (idx === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        });
        ctx.stroke();
      }

      ctx.fillStyle = '#667085';
      ctx.fillText(`frame ${minX}`, pad.left, height - 10);
      const endLabel = `frame ${maxX}`;
      const endWidth = ctx.measureText(endLabel).width;
      ctx.fillText(endLabel, width - pad.right - endWidth, height - 10);
    }

    function drawDonut(id, slices) {
      const { ctx, width, height } = setupCanvas(id);
      ctx.clearRect(0, 0, width, height);
      const cx = width / 2;
      const cy = height / 2 - 6;
      const r = Math.min(width, height) * 0.34;
      const inner = r * 0.58;
      const total = slices.reduce((sum, s) => sum + s.value, 0) || 1;
      let angle = -Math.PI / 2;
      for (const s of slices) {
        const next = angle + (s.value / total) * Math.PI * 2;
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.arc(cx, cy, r, angle, next);
        ctx.closePath();
        ctx.fillStyle = s.color;
        ctx.fill();
        angle = next;
      }
      ctx.beginPath();
      ctx.arc(cx, cy, inner, 0, Math.PI * 2);
      ctx.fillStyle = '#fff';
      ctx.fill();
      ctx.fillStyle = '#151922';
      ctx.font = '700 22px system-ui, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(`${fmt(total, 1)} ms`, cx, cy - 2);
      ctx.font = '12px system-ui, sans-serif';
      ctx.fillStyle = '#667085';
      ctx.fillText('avg stage cost', cx, cy + 18);
      ctx.textAlign = 'left';
    }

    function fillSummary() {
      document.getElementById('summary').innerHTML = report.summary.map(item => `
        <div class="metric">
          <div class="metric-label">${item.label}</div>
          <div class="metric-value">${item.value}</div>
        </div>
      `).join('');
      const sys = report.system_summary || [];
      const sysEl = document.getElementById('systemSummary');
      if (sys.length) {
        sysEl.style.display = '';
        sysEl.innerHTML = sys.map(item => `
          <div class="metric">
            <div class="metric-label">${item.label}</div>
            <div class="metric-value">${item.value}</div>
          </div>
        `).join('');
      }
    }

    function fillTables() {
      const stageRows = report.stage_stats.map(s => `
        <tr>
          <td>${s.label}</td>
          <td>${fmt(s.avg)}</td>
          <td>${fmt(s.p50)}</td>
          <td>${fmt(s.p95)}</td>
          <td>${fmt(s.max)}</td>
          <td>${fmt(s.warm_avg)}</td>
          <td>${fmt(s.warm_p95)}</td>
          <td>${fmt(s.warm_max)}</td>
        </tr>
      `).join('');
      document.getElementById('stageTable').innerHTML = `
        <thead><tr>
          <th>Stage</th><th>Avg</th><th>P50</th><th>P95</th><th>Max</th>
          <th>Warm avg</th><th>Warm P95</th><th>Warm max</th>
        </tr></thead><tbody>${stageRows}</tbody>
      `;

      const outlierRows = report.outliers.map(r => `
        <tr>
          <td>${r.frame}</td>
          <td>${fmt(r.frame_ms)}</td>
          <td>${fmt(r.profile.rknn_total_outer || 0)}</td>
          <td>${fmt(r.profile.jpeg_decode || 0)}</td>
          <td>${fmt(r.profile.opencl_ensure || 0)}</td>
          <td>${fmt(r.profile.opencl_run_outer || 0)}</td>
          <td>${r.detection_count}</td>
          <td>${r.tracks}</td>
          <td>${r.image}</td>
        </tr>
      `).join('');
      document.getElementById('outlierTable').innerHTML = `
        <thead><tr>
          <th>Frame</th><th>Total</th><th>RKNN</th><th>JPEG</th><th>OpenCL ensure</th>
          <th>OpenCL slice</th><th>Det</th><th>Tracks</th><th>Image</th>
        </tr></thead><tbody>${outlierRows}</tbody>
      `;
    }

    function render() {
      fillSummary();
      fillTables();
      const records = report.records;
      const timelineLines = [
        { label: 'Frame total', color: palette.frame, value: r => r.frame_ms, width: 2.5 },
        { label: 'RKNN total', color: palette.rknn, value: r => r.profile.rknn_total_outer || 0 },
        { label: 'JPEG decode', color: palette.jpeg, value: r => r.profile.jpeg_decode || 0 },
        { label: 'OpenCL slice', color: palette.opencl, value: r => r.profile.opencl_run_outer || 0 },
        { label: 'Decode wait', color: palette.wait, value: r => r.profile.decode_wait || 0 },
        { label: 'Tracker', color: palette.tracker, value: r => r.profile.tracker_outer || 0 }
      ];
      drawLineChart('timelineChart', records, timelineLines, { digits: 0 });
      drawLegend('timelineLegend', timelineLines);

      drawDonut('donutChart', report.pie);
      drawLegend('donutLegend', report.pie.map(s => ({ label: `${s.label}: ${fmt(s.value)} ms`, color: s.color })));

      const systemRecords = report.system_records || [];
      if (systemRecords.length) {
        document.getElementById('systemSection').style.display = '';
        const loadLines = [
          { label: 'CPU %', color: '#2563eb', value: r => r.cpu_percent ?? 0 },
          { label: 'Memory %', color: '#14b8a6', value: r => r.memory_percent ?? 0 },
          { label: 'GPU %', color: '#10b981', value: r => r.gpu_percent ?? 0 },
          { label: 'NPU %', color: '#f59e0b', value: r => r.npu_percent ?? 0 }
        ];
        drawLineChart('systemLoadChart', systemRecords.map((r, idx) => ({ ...r, frame: r.time_s || idx })), loadLines, { digits: 0 });
        drawLegend('systemLoadLegend', loadLines);

        const freqLines = [
          { label: 'Thermal max C', color: '#ef4444', value: r => r.thermal_max_c ?? 0 },
          { label: 'GPU GHz x10', color: '#10b981', value: r => r.gpu_freq_hz ? (r.gpu_freq_hz / 1e8) : 0 },
          { label: 'NPU GHz x10', color: '#f59e0b', value: r => r.npu_freq_hz ? (r.npu_freq_hz / 1e8) : 0 }
        ];
        drawLineChart('systemFreqChart', systemRecords.map((r, idx) => ({ ...r, frame: r.time_s || idx })), freqLines, { digits: 0 });
        drawLegend('systemFreqLegend', freqLines);
      }

      const fpsLines = [
        { label: 'Instant FPS', color: palette.fps, value: r => r.instant_fps },
        { label: 'Average FPS', color: palette.avg, value: r => r.average_fps }
      ];
      drawLineChart('fpsChart', records, fpsLines, { digits: 0 });
      drawLegend('fpsLegend', fpsLines);

      const countLines = [
        { label: 'Raw detections', color: palette.raw, value: r => r.raw_detections },
        { label: 'Tracks', color: palette.tracker, value: r => r.tracks },
        { label: 'Final detections', color: palette.detections, value: r => r.detection_count }
      ];
      drawLineChart('countChart', records, countLines, { digits: 0 });
      drawLegend('countLegend', countLines);
    }

    window.addEventListener('resize', render);
    render();
  </script>
</body>
</html>
"""
    ).replace("__REPORT_JSON__", report_json)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="board_cpp debug JSONL")
    parser.add_argument("--output", required=True, type=Path, help="output standalone HTML")
    parser.add_argument("--system-profile", default=None, type=Path, help="optional system profile JSONL")
    parser.add_argument("--warmup-frames", default=1, type=int, help="frames skipped in warm statistics")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.input)
    report = build_report(args.input, rows, max(0, args.warmup_frames))
    system_path = args.system_profile if args.system_profile is not None else default_system_profile_path(args.input)
    attach_system_report(report, system_path, load_system_rows(system_path))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_html(report), encoding="utf-8")
    print(f"wrote: {args.output}")


if __name__ == "__main__":
    main()
