#!/usr/bin/env python3
"""Render a compact interactive HTML profile from a board_cpp board_output dir."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


STAGES: List[Tuple[str, str, str]] = [
    ("file_read", "File read", "#94a3b8"),
    ("jpeg_decode", "JPEG decode", "#3b82f6"),
    ("decode_wait", "Decode wait", "#60a5fa"),
    ("opencl_ensure", "OpenCL ensure", "#64748b"),
    ("opencl_run_outer", "OpenCL slice total", "#10b981"),
    ("opencl_upload", "OpenCL upload", "#34d399"),
    ("opencl_kernel", "OpenCL kernel", "#059669"),
    ("opencl_read", "OpenCL read", "#6ee7b7"),
    ("rknn_total_outer", "RKNN total", "#f59e0b"),
    ("rknn_run_max", "RKNN run max", "#d97706"),
    ("rknn_output_max", "RKNN output max", "#fbbf24"),
    ("rknn_decode_max", "RKNN decode max", "#fde68a"),
    ("native_merge", "Native merge", "#14b8a6"),
    ("tracker_outer", "Tracker", "#8b5cf6"),
    ("angle_targets", "Angle targets", "#ec4899"),
    ("build_payload", "Build JSON payload", "#06b6d4"),
    ("jsonl", "JSONL write", "#a855f7"),
    ("websocket", "WebSocket", "#ef4444"),
    ("debug_jsonl", "Debug JSONL write", "#c084fc"),
]

PIE_KEYS = [
    "jpeg_decode",
    "opencl_run_outer",
    "rknn_total_outer",
    "native_merge",
    "tracker_outer",
    "angle_targets",
    "build_payload",
    "jsonl",
    "websocket",
]


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


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


def is_debug_jsonl(path: Path) -> bool:
    if not path.name.endswith(".jsonl"):
        return False
    if "system_profile" in path.name:
        return False
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                return isinstance(row.get("profile_ms"), dict)
    except Exception:
        return False
    return False


def find_latest_debug_jsonl(board_output: Path) -> Path:
    candidates = [p for p in board_output.glob("*.jsonl") if is_debug_jsonl(p)]
    if not candidates:
        raise SystemExit(f"no debug JSONL with profile_ms found in {board_output}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def default_system_profile_path(debug_path: Path) -> Path:
    return debug_path.with_name(f"{debug_path.stem}_system_profile.jsonl")


def compact_records(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for i, row in enumerate(rows):
        profile = row.get("profile_ms") or {}
        fps = row.get("fps") or {}
        timings = row.get("timings_ms") or {}
        tracker = row.get("tracker") or {}
        records.append(
            {
                "frame": int(row.get("frame_index", i) or i),
                "frame_ms": finite_float(fps.get("frame_ms", profile.get("frame_total", 0.0))),
                "instant_fps": finite_float(fps.get("instant", timings.get("instant_fps", 0.0))),
                "average_fps": finite_float(fps.get("average", timings.get("average_fps", 0.0))),
                "detection_count": int(row.get("detection_count", 0) or 0),
                "raw_detections": int(tracker.get("raw_detections", row.get("detection_count", 0)) or 0),
                "tracks": int(tracker.get("tracks", 0) or 0),
                "profile": {k: finite_float(v) for k, v in profile.items()},
            }
        )
    return records


def stage_averages(records: Sequence[Dict[str, Any]], warmup_frames: int) -> List[Dict[str, Any]]:
    warm = records[min(warmup_frames, len(records)) :] or records
    out: List[Dict[str, Any]] = []
    for key, label, color in STAGES:
        values = [finite_float(r["profile"].get(key, 0.0)) for r in warm]
        avg = mean(values)
        if avg <= 0.0005:
            continue
        out.append({"key": key, "label": label, "avg_ms": avg, "color": color})
    return out


def system_records(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not rows:
        return []
    first_ts = finite_float(rows[0].get("timestamp", 0.0))
    records: List[Dict[str, Any]] = []
    for i, row in enumerate(rows):
        ts = finite_float(row.get("timestamp", 0.0))
        records.append(
            {
                "sample": int(row.get("sample_id", i + 1) or (i + 1)),
                "time_s": finite_float(row.get("time_s", ts - first_ts)),
                "cpu_percent": row.get("cpu_percent"),
                "memory_percent": row.get("memory_percent"),
                "gpu_percent": row.get("gpu_percent"),
                "npu_percent": row.get("npu_percent"),
                "thermal_max_c": row.get("thermal_max_c"),
                "gpu_freq_hz": row.get("gpu_freq_hz"),
                "npu_freq_hz": row.get("npu_freq_hz"),
            }
        )
    return records


def metric_avg(records: Sequence[Dict[str, Any]], key: str) -> Optional[float]:
    values = [finite_float(r.get(key), float("nan")) for r in records if r.get(key) is not None]
    values = [v for v in values if math.isfinite(v)]
    return mean(values) if values else None


def build_report(debug_path: Path, system_path: Optional[Path], warmup_frames: int) -> Dict[str, Any]:
    records = compact_records(read_jsonl(debug_path))
    if not records:
        raise SystemExit(f"{debug_path}: no records")
    warm = records[min(warmup_frames, len(records)) :] or records
    stages = stage_averages(records, warmup_frames)
    pie = [s for s in stages if s["key"] in PIE_KEYS]
    sys_records = system_records(read_jsonl(system_path)) if system_path and system_path.exists() else []

    frame_avg = mean([r["frame_ms"] for r in warm])
    summary = [
        {"label": "Debug file", "value": debug_path.name},
        {"label": "Frames", "value": str(len(records))},
        {"label": "Average FPS", "value": f"{(1000.0 / frame_avg) if frame_avg > 0 else 0.0:.2f}"},
        {"label": "Average frame", "value": f"{frame_avg:.2f} ms"},
    ]
    if sys_records:
        summary.extend(
            [
                {"label": "CPU avg", "value": f"{metric_avg(sys_records, 'cpu_percent') or 0.0:.1f}%"},
                {"label": "Memory avg", "value": f"{metric_avg(sys_records, 'memory_percent') or 0.0:.1f}%"},
                {"label": "GPU avg", "value": "N/A" if metric_avg(sys_records, "gpu_percent") is None else f"{metric_avg(sys_records, 'gpu_percent'):.1f}%"},
                {"label": "NPU avg", "value": "N/A" if metric_avg(sys_records, "npu_percent") is None else f"{metric_avg(sys_records, 'npu_percent'):.1f}%"},
            ]
        )

    return {
        "debug_source": str(debug_path),
        "system_source": str(system_path) if system_path and system_path.exists() else "",
        "warmup_frames": warmup_frames,
        "summary": summary,
        "stages": stages,
        "pie": pie,
        "records": records,
        "system_records": sys_records,
    }


def render_html(report: Dict[str, Any]) -> str:
    report_json = json.dumps(report, ensure_ascii=False, separators=(",", ":"))
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>board_cpp 平均耗时与硬件负载</title>
  <style>
    :root {{ --bg:#f6f7f9; --panel:#fff; --line:#d9dee8; --text:#151922; --muted:#667085; }}
    * {{ box-sizing: border-box; }}
    body {{ margin:0; background:var(--bg); color:var(--text); font:14px/1.45 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
    header {{ background:#fff; border-bottom:1px solid var(--line); padding:22px 28px 12px; }}
    h1 {{ margin:0 0 6px; font-size:24px; letter-spacing:0; }}
    .source {{ color:var(--muted); font-size:13px; overflow-wrap:anywhere; }}
    main {{ max-width:1500px; margin:0 auto; padding:20px 28px 32px; }}
    .metrics {{ display:grid; grid-template-columns:repeat(4,minmax(160px,1fr)); gap:12px; margin-bottom:16px; }}
    .metric,.panel {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; box-shadow:0 1px 2px rgba(15,23,42,.04); }}
    .metric {{ min-height:74px; padding:13px 15px; }}
    .metric-label {{ color:var(--muted); font-size:12px; margin-bottom:5px; }}
    .metric-value {{ font-size:20px; font-weight:700; overflow-wrap:anywhere; }}
    .grid {{ display:grid; grid-template-columns:minmax(0,1fr) minmax(360px,.82fr); gap:16px; margin-bottom:16px; }}
    .panel {{ padding:16px; min-width:0; }}
    .panel h2 {{ margin:0 0 4px; font-size:17px; }}
    .sub {{ margin:0 0 12px; color:var(--muted); font-size:12px; }}
    canvas {{ width:100%; height:340px; display:block; }}
    .legend {{ display:flex; flex-wrap:wrap; gap:8px 14px; margin-top:8px; color:var(--muted); font-size:12px; }}
    .legend span {{ display:inline-flex; align-items:center; gap:6px; }}
    .swatch {{ width:10px; height:10px; border-radius:2px; display:inline-block; }}
    table {{ width:100%; border-collapse:collapse; font-size:13px; }}
    th,td {{ border-bottom:1px solid var(--line); padding:8px 10px; text-align:right; }}
    th:first-child,td:first-child {{ text-align:left; }}
    th {{ color:var(--muted); background:#fbfcfe; }}
    .table-wrap {{ max-height:520px; overflow:auto; border:1px solid var(--line); border-radius:8px; }}
    .tip {{ margin-top:10px; color:var(--muted); font-size:12px; min-height:18px; }}
    @media (max-width:980px) {{ main {{ padding:16px; }} header {{ padding:20px 16px 10px; }} .metrics {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} .grid {{ grid-template-columns:1fr; }} }}
    @media (max-width:520px) {{ .metrics {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body>
  <header>
    <h1>board_cpp 平均耗时与硬件负载</h1>
    <div class="source">Debug: <code>{report["debug_source"]}</code></div>
    <div class="source">System: <code>{report["system_source"] or "N/A"}</code></div>
  </header>
  <main>
    <section class="metrics" id="summary"></section>

    <section class="grid">
      <div class="panel">
        <h2>每个步骤平均耗时</h2>
        <p class="sub">柱状图显示 warmup 之后每个 C++ 阶段的平均耗时，单位 ms。</p>
        <canvas id="barChart"></canvas>
        <div class="tip" id="barTip"></div>
      </div>
      <div class="panel">
        <h2>平均耗时占比</h2>
        <p class="sub">点击/移动到饼图区域查看具体阶段。这里展示主要阶段，避免重复统计子项。</p>
        <canvas id="pieChart"></canvas>
        <div class="legend" id="pieLegend"></div>
        <div class="tip" id="pieTip"></div>
      </div>
    </section>

    <section class="panel">
      <h2>步骤耗时随帧变化</h2>
      <p class="sub">用于观察某个 C++ 阶段是否有持续抖动。</p>
      <canvas id="stageLineChart"></canvas>
      <div class="legend" id="stageLineLegend"></div>
    </section>

    <section class="panel" style="margin-top:16px">
      <h2>硬件负载随时间变化</h2>
      <p class="sub">CPU/内存/GPU/NPU 负载按系统采样时间绘制；GPU/NPU 没有内核节点时显示为 0 或 N/A。</p>
      <canvas id="hardwareChart"></canvas>
      <div class="legend" id="hardwareLegend"></div>
    </section>

    <section class="panel" style="margin-top:16px">
      <h2>温度与频率</h2>
      <p class="sub">温度单位 C；GPU/NPU 频率按 GHz x10 画到同一坐标系中。</p>
      <canvas id="tempFreqChart"></canvas>
      <div class="legend" id="tempFreqLegend"></div>
    </section>

    <section class="panel" style="margin-top:16px">
      <h2>平均耗时表</h2>
      <div class="table-wrap"><table id="stageTable"></table></div>
    </section>
  </main>
  <script>
    const report = {report_json};
    function fmt(v, d=2) {{ return Number.isFinite(v) ? v.toFixed(d) : 'N/A'; }}
    function canvasCtx(id) {{
      const canvas = document.getElementById(id);
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      canvas.width = Math.max(1, Math.floor(rect.width * dpr));
      canvas.height = Math.max(1, Math.floor(rect.height * dpr));
      const ctx = canvas.getContext('2d');
      ctx.setTransform(dpr,0,0,dpr,0,0);
      return {{canvas, ctx, width:rect.width, height:rect.height}};
    }}
    function niceMax(v) {{
      if (!Number.isFinite(v) || v <= 0) return 1;
      const p = Math.pow(10, Math.floor(Math.log10(v)));
      const n = v / p;
      return (n <= 2 ? 2 : n <= 5 ? 5 : 10) * p;
    }}
    function legend(id, lines) {{
      document.getElementById(id).innerHTML = lines.map(l =>
        `<span><i class="swatch" style="background:${{l.color}}"></i>${{l.label}}</span>`).join('');
    }}
    function fillSummary() {{
      document.getElementById('summary').innerHTML = report.summary.map(m => `
        <div class="metric"><div class="metric-label">${{m.label}}</div><div class="metric-value">${{m.value}}</div></div>
      `).join('');
    }}
    function drawAxes(ctx, width, height, pad, maxY, x0, x1) {{
      const plotH = height - pad.top - pad.bottom;
      ctx.font = '12px system-ui,sans-serif';
      ctx.strokeStyle = '#e5e7eb';
      ctx.fillStyle = '#667085';
      for (let i=0;i<=4;i++) {{
        const y = pad.top + plotH*i/4;
        ctx.beginPath(); ctx.moveTo(pad.left,y); ctx.lineTo(width-pad.right,y); ctx.stroke();
        ctx.fillText(fmt(maxY*(1-i/4),0), 8, y+4);
      }}
      ctx.strokeStyle = '#cbd5e1';
      ctx.beginPath(); ctx.moveTo(pad.left,pad.top); ctx.lineTo(pad.left,height-pad.bottom); ctx.lineTo(width-pad.right,height-pad.bottom); ctx.stroke();
      ctx.fillStyle = '#667085';
      ctx.fillText(String(x0), pad.left, height-10);
      const s = String(x1); ctx.fillText(s, width-pad.right-ctx.measureText(s).width, height-10);
    }}
    function drawLine(id, records, lines, xKey='frame') {{
      const {{ctx,width,height}} = canvasCtx(id);
      const pad = {{left:52,right:18,top:18,bottom:34}};
      const xs = records.map(r => Number(r[xKey]) || 0);
      const minX = xs.length ? xs[0] : 0, maxX = xs.length ? xs[xs.length-1] : 1;
      const xr = Math.max(1e-9, maxX-minX);
      const maxY = niceMax(Math.max(1, ...records.flatMap(r => lines.map(l => Number(l.value(r)) || 0))));
      ctx.clearRect(0,0,width,height);
      drawAxes(ctx,width,height,pad,maxY,fmt(minX,1),fmt(maxX,1));
      const plotW = width-pad.left-pad.right, plotH = height-pad.top-pad.bottom;
      for (const line of lines) {{
        ctx.strokeStyle = line.color; ctx.lineWidth = line.width || 2; ctx.beginPath();
        records.forEach((r,i) => {{
          const x = pad.left + (((Number(r[xKey])||0)-minX)/xr)*plotW;
          const y = pad.top + (1-(Number(line.value(r))||0)/maxY)*plotH;
          if (i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
        }});
        ctx.stroke();
      }}
    }}
    function drawBar() {{
      const {{canvas,ctx,width,height}} = canvasCtx('barChart');
      const pad = {{left:52,right:18,top:18,bottom:84}};
      const data = report.stages;
      const maxY = niceMax(Math.max(...data.map(d => d.avg_ms), 1));
      ctx.clearRect(0,0,width,height); drawAxes(ctx,width,height,pad,maxY,'','');
      const plotW = width-pad.left-pad.right, plotH = height-pad.top-pad.bottom;
      const step = plotW / Math.max(1, data.length);
      const bars = [];
      data.forEach((d,i) => {{
        const h = (d.avg_ms/maxY)*plotH;
        const x = pad.left + i*step + step*0.16;
        const y = pad.top + plotH - h;
        const w = step*0.68;
        ctx.fillStyle = d.color; ctx.fillRect(x,y,w,h);
        ctx.save(); ctx.translate(x+w/2, height-78); ctx.rotate(-Math.PI/4);
        ctx.fillStyle = '#667085'; ctx.font='11px system-ui,sans-serif'; ctx.fillText(d.label,0,0); ctx.restore();
        bars.push({{x,y,w,h,d}});
      }});
      canvas.onmousemove = ev => {{
        const r = canvas.getBoundingClientRect(); const x = ev.clientX-r.left; const y = ev.clientY-r.top;
        const hit = bars.find(b => x>=b.x && x<=b.x+b.w && y>=b.y && y<=b.y+b.h);
        document.getElementById('barTip').textContent = hit ? `${{hit.d.label}}: ${{fmt(hit.d.avg_ms)}} ms` : '';
      }};
    }}
    function drawPie() {{
      const {{canvas,ctx,width,height}} = canvasCtx('pieChart');
      const data = report.pie;
      const total = data.reduce((s,d)=>s+d.avg_ms,0) || 1;
      const cx=width/2, cy=height/2-8, r=Math.min(width,height)*.34, inner=r*.56;
      let a=-Math.PI/2; const arcs=[];
      ctx.clearRect(0,0,width,height);
      data.forEach(d => {{
        const b=a+(d.avg_ms/total)*Math.PI*2;
        ctx.beginPath(); ctx.moveTo(cx,cy); ctx.arc(cx,cy,r,a,b); ctx.closePath(); ctx.fillStyle=d.color; ctx.fill();
        arcs.push({{a,b,d}}); a=b;
      }});
      ctx.beginPath(); ctx.arc(cx,cy,inner,0,Math.PI*2); ctx.fillStyle='#fff'; ctx.fill();
      ctx.textAlign='center'; ctx.fillStyle='#151922'; ctx.font='700 22px system-ui,sans-serif'; ctx.fillText(`${{fmt(total,1)}} ms`,cx,cy-2);
      ctx.fillStyle='#667085'; ctx.font='12px system-ui,sans-serif'; ctx.fillText('avg selected stages',cx,cy+18); ctx.textAlign='left';
      legend('pieLegend', data.map(d => ({{label:`${{d.label}} ${{fmt(d.avg_ms)}}ms`, color:d.color}})));
      canvas.onmousemove = ev => {{
        const rect=canvas.getBoundingClientRect(); const x=ev.clientX-rect.left-cx; const y=ev.clientY-rect.top-cy;
        const rr=Math.sqrt(x*x+y*y); let ang=Math.atan2(y,x); if (ang < -Math.PI/2) ang += Math.PI*2;
        const hit = rr>=inner && rr<=r ? arcs.find(s => ang>=s.a && ang<=s.b) : null;
        document.getElementById('pieTip').textContent = hit ? `${{hit.d.label}}: ${{fmt(hit.d.avg_ms)}} ms, ${{fmt(hit.d.avg_ms/total*100,1)}}%` : '';
      }};
    }}
    function fillTable() {{
      document.getElementById('stageTable').innerHTML = '<thead><tr><th>步骤</th><th>平均耗时 ms</th></tr></thead><tbody>' +
        report.stages.map(s => `<tr><td>${{s.label}}</td><td>${{fmt(s.avg_ms)}}</td></tr>`).join('') + '</tbody>';
    }}
    function render() {{
      fillSummary(); fillTable(); drawBar(); drawPie();
      const stageLines = [
        {{label:'Frame total', color:'#111827', value:r=>r.frame_ms, width:2.5}},
        {{label:'RKNN total', color:'#f59e0b', value:r=>r.profile.rknn_total_outer||0}},
        {{label:'JPEG decode', color:'#3b82f6', value:r=>r.profile.jpeg_decode||0}},
        {{label:'OpenCL slice', color:'#10b981', value:r=>r.profile.opencl_run_outer||0}},
        {{label:'Tracker', color:'#8b5cf6', value:r=>r.profile.tracker_outer||0}}
      ];
      drawLine('stageLineChart', report.records, stageLines, 'frame'); legend('stageLineLegend', stageLines);
      const hw = report.system_records || [];
      const loadLines = [
        {{label:'CPU %', color:'#2563eb', value:r=>r.cpu_percent??0}},
        {{label:'Memory %', color:'#14b8a6', value:r=>r.memory_percent??0}},
        {{label:'GPU %', color:'#10b981', value:r=>r.gpu_percent??0}},
        {{label:'NPU %', color:'#f59e0b', value:r=>r.npu_percent??0}}
      ];
      drawLine('hardwareChart', hw, loadLines, 'time_s'); legend('hardwareLegend', loadLines);
      const tfLines = [
        {{label:'Thermal max C', color:'#ef4444', value:r=>r.thermal_max_c??0}},
        {{label:'GPU GHz x10', color:'#10b981', value:r=>r.gpu_freq_hz ? r.gpu_freq_hz/1e8 : 0}},
        {{label:'NPU GHz x10', color:'#f59e0b', value:r=>r.npu_freq_hz ? r.npu_freq_hz/1e8 : 0}}
      ];
      drawLine('tempFreqChart', hw, tfLines, 'time_s'); legend('tempFreqLegend', tfLines);
    }}
    window.addEventListener('resize', render); render();
  </script>
</body>
</html>"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board-output", type=Path, default=Path("board_output"), help="board_cpp board_output directory")
    parser.add_argument("--debug-jsonl", type=Path, default=None, help="debug JSONL; default picks latest in board-output")
    parser.add_argument("--system-profile", type=Path, default=None, help="system profile JSONL; default matches debug JSONL stem")
    parser.add_argument("--output", type=Path, default=None, help="output HTML; default is <debug_stem>_avg_profile.html")
    parser.add_argument("--warmup-frames", type=int, default=1, help="frames skipped from averages")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    board_output = args.board_output
    debug_path = args.debug_jsonl or find_latest_debug_jsonl(board_output)
    system_path = args.system_profile or default_system_profile_path(debug_path)
    output = args.output or debug_path.with_name(f"{debug_path.stem}_avg_profile.html")
    report = build_report(debug_path, system_path, max(0, args.warmup_frames))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_html(report), encoding="utf-8")
    print(f"debug: {debug_path}")
    print(f"system: {system_path if system_path.exists() else 'N/A'}")
    print(f"wrote: {output}")


if __name__ == "__main__":
    main()
