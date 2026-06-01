"""
Demo video compressor for MeetEye README.

Usage:
    python compress_demo.py                          # default: compress demo video
    python compress_demo.py -i input.mp4 -o out.mp4
    python compress_demo.py --duration 90            # trim to first N seconds
    python compress_demo.py --target-mb 25           # target file size in MB

Requires ffmpeg to be installed and on PATH.
"""
import argparse
import os
import subprocess
import sys


def compress(
    input_path: str,
    output_path: str,
    target_mb: float = 25.0,
    duration: float | None = None,
    scale: str = "960:-2",          # width:height (-2 = keep aspect ratio)
    crf: int = 28,                   # H.264 quality; lower = better; 23-30 typical
    preset: str = "slow",            # encoding speed/compression trade-off
    audio: bool = False,
) -> None:
    """Compress a video to roughly `target_mb` MB."""

    # --- probe duration -------------------------------------------------------
    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", input_path],
        capture_output=True, text=True,
    )
    total_duration = float(probe.stdout.strip()) if probe.returncode == 0 else None
    clip_duration = min(duration, total_duration) if (duration and total_duration) else duration

    # --- compute target bitrate -----------------------------------------------
    used_duration = clip_duration or total_duration or 120
    # target_mb * 8 bits/byte * 1024 kbit/Mbit / seconds = kbps
    target_kbps = int(target_mb * 8 * 1024 / used_duration)
    # Leave 64 kbps for audio (if enabled), rest for video
    video_kbps = max(target_kbps - (64 if audio else 0), 200)

    print(f"Input : {input_path}")
    print(f"Output: {output_path}")
    print(f"Duration used: {used_duration:.1f}s  →  target total {target_kbps} kbps  (video {video_kbps} kbps)")

    # --- build ffmpeg command --------------------------------------------------
    cmd = ["ffmpeg", "-y", "-i", input_path]

    if clip_duration:
        cmd += ["-t", str(clip_duration)]

    vf = f"scale={scale}"
    cmd += [
        "-vf", vf,
        "-c:v", "libx264",
        "-preset", preset,
        "-b:v", f"{video_kbps}k",
        "-maxrate", f"{video_kbps * 2}k",
        "-bufsize", f"{video_kbps * 4}k",
        "-crf", str(crf),
        "-movflags", "+faststart",   # optimize for streaming / web playback
        "-pix_fmt", "yuv420p",       # broad compatibility
    ]

    if audio:
        cmd += ["-c:a", "aac", "-b:a", "64k"]
    else:
        cmd += ["-an"]

    cmd.append(output_path)

    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print("ffmpeg failed.", file=sys.stderr)
        sys.exit(1)

    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"\nDone!  Output size: {size_mb:.1f} MB  →  {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compress demo video for GitHub README.")
    parser.add_argument("-i", "--input",
                        default="yolo_pose_output/小会议室_4人黑板交流.mp4",
                        help="Input video path")
    parser.add_argument("-o", "--output",
                        default="yolo_pose_output/demo_compressed.mp4",
                        help="Output video path")
    parser.add_argument("--target-mb", type=float, default=25.0,
                        help="Target output file size in MB (default: 25)")
    parser.add_argument("--duration", type=float, default=90.0,
                        help="Trim to first N seconds (default: 90, None = full video)")
    parser.add_argument("--scale", default="960:-2",
                        help="ffmpeg scale filter value (default: 960:-2)")
    parser.add_argument("--crf", type=int, default=28,
                        help="H.264 CRF quality (18-35, lower=better, default: 28)")
    parser.add_argument("--preset", default="slow",
                        choices=["ultrafast","superfast","veryfast","faster",
                                 "fast","medium","slow","slower","veryslow"],
                        help="ffmpeg encoding preset (default: slow)")
    parser.add_argument("--audio", action="store_true",
                        help="Include audio (default: strip audio)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    compress(
        input_path=args.input,
        output_path=args.output,
        target_mb=args.target_mb,
        duration=args.duration if args.duration > 0 else None,
        scale=args.scale,
        crf=args.crf,
        preset=args.preset,
        audio=args.audio,
    )


if __name__ == "__main__":
    main()
