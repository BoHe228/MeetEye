#!/usr/bin/env python3
"""Fine-tune yolo26n-pose.pt on OmniLab + PoseFES + COCO-Pose at imgsz 640."""

from __future__ import annotations

import sys

from train_yolo26_pose import main


if __name__ == "__main__":
    defaults = [
        "--data",
        "datasets/mixed_pose_640/mixed_pose_640.yaml",
        "--name",
        "yolo26n-pose-mixed-640",
        "--imgsz",
        "640",
        "--batch",
        "16",
        "--optimizer",
        "AdamW",
        "--lr0",
        "0.0005",
        "--warmup-epochs",
        "1.0",
        "--mosaic",
        "0.3",
        "--scale",
        "0.3",
    ]
    sys.argv = [sys.argv[0], *defaults, *sys.argv[1:]]
    main()
