"""
人脸识别诊断脚本
运行: python mytest/face_rec/diagnose.py
"""
import sys, os
import numpy as np
import cv2

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)
sys.path.insert(0, os.path.join(_DIR, 'AdaFace'))

from face_rec_manager import FaceRecManager

CKPT  = 'face_rec_model/adaface_ir18_webface4m.ckpt'
LIB   = 'face_library'

mgr = FaceRecManager(model_path=CKPT, library_dir=LIB, device='cpu')

# ── 步骤 1：库特征自一致性检查 ──────────────────────────────────────────
print("\n=== 步骤1：库特征自一致性（相同图片重新提取，应接近 1.0） ===")
preview_dir = os.path.join(LIB, '_preview')
for fname in sorted(os.listdir(preview_dir)):
    if not fname.endswith('_crop.jpg'):
        continue
    name = fname.replace('_crop.jpg', '')
    npy_path = os.path.join(LIB, f'{name}.npy')
    if not os.path.exists(npy_path):
        continue

    saved_feat = np.load(npy_path).flatten().astype(np.float32)
    saved_feat /= (np.linalg.norm(saved_feat) + 1e-6)

    crop = cv2.imread(os.path.join(preview_dir, fname))  # 112×112 BGR
    reextracted = mgr.extract_feature(crop)

    sim = float(saved_feat @ reextracted)
    norm_saved = np.linalg.norm(saved_feat)
    norm_re    = np.linalg.norm(reextracted)
    print(f"  {name}: 自一致相似度={sim:.4f}  (saved_norm={norm_saved:.4f}, reextracted_norm={norm_re:.4f})")

# ── 步骤 2：库内两两相似度（不同人应在 0~0.4，同人应 > 0.5）────────────
npy_files = [f for f in os.listdir(LIB) if f.endswith('.npy')]
if len(npy_files) >= 2:
    print("\n=== 步骤2：库内两两相似度 ===")
    feats = {}
    for f in npy_files:
        n = f[:-4]
        v = np.load(os.path.join(LIB, f)).flatten().astype(np.float32)
        v /= (np.linalg.norm(v) + 1e-6)
        feats[n] = v
    names = list(feats.keys())
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            sim = float(feats[names[i]] @ feats[names[j]])
            print(f"  {names[i]} vs {names[j]}: {sim:.4f}")

# ── 步骤 3：RGB vs BGR 敏感性测试 ────────────────────────────────────────
print("\n=== 步骤3：BGR vs RGB 敏感性（检查通道顺序是否影响结果） ===")
for fname in sorted(os.listdir(preview_dir))[:1]:
    if not fname.endswith('_crop.jpg'):
        continue
    name = fname.replace('_crop.jpg', '')
    crop_bgr = cv2.imread(os.path.join(preview_dir, fname))
    crop_rgb = crop_bgr[:, :, ::-1].copy()  # 转 RGB

    feat_bgr = mgr.extract_feature(crop_bgr)
    feat_rgb = mgr.extract_feature(crop_rgb)

    sim_self  = float(feat_bgr @ feat_bgr)
    sim_cross = float(feat_bgr @ feat_rgb)
    print(f"  {name}: BGR自相似={sim_self:.4f}  BGR vs RGB={sim_cross:.4f}")
    print(f"  （如果 BGR vs RGB 差异很大，说明通道顺序对结果敏感，需确认输入格式）")
