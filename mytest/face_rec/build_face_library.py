"""
人脸特征库构建脚本

用法：
    将人脸照片放入 face_photos/ 目录，文件名即为人名：
        face_photos/
            张三.jpg
            李四.png
            ...

    运行：
        python mytest/face_rec/build_face_library.py

    输出：face_library/ 目录下生成对应的 .npy 特征文件。

支持格式：jpg / jpeg / png / bmp / webp
人脸检测：YOLOv8-face（yolov8n-face.pt）
          检测失败时自动回退为中心裁切
"""

import os
import sys
import argparse
import numpy as np
import cv2

_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_DIR, '..', '..'))

sys.path.insert(0, _DIR)
from face_rec_manager import FaceRecManager

_IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

# YOLOv8-face 模型（懒加载，首次调用时初始化）
_yolo_face_model = None


def _get_yolo_model(model_path: str):
    global _yolo_face_model
    if _yolo_face_model is None:
        from ultralytics import YOLO
        _yolo_face_model = YOLO(model_path)
        print(f"[FaceDetect] YOLOv8-face 已加载：{model_path}")
    return _yolo_face_model


def detect_and_crop_face(
    img_bgr: np.ndarray,
    yolo_model_path: str,
    padding: float = 0.3,
) -> np.ndarray:
    """
    用 YOLOv8-face 检测人脸并裁切，失败时回退到中心裁切。

    Args:
        img_bgr: 输入 BGR 图像
        yolo_model_path: yolov8n-face.pt 路径
        padding: 人脸框向外扩展比例（保留额头/下巴）

    Returns:
        112×112 BGR 人脸图像
    """
    h, w = img_bgr.shape[:2]

    try:
        model = _get_yolo_model(yolo_model_path)
        results = model(img_bgr, verbose=False)
        boxes = results[0].boxes

        if boxes is not None and len(boxes) > 0:
            # 取置信度最高的检测框
            confs = boxes.conf.cpu().numpy()
            best_idx = int(np.argmax(confs))
            x1, y1, x2, y2 = boxes.xyxy[best_idx].cpu().numpy().astype(int)

            # 向外扩展 padding
            fw, fh = x2 - x1, y2 - y1
            pad_x = int(fw * padding)
            pad_y = int(fh * padding)
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(w, x2 + pad_x)
            y2 = min(h, y2 + pad_y)
            crop = img_bgr[y1:y2, x1:x2]
            print(f"    ✓ 检测到人脸（置信度={confs[best_idx]:.2f}），"
                  f"裁切区域 ({x1},{y1})→({x2},{y2})")
            return cv2.resize(crop, (112, 112))
        else:
            print(f"    ! 未检测到人脸，使用中心裁切（建议换一张更清晰的正面照）")

    except Exception as e:
        print(f"    ! YOLOv8-face 推理失败：{e}，使用中心裁切")

    # 回退：取图像中心 70% 区域
    margin_x = int(w * 0.15)
    margin_y = int(h * 0.15)
    crop = img_bgr[margin_y:h - margin_y, margin_x:w - margin_x]
    if crop.size == 0:
        crop = img_bgr
    return cv2.resize(crop, (112, 112))


def build_library(
    photo_dir: str,
    library_dir: str,
    model_path: str,
    yolo_model_path: str,
    device: str = 'cpu',
    overwrite: bool = False,
):
    os.makedirs(library_dir, exist_ok=True)

    manager = FaceRecManager(
        model_path=model_path,
        library_dir=library_dir,
        threshold=0.35,
        device=device,
    )

    image_files = [
        f for f in os.listdir(photo_dir)
        if os.path.splitext(f)[1].lower() in _IMG_EXTS
    ]

    if not image_files:
        print(f"[Error] {photo_dir} 中没有找到图片文件")
        return

    print(f"\n共找到 {len(image_files)} 张照片，开始处理...\n")

    success, skipped, failed = 0, 0, 0

    for fname in sorted(image_files):
        name = os.path.splitext(fname)[0]
        out_path = os.path.join(library_dir, f"{name}.npy")

        print(f"[{name}]  {fname}")

        if os.path.exists(out_path) and not overwrite:
            print(f"    → 已存在 {name}.npy，跳过（用 --overwrite 强制重建）")
            skipped += 1
            continue

        img_path = os.path.join(photo_dir, fname)
        img = cv2.imread(img_path)
        if img is None:
            print(f"    ✗ 读取失败，跳过")
            failed += 1
            continue

        face_112 = detect_and_crop_face(img, yolo_model_path)

        # 保存裁切结果供人工检查
        preview_dir = os.path.join(library_dir, '_preview')
        os.makedirs(preview_dir, exist_ok=True)
        preview_path = os.path.join(preview_dir, f"{name}_crop.jpg")
        cv2.imwrite(preview_path, face_112)
        print(f"    → 裁切预览已保存：{preview_path}")

        feature = manager.extract_feature(face_112)
        np.save(out_path, feature)
        print(f"    → 已保存 {name}.npy（特征维度: {feature.shape}）")
        success += 1

    print(f"\n{'='*50}")
    print(f"完成：成功 {success} 人 | 跳过 {skipped} 人 | 失败 {failed} 人")
    print(f"特征库位置：{os.path.abspath(library_dir)}")
    print(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(description='人脸特征库构建工具')
    parser.add_argument('--photo-dir', type=str,
                        default=os.path.join(_DIR, 'face_photos'),
                        help='人脸照片目录，文件名即为人名 (默认: mytest/face_rec/face_photos)')
    parser.add_argument('--library-dir', type=str, default='face_library',
                        help='特征库输出目录 (默认: face_library)')
    parser.add_argument('--model', type=str,
                        default='face_rec_model/adaface_ir18_webface4m.ckpt',
                        help='AdaFace 模型权重路径')
    parser.add_argument('--yolo-face-model', type=str,
                        default=os.path.join(_REPO_ROOT, 'yolo_model', 'yolov8n-face.pt'),
                        help='YOLOv8-face 模型路径 (默认: MeetEye/yolo_model/yolov8n-face.pt)')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'],
                        help='AdaFace 推理设备 (默认: cpu)')
    parser.add_argument('--overwrite', action='store_true',
                        help='重新提取已存在的特征（默认跳过）')
    args = parser.parse_args()

    if not os.path.isdir(args.photo_dir):
        print(f"[Error] 照片目录不存在: {args.photo_dir}")
        print(f"请创建目录并放入照片，文件名即为人名，例如：")
        print(f"  mytest/face_rec/face_photos/张三.jpg")
        print(f"  mytest/face_rec/face_photos/李四.png")
        sys.exit(1)

    if not os.path.exists(args.yolo_face_model):
        print(f"[Error] YOLOv8-face 模型不存在: {args.yolo_face_model}")
        sys.exit(1)

    build_library(
        photo_dir=args.photo_dir,
        library_dir=args.library_dir,
        model_path=args.model,
        yolo_model_path=args.yolo_face_model,
        device=args.device,
        overwrite=args.overwrite,
    )


if __name__ == '__main__':
    main()
