"""
鱼眼相机实时展开 - 主程序入口

功能：
1. 摄像头模式 - 使用摄像头实时展开
2. 视频模式 - 使用视频文件实时展开
"""

import os
from fisheye_unwrapper import FisheyeUnwrapper


def main():
    """主程序"""
    print("=" * 70)
    print("🎬 鱼眼相机展开工具")
    print("=" * 70)

    # ========== 配置区域 - 选择模式 ==========
    # 📌 选择模式: "camera" 或 "video"
    MODE = "video"  # 可选: "camera" (摄像头) / "video" (视频文件)

    # ========== 通用配置 ==========
    MAP_FILE = r"zhankai\map\大会议室_6.4_多人开会.npz"
    OUTPUT_WIDTH = 3840
    OUTPUT_HEIGHT = 1080
    CUT_POSITION = "right"  # "right" 或 "left"
    DISPLAY_SCALE = 0.5

    # ========== 摄像头模式配置 ==========
    CAMERA_INDEX = 1

    # ========== 视频模式配置 ==========
    INPUT_VIDEO = r"fish_checkerboard_videos\大会议室_6.4_多人开会_3秒.mp4"

    # =====================================

    if MODE == "camera":
        print("\n📹 摄像头模式")
        print("=" * 70)

        # 检查映射文件
        if not os.path.exists(MAP_FILE):
            print(f"⚠️  映射文件不存在: {MAP_FILE}")
            print("   将从摄像头自动计算并保存...")

        # 创建展开器
        unwrapper = FisheyeUnwrapper.with_camera(
            camera_index=CAMERA_INDEX,
            map_file=MAP_FILE,
            output_width=OUTPUT_WIDTH,
            output_height=OUTPUT_HEIGHT,
            cut_position=CUT_POSITION,
            display_scale=DISPLAY_SCALE,
        )

        # 打开输入源
        if not unwrapper.open():
            print("\n无法打开摄像头！")
            print("\n摄像头使用建议:")
            print("  1. 运行 scan_cameras.py 查看可用摄像头索引")
            print("  2. 确认摄像头未被其他程序占用")
            print("  3. 尝试更改 camera_index 参数 (0, 1, 2...)")
            input("\n按回车键退出...")
            return

        # 若映射文件不存在则保存新计算的结果
        if not os.path.exists(MAP_FILE):
            unwrapper.save_mapping(MAP_FILE)

        # 运行
        unwrapper.run()
        unwrapper.release()

    elif MODE == "video":
        print("\n🎥 视频模式")
        print("=" * 70)

        # 检查视频文件是否存在
        if not os.path.exists(INPUT_VIDEO):
            print(f"❌ 输入视频不存在: {INPUT_VIDEO}")
            input("\n按回车键退出...")
            return

        # 检查映射文件是否存在，不存在则自动计算
        if not os.path.exists(MAP_FILE):
            print(f"⚠️  映射文件不存在: {MAP_FILE}")
            print("   将从视频自动计算并保存...")

        print(f"📂 输入: {os.path.basename(INPUT_VIDEO)}")
        print(f"🗺️  映射: {os.path.basename(MAP_FILE)}")
        print(f"📐 分辨率: {OUTPUT_WIDTH}x{OUTPUT_HEIGHT}")
        print(f"✂️  切开点: {CUT_POSITION}")
        print("=" * 70)

        # 创建展开器
        unwrapper = FisheyeUnwrapper.with_video(
            video_path=INPUT_VIDEO,
            map_file=MAP_FILE,
            output_width=OUTPUT_WIDTH,
            output_height=OUTPUT_HEIGHT,
            cut_position=CUT_POSITION,
            display_scale=DISPLAY_SCALE,
        )

        # 打开输入源
        if not unwrapper.open():
            print("\n无法打开视频文件！")
            input("\n按回车键退出...")
            return

        # 若映射文件不存在则保存新计算的结果
        if not os.path.exists(MAP_FILE):
            unwrapper.save_mapping(MAP_FILE)

        # 运行
        unwrapper.run()
        unwrapper.release()

    else:
        print(f"\n❌ 未知模式: {MODE}")
        print("请选择 'camera' 或 'video'")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 程序被用户中断")
    except Exception as e:
        print(f"\n❌ 程序发生错误: {e}")
        import traceback
        traceback.print_exc()
        input("\n按回车键退出...")
