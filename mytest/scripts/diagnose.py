"""
边界匹配诊断脚本

检查边界匹配功能是否正常工作
"""
import sys
import numpy as np

print("=" * 60)
print("边界匹配功能诊断")
print("=" * 60)

# 1. 检查BoundaryIDMatcher是否可以导入
print("\n[1] 检查模块导入...")
try:
    from BoundaryIDMatcher import BoundaryIDMatcher, BoundaryCrossingTracker
    print("  ✓ BoundaryIDMatcher 导入成功")
except Exception as e:
    print(f"  ✗ BoundaryIDMatcher 导入失败: {e}")
    sys.exit(1)

# 2. 检查BoT_SORT是否可以导入
print("\n[2] 检查BoT_SORT集成...")
try:
    from BoT_SORT import BoT_SORTTracker
    print("  ✓ BoT_SORTTracker 导入成功")
except Exception as e:
    print(f"  ✗ BoT_SORTTracker 导入失败: {e}")
    sys.exit(1)

# 3. 测试BoundaryIDMatcher基本功能
print("\n[3] 测试BoundaryIDMatcher基本功能...")
try:
    matcher = BoundaryIDMatcher(
        frame_width=3840,
        frame_height=1080,
        boundary_margin=0.1,
        time_window=30,
        similarity_threshold=0.6
    )
    print("  ✓ BoundaryIDMatcher 初始化成功")

    # 测试边界检测
    test_bbox_left = [10, 500, 200, 700]  # 左边界
    test_bbox_right = [3600, 500, 3800, 700]  # 右边界
    test_bbox_center = [1800, 500, 2000, 700]  # 中间

    side_left = matcher.get_boundary_side(test_bbox_left)
    side_right = matcher.get_boundary_side(test_bbox_right)
    side_center = matcher.get_boundary_side(test_bbox_center)

    print(f"  ✓ 边界检测: 左侧框={side_left}, 右侧框={side_right}, 中间框={side_center}")

    # 测试特征匹配
    feat = np.random.randn(512)
    matcher.add_disappeared_target(
        track_id=1,
        bbox=test_bbox_left,
        feature=feat,
        frame_id=100
    )
    print("  ✓ 消失目标添加成功")

    matched_id = matcher.find_matching_id(
        bbox=test_bbox_right,
        feature=feat,
        frame_id=105
    )
    print(f"  ✓ 特征匹配: 匹配ID={matched_id}")

    stats = matcher.get_stats()
    print(f"  ✓ 统计信息: {stats}")

except Exception as e:
    print(f"  ✗ BoundaryIDMatcher 测试失败: {e}")
    import traceback
    traceback.print_exc()

# 4. 测试带边界匹配的BoT_SORTTracker
print("\n[4] 测试带边界匹配的BoT_SORTTracker...")
try:
    tracker = BoT_SORTTracker(
        track_high_thresh=0.3,
        track_low_thresh=0.1,
        new_track_thresh=0.3,
        track_buffer=50,
        match_thresh=0.7,
        with_reid=True,
        enable_boundary_matching=True,  # 启用边界匹配
        frame_width=3840,
        frame_height=1080,
        boundary_margin=0.1,
        boundary_time_window=30,
        boundary_similarity_thresh=0.6
    )
    print("  ✓ BoT_SORTTracker 初始化成功 (带边界匹配)")

    # 检查边界匹配器是否真的启用
    print(f"  ✓ enable_boundary_matching = {tracker.enable_boundary_matching}")
    print(f"  ✓ boundary_tracker is not None = {tracker.boundary_tracker is not None}")

except Exception as e:
    print(f"  ✗ BoT_SORTTracker 测试失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("诊断完成！")
print("=" * 60)

print("""
发现的问题（可能的原因）:
---------------------------
1. 边界匹配器在main.py中启用了，但没有设置正确的frame_width/frame_height
2. 没有在全景图尺寸确定后调用 set_boundary_frame_size()
3. 需要确认特征是否正确传递给跟踪器
4. 可能需要添加调试打印来查看边界匹配的执行情况

建议的修复:
-----------
1. 在全景图尺寸确定后，调用 tracker.set_boundary_frame_size()
2. 添加调试输出查看边界匹配统计
3. 验证检测结果中的'feature'字段不是None
""")
