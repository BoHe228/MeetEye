from db import PersonFeatureSystem


def add_new_images_to_database(new_image_paths, db_path='person_feature_db.pkl',
                                  csv_path='person_features.csv', model_path=None,
                                  match_threshold=0.75, add_new_feature_threshold=0.7,
                                  auto_add_new_feature=True, max_features_per_id=5):
    """
    将新图片添加到现有数据库（支持特征漂移处理）

    方案说明：
    - 每个ID维护一个特征集合（默认最多5个特征）
    - 新图片与该ID的所有特征比较，取最高相似度
    - 相似度 >= match_threshold (0.75)：判定为同一人，自动添加新特征到该ID
    - 相似度在 [add_new_feature_threshold, match_threshold) 区间：可能是同一人但不确定，保守创建新ID
    - 相似度 < add_new_feature_threshold：确定是新人，创建新ID

    参数:
        new_image_paths: 新图片路径或路径列表
        db_path: 数据库文件路径
        csv_path: CSV记录文件路径
        model_path: 模型文件路径
        match_threshold: 主要匹配阈值（默认0.75）
        add_new_feature_threshold: 添加新特征阈值（默认0.7）
        auto_add_new_feature: 匹配成功时是否自动添加新特征（默认True）
        max_features_per_id: 每个ID最多保留的特征数量（默认5）

    返回:
        system: PersonFeatureSystem 实例
    """
    if isinstance(new_image_paths, str):
        new_image_paths = [new_image_paths]

    print("=" * 75)
    print("行人特征管理系统 - 支持特征漂移处理")
    print("=" * 75)
    print(f"匹配阈值 (match_threshold): {match_threshold}")
    print(f"添加阈值 (add_new_feature_threshold): {add_new_feature_threshold}")
    print(f"每个ID最大特征数: {max_features_per_id}")
    print(f"自动添加新特征: {auto_add_new_feature}")
    print()

    system = PersonFeatureSystem(
        model_path=model_path,
        db_path=db_path,
        csv_path=csv_path,
        max_features_per_id=max_features_per_id
    )

    print("\n当前数据库状态:")
    system.feature_db.print_database()

    print(f"\n开始处理 {len(new_image_paths)} 张新图片...")
    system.add_images(
        new_image_paths,
        match_threshold=match_threshold,
        add_new_feature_threshold=add_new_feature_threshold,
        auto_add_new_feature=auto_add_new_feature
    )

    print("\n最终数据库状态:")
    system.feature_db.print_database()

    print("\n导出更新后的CSV...")
    system.feature_db.export_to_csv(csv_path)

    stats = system.feature_db.get_statistics()
    print(f"\n数据库统计:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    system.save()

    print("\n" + "=" * 75)
    print("处理完成！")
    print("=" * 75)

    return system


if __name__ == "__main__":
    model_path = r"imagenet.pyth\osnet_ain_x0_25_imagenet.pyth"

    add_new_images_to_database(
        r'data\1.jpg',
        db_path='person_feature_db.pkl',
        csv_path='person_features.csv',
        model_path=model_path,
        match_threshold=0.80,        # 主要匹配阈值
        add_new_feature_threshold=0.75,  # 次阈值
        auto_add_new_feature=True,      # 匹配到同一人时自动添加新特征
        max_features_per_id=5           # 每个ID最多保留5个特征
    )

