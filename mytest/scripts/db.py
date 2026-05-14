import torch
import torchreid
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import os
import json
import pickle
from datetime import datetime
import pandas as pd


def cosine_similarity_numpy(vec1, vec2):
    """计算两个向量的余弦相似度"""
    if torch.is_tensor(vec1):
        vec1 = vec1.numpy()
    if torch.is_tensor(vec2):
        vec2 = vec2.numpy()

    vec1 = vec1.flatten()
    vec2 = vec2.flatten()

    dot_product = np.dot(vec1, vec2)

    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    similarity = dot_product / (norm1 * norm2)
    return float(similarity)


def cosine_similarity_batch(query_feature, db_features):
    """批量计算余弦相似度（查询特征与数据库所有特征的相似度）"""
    if torch.is_tensor(query_feature):
        query_feature = query_feature.numpy()
    if torch.is_tensor(db_features):
        db_features = db_features.numpy()

    if query_feature.ndim == 1:
        query_feature = query_feature.reshape(1, -1)

    dot_products = np.dot(query_feature, db_features.T)

    query_norm = np.linalg.norm(query_feature, axis=1, keepdims=True)
    db_norms = np.linalg.norm(db_features, axis=1, keepdims=True)

    if np.any(query_norm == 0) or np.any(db_norms == 0):
        return np.zeros(query_feature.shape[0])

    similarities = dot_products / (query_norm * db_norms.T)

    return similarities.flatten()


class FeatureExtractorManager:
    """特征提取器管理类，封装模型加载和特征提取功能"""

    def __init__(self, model_name='osnet_x0_25', model_path=None, device=None):
        self.model_name = model_name
        self.model_path = model_path
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.extractor = None
        self.transform = None
        self._initialize_extractor()

    def _initialize_extractor(self):
        print("可用模型：")
        torchreid.models.show_avai_models()
        from torchreid.reid.utils import FeatureExtractor

        kwargs = {
            'model_name': self.model_name,
            'device': self.device,
            'verbose': True
        }

        if self.model_path and os.path.exists(self.model_path):
            kwargs['model_path'] = self.model_path
        else:
            print(f"警告: 模型文件 {self.model_path} 不存在，将使用预训练模型")

        self.extractor = FeatureExtractor(**kwargs)

        # 定义图像预处理变换（与 FeatureExtractor 内部保持一致）
        self.transform = transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __call__(self, image_paths):
        return self.extractor(image_paths)

    def extract_features(self, image_paths):
        return self.extractor(image_paths)

    def extract_features_from_image_array(self, image_array):
        """
        直接从图像数组提取特征（无需保存到文件）

        参数:
            image_array: numpy数组格式的图像 (H, W, C)，BGR格式（OpenCV格式）

        返回:
            特征向量
        """
        # 将 BGR 转换为 RGB
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

        # 转换为 PIL Image
        image = Image.fromarray(image_array)

        # 预处理
        image_tensor = self.transform(image).unsqueeze(0)
        image_tensor = image_tensor.to(self.device)

        # 提取特征
        self.extractor.model.eval()
        with torch.no_grad():
            features = self.extractor.model(image_tensor)

        # 返回特征向量
        return features.cpu()


# 添加cv2导入（如果还没有的话）
import cv2


class FeatureDatabase:
    """
    特征数据库类 - 支持每个ID多个特征，解决特征漂移问题

    方案说明：
    - 每个person_id维护一个特征列表（features）
    - 匹配时：新特征与该ID的所有特征比较，取最高相似度
    - 如果最高相似度超过阈值，则判定为同一人
    - 可选择性地将新特征添加到该ID的特征列表中，用于跟踪特征变化
    - 可限制每个ID的最大特征数量，避免无限增长
    """

    def __init__(self, db_path='feature_database.pkl', csv_path='person_features.csv',
                 max_features_per_id=5):
        """
        初始化数据库

        参数:
            db_path: 数据库文件路径
            csv_path: CSV记录文件路径
            max_features_per_id: 每个ID最多保留的特征数量
        """
        self.db_path = db_path
        self.csv_path = csv_path
        self.max_features_per_id = max_features_per_id
        self.database = self._load_database()
        self._sync_csv_to_db()

    def _load_database(self):
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'rb') as f:
                    db = pickle.load(f)
                total_features = sum(len(entry['features']) for entry in db)
                print(f"已加载现有数据库，包含 {len(db)} 个ID，共 {total_features} 个特征")
                return db
            except Exception as e:
                print(f"加载数据库失败: {e}，创建新的数据库")
                return []
        else:
            print("创建新的数据库")
            return []

    def _get_next_person_id(self):
        """获取下一个递增的人员ID (person_1, person_2, ...)"""
        if not self.database:
            return "person_1"
        max_num = 0
        for entry in self.database:
            pid = entry['person_id']
            if pid.startswith('person_'):
                try:
                    num = int(pid.split('_')[1])
                    if num > max_num:
                        max_num = num
                except (IndexError, ValueError):
                    continue
        return f"person_{max_num + 1}"

    def _sync_csv_to_db(self):
        if os.path.exists(self.csv_path):
            try:
                df = pd.read_csv(self.csv_path, encoding='utf-8-sig')
                print(f"CSV文件已加载，包含 {len(df)} 条记录")
            except Exception as e:
                print(f"读取CSV失败: {e}")

    def save_database(self):
        with open(self.db_path, 'wb') as f:
            pickle.dump(self.database, f)
        total_features = sum(len(entry['features']) for entry in self.database)
        print(f"数据库已保存到 {self.db_path}，共 {len(self.database)} 个ID，{total_features} 个特征")
        self.export_to_csv()

    def check_match(self, feature, threshold=0.75):
        """
        检查特征是否匹配数据库中的某个ID

        匹配策略：
        - 与所有ID的所有特征计算相似度
        - 对每个ID取最高相似度
        - 如果某个ID的最高相似度 >= threshold，则匹配成功

        返回:
            (matched, matched_entry, max_similarity, all_similarities)
            - matched: 是否匹配成功
            - matched_entry: 匹配到的数据库条目
            - max_similarity: 最高相似度
            - all_similarities: 所有ID的最高相似度列表 [(person_id, similarity), ...]
        """
        if len(self.database) == 0:
            return False, None, 0.0, []

        all_id_similarities = []

        for entry in self.database:
            pid = entry['person_id']
            # 收集该ID的所有特征
            id_features = np.array([f['feature'] for f in entry['features']])
            # 计算与该ID所有特征的相似度
            similarities = cosine_similarity_batch(feature, id_features)
            # 取最高相似度
            max_sim_for_id = float(np.max(similarities))
            all_id_similarities.append((pid, max_sim_for_id, entry))

        # 按相似度降序排序
        all_id_similarities.sort(key=lambda x: x[1], reverse=True)

        best_pid, best_sim, best_entry = all_id_similarities[0]

        print(f"  匹配结果 (阈值={threshold}):")
        for i, (pid, sim, _) in enumerate(all_id_similarities[:3]):
            print(f"    {i+1}. {pid}: 最高相似度 {sim:.4f}")

        if best_sim >= threshold:
            return True, best_entry, best_sim, all_id_similarities
        return False, None, best_sim, all_id_similarities

    def _add_feature_to_entry(self, entry, feature, image_path, metadata=None):
        """
        将新特征添加到现有ID的特征列表中

        策略：
        - 如果特征数未达上限，直接添加
        - 如果已达上限，移除最旧的特征（或与其他特征最相似的特征）
        """
        img_filename = os.path.basename(image_path)

        feature_entry = {
            'feature': feature.numpy() if torch.is_tensor(feature) else feature,
            'image_path': image_path,
            'image_filename': img_filename,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        if metadata:
            feature_entry.update(metadata)

        entry['features'].append(feature_entry)
        entry['last_updated'] = feature_entry['timestamp']

        # 如果超过最大特征数，移除旧特征
        while len(entry['features']) > self.max_features_per_id:
            removed = entry['features'].pop(0)
            print(f"    特征数超限，移除最旧特征: {removed['image_filename']}")

        return entry

    def add_feature(self, image_path, feature, metadata=None,
                    match_threshold=0.75, add_new_feature_threshold=0.7,
                    auto_add_new_feature=True):
        """
        添加单张图片特征（带去重和特征漂移处理）

        参数:
            image_path: 图片路径
            feature: 特征向量
            metadata: 额外元数据
            match_threshold: 匹配阈值，相似度>=此值认为是同一人
            add_new_feature_threshold: 添加新特征阈值，相似度在此区间内添加新特征
                                       [0, add_new_feature_threshold): 新人，创建新ID
                                       [add_new_feature_threshold, match_threshold): 可能是同一人但不确定，可选是否添加
                                       [match_threshold, 1]: 同一人，自动添加新特征
            auto_add_new_feature: 是否自动将新特征添加到匹配到的ID中

        返回:
            (is_new_person, person_id, message, match_info)
            - is_new_person: 是否为新人
            - person_id: 人员ID
            - message: 描述信息
            - match_info: 匹配详情
        """
        img_filename = os.path.basename(image_path)
        print(f"\n处理图片: {img_filename}")
        print("-" * 60)

        # 第一步：检查是否匹配现有ID
        matched, matched_entry, max_sim, all_sims = self.check_match(feature, match_threshold)

        if matched:
            # 匹配成功：是同一人
            pid = matched_entry['person_id']

            if auto_add_new_feature:
                self._add_feature_to_entry(matched_entry, feature, image_path, metadata)
                msg = f"匹配到 {pid} (相似度 {max_sim:.4f} >= {match_threshold})，已添加新特征 (该ID现共{len(matched_entry['features'])}个特征)"
                print(f"  {msg}")
            else:
                msg = f"匹配到 {pid} (相似度 {max_sim:.4f} >= {match_threshold})，未添加新特征"
                print(f"  {msg}")

            self.save_database()
            return False, pid, msg, {
                'matched': True,
                'matched_id': pid,
                'similarity': max_sim,
                'all_similarities': all_sims
            }

        # 未匹配到：检查是否在"可能匹配"区间
        if all_sims and max_sim >= add_new_feature_threshold:
            best_pid, best_sim, best_entry = all_sims[0]
            msg = f"与 {best_pid} 相似度 {max_sim:.4f} (区间 [{add_new_feature_threshold}, {match_threshold}))，可能是同一人但不确定"
            print(f"  {msg}")
            print(f"  提示：降低 match_threshold 或手动确认")

            # 这里可以选择：保守策略 -> 创建新ID；或者激进策略 -> 添加到现有ID
            # 我们采用保守策略：创建新ID，但记录提示信息
            pass

        # 创建新ID
        person_id = self._get_next_person_id()

        entry = {
            'person_id': person_id,
            'features': [],
            'first_seen': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'model_name': 'osnet_x0_25'
        }

        self._add_feature_to_entry(entry, feature, image_path, metadata)
        self.database.append(entry)

        msg = f"创建新ID {person_id} (最高相似度 {max_sim:.4f} < {match_threshold})"
        print(f"  {msg}")

        self.save_database()
        return True, person_id, msg, {
            'matched': False,
            'max_similarity': max_sim,
            'all_similarities': all_sims
        }

    def add_features(self, image_paths, features, metadata_list=None, **kwargs):
        """批量添加特征"""
        if len(image_paths) != len(features):
            raise ValueError("图片路径数量与特征数量不匹配")

        results = []
        new_count = 0
        existing_count = 0

        for i, (img_path, feature) in enumerate(zip(image_paths, features)):
            md = metadata_list[i] if (metadata_list and i < len(metadata_list)) else None
            is_new, pid, msg, info = self.add_feature(img_path, feature, md, **kwargs)
            results.append((is_new, pid, msg, info))
            if is_new:
                new_count += 1
            else:
                existing_count += 1

        print(f"\n{'='*60}")
        print(f"批量处理完成: 新增 {new_count} 人，匹配到 {existing_count} 人")
        print(f"{'='*60}")
        return results

    def add_feature_from_array(self, image_array, feature, identifier="frame_crop", metadata=None,
                                match_threshold=0.75, add_new_feature_threshold=0.7,
                                auto_add_new_feature=True):
        """
        直接从图像数组添加特征（无需保存到文件）

        参数:
            image_array: numpy数组格式的图像
            feature: 特征向量
            identifier: 用于标识的名称（如 "crop_001"）
            metadata: 额外元数据
            其他参数同 add_feature()

        返回:
            同 add_feature()
        """
        img_filename = identifier

        print(f"\n处理图像: {img_filename}")
        print("-" * 60)

        # 第一步：检查是否匹配现有ID
        matched, matched_entry, max_sim, all_sims = self.check_match(feature, match_threshold)

        if matched:
            # 匹配成功：是同一人
            pid = matched_entry['person_id']

            if auto_add_new_feature:
                self._add_feature_to_entry_array(matched_entry, feature, identifier, image_array, metadata)
                msg = f"匹配到 {pid} (相似度 {max_sim:.4f} >= {match_threshold})，已添加新特征 (该ID现共{len(matched_entry['features'])}个特征)"
                print(f"  {msg}")
            else:
                msg = f"匹配到 {pid} (相似度 {max_sim:.4f} >= {match_threshold})，未添加新特征"
                print(f"  {msg}")

            self.save_database()  # === 关键修复：保存数据库到磁盘 ===
            return False, pid, msg, {
                'matched': True,
                'matched_id': pid,
                'similarity': max_sim,
                'all_similarities': all_sims
            }

        # 未匹配到：检查是否在"可能匹配"区间
        if all_sims and max_sim >= add_new_feature_threshold:
            best_pid, best_sim, best_entry = all_sims[0]
            msg = f"与 {best_pid} 相似度 {max_sim:.4f} (区间 [{add_new_feature_threshold}, {match_threshold}))，可能是同一人但不确定"
            print(f"  {msg}")
            pass

        # 创建新ID
        person_id = self._get_next_person_id()

        entry = {
            'person_id': person_id,
            'features': [],
            'first_seen': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'model_name': 'osnet_x0_25'
        }

        self._add_feature_to_entry_array(entry, feature, identifier, image_array, metadata)
        self.database.append(entry)

        msg = f"创建新ID {person_id} (最高相似度 {max_sim:.4f} < {match_threshold})"
        print(f"  {msg}")

        self.save_database()  # === 关键修复：保存数据库到磁盘 ===

        return True, person_id, msg, {
            'matched': False,
            'max_similarity': max_sim,
            'all_similarities': all_sims
        }

    def _add_feature_to_entry_array(self, entry, feature, identifier, image_array, metadata=None):
        """将图像数组特征添加到现有ID"""
        feature_entry = {
            'feature': feature.numpy() if torch.is_tensor(feature) else feature,
            'identifier': identifier,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        if metadata:
            feature_entry.update(metadata)

        entry['features'].append(feature_entry)
        entry['last_updated'] = feature_entry['timestamp']

        # 如果超过最大特征数，移除旧特征
        while len(entry['features']) > self.max_features_per_id:
            removed = entry['features'].pop(0)
            print(f"    特征数超限，移除最旧特征: {removed.get('identifier', 'unknown')}")

        return entry

    def clear_database(self):
        """清空数据库（程序结束时调用）"""
        self.database = []
        print("数据库已清空")

    def get_all_features_for_id(self, person_id):
        """获取某个ID的所有特征"""
        for entry in self.database:
            if entry['person_id'] == person_id:
                return entry['features']
        return None

    def search_similar(self, query_feature, top_k=5, threshold=0.0):
        """搜索相似的ID（基于该ID的最高相似度）"""
        if len(self.database) == 0:
            print("数据库为空")
            return []

        results = []
        for entry in self.database:
            id_features = np.array([f['feature'] for f in entry['features']])
            similarities = cosine_similarity_batch(query_feature, id_features)
            max_sim = float(np.max(similarities))

            if max_sim >= threshold:
                latest_feature = entry['features'][-1]
                results.append({
                    'person_id': entry['person_id'],
                    'image_path': latest_feature.get('image_path', latest_feature.get('identifier', '')),
                    'image_filename': latest_feature.get('image_filename', latest_feature.get('identifier', '')),
                    'similarity': max_sim,
                    'feature_count': len(entry['features']),
                    'first_seen': entry['first_seen'],
                    'last_updated': entry['last_updated']
                })

        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:top_k]

    def export_to_csv(self, csv_path=None):
        """导出数据库到CSV（每条特征一行）"""
        csv_path = csv_path or self.csv_path

        if len(self.database) == 0:
            print("数据库为空，无法导出")
            return None

        data = []
        for entry in self.database:
            for i, feat_entry in enumerate(entry['features'], 1):
                row = {
                    'person_id': entry['person_id'],
                    'feature_index': i,
                    'image_filename': feat_entry.get('image_filename', feat_entry.get('identifier', '')),
                    'image_path': feat_entry.get('image_path', feat_entry.get('identifier', '')),
                    'timestamp': feat_entry['timestamp'],
                    'first_seen': entry['first_seen'],
                    'last_updated': entry['last_updated'],
                    'feature_count_for_id': len(entry['features']),
                    'model_name': entry['model_name']
                }
                data.append(row)

        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"数据库已导出到 {csv_path} (共 {len(df)} 条特征记录)")
        return df

    def export_features_to_npy(self, npy_dir='feature_vectors'):
        if not os.path.exists(npy_dir):
            os.makedirs(npy_dir)

        for entry in self.database:
            pid = entry['person_id']
            for i, feat_entry in enumerate(entry['features'], 1):
                npy_path = os.path.join(npy_dir, f"{pid}_feat{i}.npy")
                np.save(npy_path, feat_entry['feature'])

        print(f"特征向量已保存到 {npy_dir} 目录")

    def get_statistics(self):
        if not self.database:
            return {
                'total_persons': 0,
                'total_features': 0,
                'avg_features_per_person': 0,
                'model_name': 'osnet_x0_25'
            }

        total_features = sum(len(e['features']) for e in self.database)
        stats = {
            'total_persons': len(self.database),
            'total_features': total_features,
            'avg_features_per_person': total_features / len(self.database),
            'model_name': 'osnet_x0_25'
        }
        return stats

    def print_database(self):
        print(f"\n{'='*90}")
        print(f"特征数据库 (共 {len(self.database)} 个ID)")
        print(f"{'='*90}")
        print(f"{'序号':<4} {'ID':<12} {'特征数':<6} {'最新图片':<25} {'首次出现':<20} {'最后更新':<20}")
        print(f"{'-'*4} {'-'*12} {'-'*6} {'-'*25} {'-'*20} {'-'*20}")

        for i, entry in enumerate(self.database, 1):
            latest_feat = entry['features'][-1]
            img_name = latest_feat.get('image_filename', latest_feat.get('identifier', ''))
            if len(img_name) > 22:
                img_name = img_name[:19] + "..."
            print(f"{i:<4} {entry['person_id']:<12} {len(entry['features']):<6} {img_name:<25} {entry['first_seen']:<20} {entry['last_updated']:<20}")

        print(f"{'='*90}")


class PersonFeatureSystem:
    """完整的行人特征管理系统"""

    def __init__(self, model_path=None, db_path='person_feature_db.pkl',
                 csv_path='person_features.csv', max_features_per_id=5):
        self.extractor_manager = FeatureExtractorManager(model_path=model_path)
        self.feature_db = FeatureDatabase(
            db_path=db_path,
            csv_path=csv_path,
            max_features_per_id=max_features_per_id
        )

    def add_image(self, image_path, match_threshold=0.75, add_new_feature_threshold=0.7,
                  auto_add_new_feature=True):
        """添加单张图片"""
        if not os.path.exists(image_path):
            msg = f"错误: 图片文件不存在 {image_path}"
            print(msg)
            return False, None, msg, None

        features = self.extractor_manager.extract_features([image_path])
        feature = features[0]

        metadata = None
        try:
            img = Image.open(image_path)
            metadata = {
                'image_size': img.size,
                'image_format': os.path.splitext(image_path)[1],
                'file_size_kb': os.path.getsize(image_path) / 1024
            }
        except Exception as e:
            print(f"读取图片元数据失败: {e}")

        return self.feature_db.add_feature(
            image_path, feature, metadata,
            match_threshold=match_threshold,
            add_new_feature_threshold=add_new_feature_threshold,
            auto_add_new_feature=auto_add_new_feature
        )

    def add_images(self, image_paths, **kwargs):
        """批量添加图片"""
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        valid_images = [p for p in image_paths if os.path.exists(p)]
        for p in image_paths:
            if not os.path.exists(p):
                print(f"警告: 图片文件不存在 {p}")

        if not valid_images:
            print("错误: 没有有效的图片文件")
            return None

        print(f"\n正在提取 {len(valid_images)} 张图片的特征...")
        features = self.extractor_manager.extract_features(valid_images)

        metadata_list = []
        for img_path in valid_images:
            try:
                img = Image.open(img_path)
                metadata_list.append({
                    'image_size': img.size,
                    'image_format': os.path.splitext(img_path)[1],
                    'file_size_kb': os.path.getsize(img_path) / 1024
                })
            except Exception:
                metadata_list.append({})

        return self.feature_db.add_features(valid_images, features, metadata_list, **kwargs)

    def search(self, query_image_path, top_k=5, threshold=0.0):
        if not os.path.exists(query_image_path):
            print(f"错误: 查询图片不存在 {query_image_path}")
            return []

        query_features = self.extractor_manager.extract_features([query_image_path])[0]
        results = self.feature_db.search_similar(query_features, top_k=top_k, threshold=threshold)

        print(f"\n搜索结果 (查询图片: {os.path.basename(query_image_path)}):")
        print("-" * 70)
        for i, result in enumerate(results, 1):
            print(f"{i}. ID: {result['person_id']} | 特征数: {result['feature_count']} | "
                  f"相似度: {result['similarity']:.4f} | 图片: {result['image_filename']}")

        return results

    def save(self):
        self.feature_db.save_database()

    def add_image_array(self, image_array, identifier="crop",
                        match_threshold=0.75, add_new_feature_threshold=0.7,
                        auto_add_new_feature=True):
        """
        直接从图像数组添加特征到数据库

        参数:
            image_array: numpy数组格式的图像
            identifier: 标识符
            其他参数同 add_image()

        返回:
            同 add_image()
        """
        features = self.extractor_manager.extract_features_from_image_array(image_array)
        feature = features[0]

        metadata = {
            'image_size': image_array.shape[:2],
        }

        return self.feature_db.add_feature_from_array(
            image_array, feature, identifier, metadata,
            match_threshold=match_threshold,
            add_new_feature_threshold=add_new_feature_threshold,
            auto_add_new_feature=auto_add_new_feature
        )

    def clear_database(self):
        """清空数据库"""
        self.feature_db.clear_database()

    def export_to_csv(self, csv_path=None):
        """导出数据库到CSV"""
        return self.feature_db.export_to_csv(csv_path)


if __name__ == "__main__":
    model_path = r"osnet_ain_x0_25_imagenet.pyth"

    image_list = [
        r"data\1.png",
        r"data\2.jpg",
        r"data\3.jpg"
    ]

    system = PersonFeatureSystem(
        model_path=model_path,
        db_path='person_feature_db.pkl',
        csv_path='person_features.csv',
        max_features_per_id=5  # 每个ID最多保留5个特征
    )

    system.add_images(
        image_list,
        match_threshold=0.75,       # 主要匹配阈值
        add_new_feature_threshold=0.7,  # 添加新特征阈值
        auto_add_new_feature=True    # 匹配成功时自动添加新特征
    )

    system.feature_db.print_database()

    stats = system.feature_db.get_statistics()
    print(f"\n数据库统计信息:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print(f"\n导出数据库...")
    system.feature_db.export_to_csv('person_features.csv')
    system.feature_db.export_features_to_npy('feature_vectors')

    system.save()

    print(f"\n{'='*60}")
    print("特征提取和保存完成！")
    print(f"{'='*60}")

