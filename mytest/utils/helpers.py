import numpy as np
import torch


def cosine_similarity(vec1, vec2):
    """
    计算两个向量的余弦相似度

    参数:
        vec1: 向量1 (numpy数组、torch张量或list)
        vec2: 向量2 (numpy数组、torch张量或list)

    返回:
        余弦相似度 (float), 范围 [0, 1]
    """
    if vec1 is None or vec2 is None:
        return 0.0

    # 统一转换为numpy数组
    if torch.is_tensor(vec1):
        vec1 = vec1.numpy()
    if torch.is_tensor(vec2):
        vec2 = vec2.numpy()

    vec1 = np.array(vec1, dtype=np.float32).flatten()
    vec2 = np.array(vec2, dtype=np.float32).flatten()

    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(dot_product / (norm1 * norm2))


def cosine_similarity_batch(query_feature, db_features):
    """
    批量计算余弦相似度（查询特征与数据库所有特征的相似度）

    参数:
        query_feature: 查询特征向量
        db_features: 特征列表/矩阵 (shape: [n_features, feature_dim])

    返回:
        相似度数组 (shape: [n_features])
    """
    if query_feature is None or db_features is None or len(db_features) == 0:
        return np.array([])

    # 统一转换为numpy数组
    if torch.is_tensor(query_feature):
        query_feature = query_feature.numpy()
    if torch.is_tensor(db_features):
        db_features = db_features.numpy()

    if query_feature.ndim == 1:
        query_feature = query_feature.reshape(1, -1)

    db_features = np.array(db_features, dtype=np.float32)
    if db_features.ndim == 1:
        db_features = db_features.reshape(1, -1)

    dot_products = np.dot(query_feature, db_features.T)

    query_norm = np.linalg.norm(query_feature, axis=1, keepdims=True)
    db_norms = np.linalg.norm(db_features, axis=1, keepdims=True)

    if np.any(query_norm == 0) or np.any(db_norms == 0):
        return np.zeros(query_feature.shape[0])

    similarities = dot_products / (query_norm * db_norms.T)

    return similarities.flatten()


def box_iou(box1, box2):
    """
    计算两个边界框的IoU (Intersection over Union)

    参数:
        box1: 边界框1 [x1, y1, x2, y2]
        box2: 边界框2 [x1, y1, x2, y2]

    返回:
        IoU值 (float), 范围 [0, 1]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / (union + 1e-6)


def box_iou_batch(box1, boxes2):
    """
    批量计算IoU：一个框与多个框的IoU

    参数:
        box1: 单个边界框 [x1, y1, x2, y2]
        boxes2: 边界框数组 (shape: [N, 4])

    返回:
        IoU数组 (shape: [N])
    """
    x1 = np.maximum(box1[0], boxes2[:, 0])
    y1 = np.maximum(box1[1], boxes2[:, 1])
    x2 = np.minimum(box1[2], boxes2[:, 2])
    y2 = np.minimum(box1[3], boxes2[:, 3])

    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1 + area2 - intersection

    return intersection / (union + 1e-6)


def cosine_distance(vec1, vec2):
    """
    计算余弦距离 = 1 - 余弦相似度

    参数:
        vec1: 向量1
        vec2: 向量2

    返回:
        余弦距离 (float), 范围 [0, 1]
    """
    return 1.0 - cosine_similarity(vec1, vec2)

