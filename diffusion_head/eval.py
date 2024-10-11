import torch
from torch.utils.data import Dataset, DataLoader
import pickle

class FeatureDataset(Dataset):
    def __init__(self, features_file, labels_file):
        # 读取特征和标签文件
        with open(features_file, 'rb') as f:
            self.features = pickle.load(f)
        with open(labels_file, 'rb') as f:
            self.labels = pickle.load(f)
        
        # 确保特征和标签长度一致
        assert len(self.features) == len(self.labels), "Features and labels must have the same length"

    def __len__(self):
        # 返回数据的总长度
        return len(self.features)

    def __getitem__(self, idx):
        # 根据索引返回特征和对应的标签
        feature = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return feature, label

import torch
import torch.nn as nn

class CrossAttentionModel(nn.Module):
    def __init__(self, feature_dim, query_dim, num_heads=4, hidden_dim=128):
        super(CrossAttentionModel, self).__init__()
        
        # 可学习的 query，初始化为随机值
        self.query = nn.Parameter(torch.randn(1, query_dim))
        
        # Cross-Attention 机制
        self.cross_attention = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads, batch_first=True)
        
        # 一个简单的前馈网络
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # 假设最后是一个二分类任务
        )
        
    def forward(self, feature1, feature2):
        # 将两个特征拼接在一起 (batch_size, 2, feature_dim)
        features = torch.stack([feature1, feature2], dim=1)
        
        # 将 query 扩展到 batch 大小 (batch_size, 1, query_dim)
        query = self.query.expand(features.size(0), -1, -1)
        
        # 通过 cross-attention 学习特征
        attended_features, _ = self.cross_attention(query, features, features)
        
        # 将 attention 后的特征通过前馈网络
        output = self.fc(attended_features.squeeze(1))
        return output


if __name__ == "__main__":
    pass
