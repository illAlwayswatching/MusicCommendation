import pandas as pd
import numpy as np
import networkx as nx
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# 读取用户-歌曲评分数据（user_id, song_id, rating）
music_ratings = pd.read_csv("music_ratings.csv")
social_network = pd.read_csv("social_network.csv")  # (user_id, friend_id)

# 评分数据格式化
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(music_ratings[['user_id', 'song_id', 'rating']], reader)

# 划分训练集 & 测试集
trainset, testset = train_test_split(data, test_size=0.2)

# 计算社交网络相似度
G = nx.Graph()
for _, row in social_network.iterrows():
    G.add_edge(row["user_id"], row["friend_id"])

def social_similarity(user1, user2):
    """ 计算用户之间的社交相似度（基于共同好友数）"""
    if G.has_node(user1) and G.has_node(user2):
        common_friends = len(set(G.neighbors(user1)) & set(G.neighbors(user2)))
        total_friends = len(set(G.neighbors(user1)) | set(G.neighbors(user2)))
        return common_friends / total_friends if total_friends > 0 else 0
    return 0

# 训练 SVD 模型
svd = SVD(n_factors=50, reg_all=0.02, lr_all=0.005, n_epochs=20)
svd.fit(trainset)

# 计算 SVD 预测
predictions = svd.test(testset)
print("RMSE:", accuracy.rmse(predictions))

# 训练 SVD 模型
svd = SVD(n_factors=50, reg_all=0.02, lr_all=0.005, n_epochs=20)
svd.fit(trainset)

# 计算 SVD 预测
predictions = svd.test(testset)
print("RMSE:", accuracy.rmse(predictions))

# 设定社交网络影响权重
alpha = 0.7  

def hybrid_recommend(user_id, song_id):
    """ 结合 SVD 预测 + 社交网络影响进行推荐 """
    svd_score = svd.predict(user_id, song_id).est  # SVD 预测评分
    social_score = np.mean([social_similarity(user_id, friend) for friend in G.neighbors(user_id)] or [0])
    final_score = alpha * svd_score + (1 - alpha) * social_score
    return final_score

# 示例：为用户 10 推荐歌曲 100
print("推荐分数:", hybrid_recommend(10, 100))

def precision_at_k(k=10):
    relevant = 0
    total = 0

    for user_id in music_ratings['user_id'].unique():
        songs = music_ratings[music_ratings['user_id'] == user_id]['song_id'].tolist()
        predicted_scores = {song: hybrid_recommend(user_id, song) for song in songs}
        top_k_songs = sorted(predicted_scores, key=predicted_scores.get, reverse=True)[:k]

        # 计算 Precision@K
        relevant += sum(1 for song in top_k_songs if (music_ratings[(music_ratings['user_id'] == user_id) & (music_ratings['song_id'] == song)]['rating'] >= 4).any())
        total += k

    return relevant / total

print("Precision@10:", precision_at_k(10))

