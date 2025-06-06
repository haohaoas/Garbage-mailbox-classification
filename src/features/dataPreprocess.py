import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
matplotlib.use('TkAgg')
df = pd.read_excel("cleaned_data.xlsx")
sample = df.iloc[0, 1:]  # 第一个样本的光谱数据（跳过编号列）
wavenumbers = df.columns[1:].astype(int)  # 假设列名是波数，转为整数

plt.plot(wavenumbers, sample)
plt.gca().invert_xaxis()
plt.xlabel("Wavenumber (cm⁻¹)")
plt.ylabel("Intensity")
plt.title("Sample 1 Spectrum")
plt.grid(True)
plt.show()


X = df.iloc[:, 1:]  # 只取光谱数据，不含 No

# 标准化
X_scaled = StandardScaler().fit_transform(X)

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 可视化
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7)
plt.title("PCA Projection of Infrared Spectra (Unlabeled)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True)
plt.tight_layout()
plt.show()


kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# 把聚类结果添加回原始 DataFrame
df["cluster"] = clusters

# 可视化每个聚类中心的平均光谱
for i in range(kmeans.n_clusters):
    cluster_mean = df[df["cluster"] == i].iloc[:, 1:-1].mean(axis=0)  # 平均谱
    plt.plot(wavenumbers, cluster_mean, label=f'Cluster {i}')

plt.gca().invert_xaxis()
plt.title("Mean Spectrum of Each Cluster")
plt.xlabel("Wavenumber (cm⁻¹)")
plt.ylabel("Mean Intensity")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



# 计算每个波数在各个聚类中的均值差异
variances = []

for col in df.columns[1:-1]:  # 不包括 No 和 cluster 列
    group_means = df.groupby('cluster')[col].mean()
    variance = group_means.max() - group_means.min()
    variances.append((col, variance))

# 找出前10个差异最大的波数
top_var = sorted(variances, key=lambda x: x[1], reverse=True)[:10]
print("波数差异最大的点：", top_var)

import seaborn as sns

selected_wavenumbers = [655, 659, 657, 661, 663, 653, 665, 667, 669, 671]

# for w in selected_wavenumbers:
#     sns.boxplot(x='cluster', y=w, data=df)
#     plt.title(f'Intensity at {w} cm⁻¹ by Cluster')
#     plt.xlabel('Cluster')
#     plt.ylabel('Intensity')
#     plt.tight_layout()
#     plt.show()
key_wave_means = df.groupby('cluster')[selected_wavenumbers].mean()
# print(key_wave_means)

for w in selected_wavenumbers:
    for i in range(kmeans.n_clusters):
        avg = df[df['cluster'] == i][w].mean()
        plt.bar(f'{w}-C{i}', avg)

plt.title("Mean Intensities at Key Wavenumbers by Cluster")
plt.ylabel("Mean Intensity")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# important_wavenumbers = [w for w, _ in top_var]
# df_selected = df[["No"] + important_wavenumbers + ["cluster"]]
# df_selected.to_excel("top10_wavenumbers_with_cluster.xlsx", index=False)

