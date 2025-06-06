import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.signal import savgol_filter
from sklearn.feature_selection import VarianceThreshold
matplotlib.use('TkAgg')
#获取数据
df=pd.read_csv('../features/tran_data_100.csv')
#删除No列
X=df.iloc[:, 1:]

pca=PCA(n_components=2)
X_pca =pca.fit_transform(X)
plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.title('PCA Visualization of Spectral Data')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()

selector = VarianceThreshold(threshold=0.01)
X_var = selector.fit_transform(X)
# 去除所有方差为 0 的波段
# 假设初步猜测分3类
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X_var)

#  肘部法则
inertias = []
inertiaz = []
k_range = range(1, 10)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42).fit(X_var)
    inertias.append(kmeans.inertia_)
plt.plot(k_range, inertias, marker='o')

plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()

k_range1=range(2, 10)
for k in k_range1:
    kmeans = KMeans(n_clusters=k, random_state=42).fit(X_var)

    inertiaz.append(silhouette_score(X_var, kmeans.labels_))
plt.plot(k_range1, inertiaz, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()