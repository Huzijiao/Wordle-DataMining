import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# 导入数据
data = pd.read_excel('./data/beforeDBSCAN.xlsx') # 默认读取第一个sheet
# dataset = np.array(data.iloc[1:,13:].values.tolist()[::-1]) #读取指定列的所有行数据
beer = data.iloc[0:, 7:].copy()
# print(beer)
# .values.tolist()
Xunguiyi = beer
# normalizer=Normalizer(norm='l2') # L2范式
# print("after transform:\n",normalizer.transform(Xunnorm))
scaler = MinMaxScaler()
print('自动归一化结果:\n{}'.format(scaler.fit_transform(Xunguiyi)))

colors = np.array(['red','green','blue','yellow','orange',"purple"])

X = pd.DataFrame(scaler.fit_transform(Xunguiyi))
X.columns = beer.columns

# X = beer.copy()
# [["1 try", "2 tries", "3 tries", "4 tries", "5 tries", "6 tries", "7 or more tries (X)"]]
# 设置半径为10，最小样本量为2，建模
db = DBSCAN(eps=0.06, min_samples=2).fit(X)

labels = db.labels_
for i in range(len(labels)):
    if labels[i]==-1 and beer.values.tolist()[i][0]>4 :
        labels[i] = 2
beer['cluster_db'] = labels  # 在数据集最后一列加上经过DBSCAN聚类后的结果
beer.sort_values('cluster_db')
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
print(n_clusters_)

# 注：cluster列是kmeans聚成3类的结果；cluster2列是kmeans聚类成2类的结果；scaled_cluster列是kmeans聚类成3类的结果（经过了数据标准化）

# 画出在不同两个指标下样本的分布情况
pd.plotting.scatter_matrix(X, c=colors[beer.cluster_db], figsize=(10, 10), s=100)
x_ticks = np.linspace(0, 1, 6)  # 产生区间在-5至4间的10个均匀数值
plt.xticks(x_ticks)  # 将linspace产生的新的十个值传给xticks( )函数，用以改变坐标刻度
plt.yticks(x_ticks)  # 将linspace产生的新的十个值传给xticks( )函数，用以改变坐标刻度
plt.savefig('./scatter_matrix.png')
plt.show()

# 我们可以从上面这个图里观察聚类效果的好坏，但是当数据量很大，或者指标很多的时候，观察起来就会非常麻烦。

# # 保存到文件

df = pd.DataFrame(beer)
write = pd.ExcelWriter("DBSCAN_new.xlsx")   # 新建xlsx文件
df.to_excel(write, sheet_name='Sheet1', index=False)  # 写入文件的Sheet1
write.save()  # 这里一定要保存

from sklearn import metrics
# 就是下面这个函数可以计算轮廓系数（sklearn真是一个强大的包）
score = metrics.silhouette_score(X,beer.cluster_db)
print(score)

A = []  # 列表A用以记录接下来循环中每次DBSCAN时的参数eps
B = []  # 列表B用以记录接下来循环中每次DBSCAN之后聚出了几个簇
C = []  # 列表C用以记录接下来循环中每次DBSCAN之后的被视为异常值的对象个数

# 通过循环不同的eps的值（从0.1到5，每次增加0.1），来绘制出肘线图
for i in np.linspace(0.01, 0.1, 90):
    db = DBSCAN(eps=i, min_samples=2).fit(X)
    # print(db)

    # 构造一个矩阵core_samples_mask，其维度与矩阵db.labels_一致，并为其初始化为全0；这个函数方便的构造了新矩阵，无需参数指定shape大小
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    # labels_参数是分类结果，即训练后每条数据所属簇。这里是一个（1795，）的一维向量，每一位是每个样本进行聚类后所属的类别

    # 把矩阵core_samples_mask中核心对象对应的位置被置为True
    core_samples_mask[db.core_sample_indices_] = True
    # db.core_sample_indices_参数包含了每个核心对象的索引，是一个（n,）的一维向量，
    # 每一位是一个核心对象在训练数据中的索引，n在每次DBSCAN后均不一样

    labels = db.labels_
    # n_clusters用于计录聚出了几个簇
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    # set()方法把输入的列表转换为集合，实现了相同元素的筛除。减去的一项是为了不把异常值算作单独的一类簇

    sum = 0
    for t in labels:
        if t == -1:
            sum = sum + 1
    C.append(sum)

    A.append(i)
    B.append(int(n_clusters_))

results = pd.DataFrame([A,B,C]).T
results.columns = ['eps','Number of clusters','Number of outliers']
results.plot(x='eps',y='Number of clusters',figsize=(10,6))
plt.show()

A = []  # 列表A用以记录接下来循环中每次DBSCAN时的参数min_samples
B = []  # 列表B用以记录接下来循环中每次DBSCAN之后聚出了几个簇
C = []  # 列表C用以记录接下来循环中每次DBSCAN之后的被视为异常值的对象个数

# 通过循环不同的min_samples的值（从1到15，每次增加1），来绘制出肘线图
for i in np.linspace(1, 15, 15, dtype=int):
    db = DBSCAN(eps=0.06, min_samples=i).fit(X)
    # print(db)

    # 构造一个矩阵core_samples_mask，其维度与矩阵db.labels_一致，并为其初始化为全0；这个函数方便的构造了新矩阵，无需参数指定shape大小
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    # labels_参数是分类结果，即训练后每条数据所属簇。这里是一个（1795，）的一维向量，每一位是每个样本进行聚类后所属的类别

    # 把矩阵core_samples_mask中核心对象对应的位置被置为True
    core_samples_mask[db.core_sample_indices_] = True
    # db.core_sample_indices_参数包含了每个核心对象的索引，是一个（n,）的一维向量，
    # 每一位是一个核心对象在训练数据中的索引，n在每次DBSCAN后均不一样

    labels = db.labels_
    # n_clusters用于计录聚出了几个簇
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    # set()方法把输入的列表转换为集合，实现了相同元素的筛除。减去的一项是为了不把异常值算作单独的一类簇

    sum = 0
    for t in labels:
        if t == -1:
            sum = sum + 1
    C.append(sum)

    A.append(i)
    B.append(int(n_clusters_))
results = pd.DataFrame([A,B,C]).T
results.columns = ['Min_samples','Number of clusters','Number of outliers']
results.plot(x='Min_samples',y='Number of clusters',figsize=(10,6))
plt.show()