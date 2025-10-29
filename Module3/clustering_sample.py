import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn_extra.cluster import KMedoids
from kmodes.kmodes import KModes
from pyclustering.cluster.kmedians import kmedians
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering


#Input Dateset
train_feat = pd.read_csv("Nutritions.csv")
train_feat = (train_feat - train_feat.mean()) / train_feat.std()

#kmeans
model = KMeans(n_clusters=3)
model.fit(train_feat)


# # filter rows based on cluster
# first_cluster = train_feat.loc[model.labels_ == 0,:]
# second_cluster = train_feat.loc[model.labels_ == 1,:]
# third_cluster = train_feat.loc[model.labels_ == 2,:]
# #
# # Plotting the results
# plt.scatter(first_cluster.loc[:, 'calories'], first_cluster.loc[:, 'fat'], color='red')
# plt.scatter(second_cluster.loc[:, 'calories'], second_cluster.loc[:, 'fat'], color='black')
# plt.scatter(third_cluster.loc[:, 'calories'], third_cluster.loc[:, 'fat'], color='green')
# plt.show()
#
# inertias = []
# for i in range(1,11):
#     kmeans = KMeans(n_clusters=i).fit(train_feat)
#     inertias.append(kmeans.inertia_) #inertia: Sum of squared distances of data points from their cluster centers (WCSS)

# #kmode
# model =KModes(n_clusters=4)
# model.fit(train_feat)
# print(model.labels_)


# plt.plot(range(1,11), inertias, marker='o')
# plt.title('Elbow method')
# plt.xlabel('Number of clusters')
# plt.ylabel('Sum of squared distances')
# plt.show()


# #Agnes
# linkage_data = linkage(train_feat, method='single', metric='euclidean')
# dendrogram(linkage_data, truncate_mode = 'level' ,p=7 )
# plt.show()

#
#DBScan
dbscan = DBSCAN(eps=0.6, min_samples=5)
dbscan.fit(train_feat)
print(np.unique(dbscan.labels_))