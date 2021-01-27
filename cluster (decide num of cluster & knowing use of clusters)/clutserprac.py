import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.cluster import KMeans
from sklearn import preprocessing

#https://medium.com/code-to-express/k-means-clustering-for-beginners-using-python-from-scratch-f20e79c8ad00
#https://stackoverflow.com/questions/59690058/kmeans-in-python-centroids-not-in-correct-position-how-do-i-unscale-them

data = pd.read_csv("eg.csv")
x = data.copy()
xscale = preprocessing.scale(x)

# determine the num of cluster needed using elbow method
wcss=[]
for i in range(1,30):  # -- the range is the num of rows
    kmeans = KMeans(i)
    kmeans.fit(xscale)
    wcss.append(kmeans.inertia_)

# graph elbow method
'''
plt.plot(range(1,30), wcss)
plt.xlabel("num of cluster")
plt.ylabel("wcss")
plt.savefig("wcss graph.png")
'''
# from graph see need 4 clusters
kmeans = KMeans(4)
kmeans.fit(xscale)
cluster = data.copy()
cluster['cluspred'] = kmeans.fit_predict(xscale)

plt.scatter(cluster['Satisfaction'], cluster['Loyalty'],
            c=cluster['cluspred'], cmap='rainbow')

kmeans_unscale = KMeans(n_clusters=4).fit(x)
centroids = kmeans_unscale.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", c='b', s=50)

plt.xlabel("Satisfaction")
plt.ylabel("Loyalty")
plt.savefig("clusters.png")
plt.show()
plt.close()