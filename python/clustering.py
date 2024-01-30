import numpy as np
from sklearn.cluster import KMeans
def apriori_clustering(X, Y, n_clusters=5):
    '''
    Clustering of the (x,y) pairs based on their coordinates
    '''
    data = np.column_stack((X, Y))
    print(data[:5])
    # K-means clustering
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(data)
    labels = kmeans.labels_

    return labels

    
    
    

