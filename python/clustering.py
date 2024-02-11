import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier

def apriori_clustering(X, Y, n_clusters=5):
    '''
    Clustering of the (x,y) pairs based on the coordinates of the x-y vector
    '''
    data = X-Y
    # K-means clustering
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(data)
    labels = kmeans.labels_

    return labels

def predict_cluster_knn(pairs, labels, new_point, n_neighbors=10):
    '''
    Predicts the label of a new datapoint using KNN
    '''
    # Create KNN classifier
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    
    # Fit the classifier with the training data
    knn.fit(pairs, labels)
    
    # Predict the label of the new array
    predicted_label = knn.predict(new_point)
    
    return predicted_label[0]

    
    
    

