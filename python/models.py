import pickle
from abc import abstractmethod

import gurobipy as gp
from gurobipy import GRB

import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
import matplotlib.pyplot as plt


class BaseModel(object):
    """
    Base class for models, to be used as coding pattern skeleton.
    Can be used for a model on a single cluster or on multiple clusters"""

    def __init__(self):
        """Initialization of your model and its hyper-parameters"""
        pass

    @abstractmethod
    def fit(self, X, Y):
        """Fit function to find the parameters according to (X, Y) data.
        (X, Y) formatting must be so that X[i] is preferred to Y[i] for all i.

        Parameters
        ----------
        X: np.ndarray
            (n_samples, n_features) features of elements preferred to Y elements
        Y: np.ndarray
            (n_samples, n_features) features of unchosen elements
        """
        # Customize what happens in the fit function
        return

    @abstractmethod
    def predict_utility(self, X):
        """Method to call the decision function of your model

        Parameters:
        -----------
        X: np.ndarray
            (n_samples, n_features) list of features of elements
        """
        # Customize what happens in the predict utility function
        return

    def predict_preference(self, X, Y):
        """Method to predict which pair is preferred between X[i] and Y[i] for all i.
        Returns a preference for each cluster.

        Parameters
        -----------
        X: np.ndarray
            (n_samples, n_features) list of features of elements to compare with Y elements of same index
        Y: np.ndarray
            (n_samples, n_features) list of features of elements to compare with X elements of same index

        Returns
        -------
        np.ndarray:
            (n_samples, n_clusters) array of preferences for each cluster. 1 if X is preferred to Y, 0 otherwise
        """
        X_u = self.predict_utility(X)
        Y_u = self.predict_utility(Y)

        return (X_u - Y_u > 0).astype(int)

    def predict_cluster(self, X, Y):
        """Predict which cluster prefers X over Y THE MOST, meaning that if several cluster prefer X over Y, it will
        be assigned to the cluster showing the highest utility difference). The reversal is True if none of the clusters
        prefer X over Y.
        Compared to predict_preference, it indicates a cluster index.

        Parameters
        -----------
        X: np.ndarray
            (n_samples, n_features) list of features of elements to compare with Y elements of same index
        Y: np.ndarray
            (n_samples, n_features) list of features of elements to compare with X elements of same index

        Returns
        -------
        np.ndarray:
            (n_samples, ) index of cluster with highest preference difference between X and Y.
        """
        X_u = self.predict_utility(X)
        Y_u = self.predict_utility(Y)

        return np.argmax(X_u - Y_u, axis=1)

    def save_model(self, path):
        """Save the model in a pickle file. Don't hesitate to change it in the child class if needed

        Parameters
        ----------
        path: str
            path indicating the file in which the model will be saved
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load_model(clf, path):
        """Load a model saved in a pickle file. Don't hesitate to change it in the child class if needed

        Parameters
        ----------
        path: str
            path indicating the path to the file to load
        """
        with open(path, "rb") as f:
            model = pickle.load(f)
        return model


class RandomExampleModel(BaseModel):
    """Example of a model on two clusters, drawing random coefficients.
    You can use it to understand how to write your own model and the data format that we are waiting for.
    This model does not work well but you should have the same data formatting with TwoClustersMIP.
    """

    def __init__(self):
        self.seed = 444
        self.weights = self.instantiate()

    def instantiate(self):
        """No particular instantiation"""
        return []

    def fit(self, X, Y):
        """fit function, sets random weights for each cluster. Totally independant from X & Y.

        Parameters
        ----------
        X: np.ndarray
            (n_samples, n_features) features of elements preferred to Y elements
        Y: np.ndarray
            (n_samples, n_features) features of unchosen elements
        """
        np.random.seed(self.seed)
        indexes = np.random.randint(0, 2, (len(X)))
        num_features = X.shape[1]
        weights_1 = np.random.rand(num_features)
        weights_2 = np.random.rand(num_features)

        weights_1 = weights_1 / np.sum(weights_1)
        weights_2 = weights_2 / np.sum(weights_2)
        self.weights = [weights_1, weights_2]
        return self

    def predict_utility(self, X):
        """Simple utility function from random weights.

        Parameters:
        -----------
        X: np.ndarray
            (n_samples, n_features) list of features of elements
        """
        return np.stack([np.dot(X, self.weights[0]), np.dot(X, self.weights[1])], axis=1)


class TwoClustersMIP(BaseModel):
    """Skeleton of MIP you have to write as the first exercise.
    You have to encapsulate your code within this class that will be called for evaluation.
    """

    def __init__(self, n_pieces, n_clusters):
        """Initialization of the MIP Variables

        Parameters
        ----------
        n_pieces: int
            Number of pieces for the utility function of each feature.
        n_clusters: int
            Number of clusters to implement in the MIP.
        """
        self.seed = 123
        self.n_pieces = n_pieces
        self.n_clusters = n_clusters
        self.model = self.instantiate()

    def instantiate(self):
        """Instantiation of the MIP Variables - To be completed."""
        model = gp.Model("MIP")
        model.Params.LogToConsole = 0
        model.Params.seed = self.seed

        return model

    def fit(self, X, Y):
        """Estimation of the parameters - To be completed.

        Parameters
        ----------
        X: np.ndarray
            (n_samples, n_features) features of elements preferred to Y elements
        Y: np.ndarray
            (n_samples, n_features) features of unchosen elements
        """
        n_samples = X.shape[0]
        n_features = X.shape[1]
        n_pieces = self.n_pieces
        n_clusters = self.n_clusters

        # We create the variable corresponding to the values of the utilies functions that we want to find
        u = {}
        for i in range(n_features):
            for l in range(n_pieces+1):
                for k in range(n_clusters):
                    u[i, l, k] = self.model.addVar(lb = 0, ub = 1, vtype=GRB.CONTINUOUS, name=f"u_{i}_{l}_{k}")

        # We create the variable corresponding to the clusters, the value is 1 if it is in the corresponding cluster
        z = {}
        for k in range(n_clusters):
            for j in range(n_samples):
                z[j, k] = self.model.addVar(vtype=GRB.BINARY, name=f"z_{j}_{k}")

        # We create the variables of surestimation and underestimation of the score function, that we want to minimize
        sigma_plus = {}
        sigma_minus = {}
        for j in range(n_samples):
            for k in range(n_clusters):
                for x_or_y in [0,1]:
                    sigma_plus[j,k,x_or_y] = self.model.addVar(lb = 0, vtype=GRB.CONTINUOUS, name=f"sigma_plus_x_{j}_{k}_{x_or_y}")
                    sigma_minus[j,k,x_or_y] = self.model.addVar(lb = 0, vtype=GRB.CONTINUOUS, name=f"sigma_minus_x_{j}_{k}_{x_or_y}")

        # We add the constraint : the utility functions are increasing
        for i in range(n_features):
            for l in range(n_pieces):
                for k in range(n_clusters):
                    self.model.addConstr(u[i, l+1, k] - u[i, l, k] >= 0)

        # Constraint for the clusters, should choose only one cluster for each sample
        for j in range(n_samples):
            self.model.addConstr(gp.quicksum([z[j, k] for k in range(n_clusters)]) == 1)

        # Constraint for the clusters, should choose only one cluster for each sample
        for k in range(n_clusters):
            self.model.addConstr(gp.quicksum([u[i, n_pieces, k] for i in range(n_features)]) == 1)

        # Cut the abscissa axis into n_pieces intervals
        self.x = np.zeros((n_features, n_pieces+1), dtype = float)
        for i in range(n_features):
            for l in range(n_pieces+1):
                self.x[i, l] = l / n_pieces

        # Function to calculate utility
        def compute_utility(k, features, x_or_y,j):
            utility = 0
            for i, feature_value in enumerate(features):
                for l in range(n_pieces):
                    if self.x[i, l] <= feature_value <= self.x[i, l+1]:
                        utility += u[i, l, k] + ((u[i, l+1, k]-u[i, l, k])/(self.x[i,l+1]-self.x[i,l])) * (feature_value - self.x[i,l]) + (sigma_plus[j,k,x_or_y] - sigma_minus[j,k,x_or_y])
                        break
            return utility

        # Adding the constraints for choosing the right cluster and the right utility function
        Maj = 100
        epsilon = 10**-3
        for j in range(n_samples):
            for k in range(n_clusters):
                self.model.addConstr(compute_utility(k, X[j],0,j) - compute_utility(k, Y[j],1,j) - epsilon >= Maj * (z[j,k] - 1))
                self.model.addConstr(compute_utility(k, X[j],0,j) - compute_utility(k, Y[j],1,j) + epsilon <= Maj * z[j,k])

        # We update the model before adding abjective and optimizing
        self.model.update()
        self.model.setObjective(gp.quicksum([sigma_plus[j,k,x_or_y] + sigma_minus[j,k,x_or_y] for j in range(n_samples) for k in range(n_clusters) for x_or_y in [0,1]]), GRB.MINIMIZE)

        self.model.optimize()

        # Get the optimal values of u and z 
        if self.model.status == GRB.OPTIMAL:
            u_opt = np.zeros((n_features, n_pieces+1, n_clusters))
            z_opt = np.zeros((n_samples, n_clusters))
            for i in range(n_features):
                for l in range(n_pieces+1):
                    for k in range(n_clusters):
                        u_opt[i, l, k] = u[i, l, k].x
            for j in range(n_samples):
                for k in range(n_clusters):
                    z_opt[j, k] = z[j, k].x
        else:
            print('No solution')

        self.u = u_opt
        self.z = z_opt


    def predict_utility(self, X):
        """Return Decision Function of the MIP for X. - To be completed.

        Parameters:
        -----------
        X: np.ndarray
            (n_samples, n_features) list of features of elements
        """
        # Use self.u obtained by the fit method to predict the utility of X, we also use the self.x[i, l] for the abscissa axis
        n_samples = X.shape[0]
        n_features = X.shape[1]
        utility = np.zeros((n_samples, self.n_clusters), dtype = float)
        for j in range(n_samples):
            for k in range(self.n_clusters):
                utility[j, k] = 0
                for i, feature_value in enumerate(X[j]):
                    for l in range(self.n_pieces):
                        if self.x[i, l] <= feature_value <= self.x[i, l+1]:
                            utility[j, k] += self.u[i, l, k] + ((self.u[i, l+1, k]-self.u[i, l, k])/(self.x[i,l+1]-self.x[i,l])) * (feature_value - self.x[i,l])
                            break
        # Do not forget that this method is called in predict_preference (line 42) and therefor should return well-organized data for it to work.
        return utility


class HeuristicModel(BaseModel):
    """Skeleton of MIP you have to write as the first exercise.
    You have to encapsulate your code within this class that will be called for evaluation.
    """

    def __init__(self, n_clusters=2, n_pieces=5, clustering_method = "difference_KMeans_L2", n_iterations = 10):
        """Initialization of the Heuristic Model.
        """
        self.seed = 123
        self.n_pieces = n_pieces
        self.n_clusters = n_clusters
        self.clustering_method = clustering_method
        self.n_iterations = n_iterations

    def fit(self, X, Y):
        """Estimation of the parameters - To be completed.

        Parameters
        ----------
        X: np.ndarray
            (n_samples, n_features) features of elements preferred to Y elements
        Y: np.ndarray
            (n_samples, n_features) features of unchosen elements
        """
        # We define some general useful variables
        n_samples = X.shape[0]
        n_features = X.shape[1]
        n_pieces = self.n_pieces
        n_clusters = self.n_clusters


        # Cut the abscissa axis into n_pieces intervals (will be useful for the predict_utility method)
        self.x = np.zeros((n_features, n_pieces+1), dtype = float)
        for i in range(n_features):
            for l in range(n_pieces+1):
                self.x[i, l] = l / n_pieces

        # We start by an apriori clustering of the samples. We can choose between different methods with the input clustering_method

        # 1. We compute the difference between the features of X and Y as coordinates, and then use KMeans clustering with L2 distance as a metric
        if self.clustering_method == "difference_KMeans_L2":
            kmeans = KMeans(n_clusters=n_clusters, random_state=self.seed, n_init='auto').fit(X - Y)
            clusters = kmeans.labels_

        # 2. In order not to have (x,y) and (y,x) in the same cluster, we can also use the cosine distance as a metric, thus two symetric points will not be in the same cluster. We use AHC to compute the clustering with a custom distance measure
        if self.clustering_method == "difference_AHC_cosine":
            AHC = AgglomerativeClustering(n_clusters=n_clusters, metric="cosine", linkage="average").fit(X - Y)
            clusters = AHC.labels_

        # 3. We can test concatenation of the features of X and Y instead of doing the difference. Here we then apply KMeans clustering with L2 distance as a metric
        if self.clustering_method == "concatenation_KMeans_L2":
            kmeans = KMeans(n_clusters=n_clusters, random_state=self.seed, n_init='auto').fit(np.concatenate((X,Y), axis=1))
            clusters = kmeans.labels_
            
        # 4. We can also try the concatenation with the cosine distance as a metric even if it less relevant in this case
        if self.clustering_method == "concatenation_AHC_cosine":
            AHC = AgglomerativeClustering(n_clusters=n_clusters, metric="cosine", linkage="average").fit(np.concatenate((X,Y), axis=1))
            clusters = AHC.labels_

        # We create a function that compute the utility function of each cluster with the MIP model
        def compute_utility(clusters):
            # We create as many new samples list as clusters, and we store the samples that belong to each cluster
            samples_per_cluster = [[] for _ in range(n_clusters)]
            for j in range(n_samples):
                samples_per_cluster[clusters[j]].append((X[j], Y[j]))
            for k in range(n_clusters):
                samples_per_cluster[k] = np.array(samples_per_cluster[k])
            # We can apply the MIP optimization on each cluster, with n_cluster = 1, and store the utility functions for each cluster's model, we then store the utility functions 
            u_opt = np.zeros((n_features, n_pieces+1, n_clusters))
            for k in range(n_clusters):
                model = TwoClustersMIP(n_pieces=n_pieces, n_clusters=1)
                model.fit(samples_per_cluster[k][:,0], samples_per_cluster[k][:,1])
                u = model.u  # we retrieve the utility functions from the MIP model
                for i in range(n_features):
                    for l in range(n_pieces+1):
                        u_opt[i, l, k] = u[i, l, 0]
            return u_opt

        # For the initial iteration, we use the a posteriori clustering to compute the utility functions
        self.u = compute_utility(clusters)

        # We can refine the clusters by changing the cluster of samples that are not well explained by the utility functions, and recompute the utility functions
        if self.n_iterations > 1:
            for iteration in range(self.n_iterations-1):
                # We use the predict_cluster method to get the cluster that is the most likely to prefer the sample
                cluster_predicted = self.predict_cluster(X, Y)
                for j in range(n_samples):
                    if cluster_predicted[j] != clusters[j]:
                        clusters[j] = cluster_predicted[j]
                # Then we recompute the utility functions with the new clusters
                self.u = compute_utility(clusters)                 
                self.u_hist.append(self.u)       



    def predict_utility(self, X):
        """Return Decision Function of the MIP for X. - To be completed.

        Parameters:
        -----------
        X: np.ndarray
            (n_samples, n_features) list of features of elements
        """
        # Use self.u obtained by the fit method to predict the utility of X, we also use the self.x[i, l] for the abscissa axis
        n_samples = X.shape[0]
        n_features = X.shape[1]
        utility = np.zeros((n_samples, self.n_clusters), dtype = float)
        for j in range(n_samples):
            for k in range(self.n_clusters):
                utility[j, k] = 0
                for i, feature_value in enumerate(X[j]):
                    for l in range(self.n_pieces):
                        if self.x[i, l] <= feature_value <= self.x[i, l+1]:
                            utility[j, k] += self.u[i, l, k] + ((self.u[i, l+1, k]-self.u[i, l, k])/(self.x[i,l+1]-self.x[i,l])) * (feature_value - self.x[i,l])
                            break

        return utility

