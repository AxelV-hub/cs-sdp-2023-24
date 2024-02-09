import pickle
from abc import abstractmethod

import gurobipy as gp
from gurobipy import GRB

import numpy as np

from clustering import apriori_clustering


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
        # create a model
        model = gp.Model("MIP")
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

        # define the variables of the MIP
        u = {}
        for k in range(self.n_clusters):
            for i in range(self.n_features):
                for l in range(self.n_pieces):
                    u[k, i, l] = self.model.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name=f"u_{k}{i}{l}")

        z = {}
        for k in range(self.n_clusters):
            for j in range(n_samples):
                z[j, k] = self.model.addVar(vtype=GRB.BINARY, name=f"z_{j}_{k}")

        sigma_plus = self.model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="sigma_plus") 
        sigma_minus = self.model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="sigma_minus")

        # Constraints fonction de décision croissantes 
        for i in range(self.n_features):
            for k in range(self.n_clusters):
                for l in range(self.n_pieces - 1):
                    self.model.addConstr(u[k, i, l] <= u[k, i, l + 1])
        
        #Contrainte d'égalite <-> préférence

        for j in range(n_samples):
            self.model.addConstr(np.sum([z[j, k] for k in range(self.n_clusters)])==1)


        # define some preliminary useful coefficients 
        # define the min_i and max_i for each feature i, computed with X and Y, the value will be later used to calculate the score function
        min_i = np.zeros(n_features)
        max_i = np.zeros(n_features)
        for i in range(n_features):
            min_i[i] = min(min(X[:, i]), min(Y[:, i]))
            max_i[i] = max(max(X[:, i]), max(Y[:, i]))

        # define the intervals boundaries x_i_l for each feature i, and l from 0 to n_pieces, computed with min_i and max_i
        x_i_l = np.zeros((n_features, n_pieces+1))
        for i in range(n_features):
            for l in range(n_pieces+1):
                x_i_l[i, l] = min_i[i] + l * (max_i[i] - min_i[i]) / n_pieces

        # define the score function, based on the alpha_i_l and max_i and min_i
        def score_function(x_i, alpha_i_l):
            # define the interval l_i of x_i_l in which x_i is located
            l_i = np.zeros(n_features)
            for i in range(n_features):
                l_i[i] = int(n_pieces * (x_i - min_i[i]) / (max_i[i] - min_i[i]))  # value of l_i between 0 and n_pieces

            # define the score function for each feature i 
            score = 0
            for i in range(n_features):
                score += alpha_i_l[i, l_i[i]] + (x_i - x_i_l[i, l_i]) / (x_i_l[i, l_i+1] - x_i_l[i, l_i]) * (alpha_i_l[i, l_i[i]+1] - alpha_i_l[i, l_i[i]])

            return score

        # define the constraints of the MIP
        # constraint 1: the sum of the alpha_i_l over i with l fixed at n_pieces is equal to 1
        self.model.addConstr(gp.quicksum(alpha_i_l[t, n_pieces] for t in range(n_samples)) == 1)

        # constraint 2: alpha_i_l is between 0 and 1 for all i and l
        # self.model.addConstrs(alpha_i_l[i, l] >= 0 for i in range(n_features) for l in range(n_pieces))

        # constraint 3: alpha_i_l[i, l] <= alpha_i_l[i, l+1] for all i and l
        self.model.addConstrs(alpha_i_l[i, l] <= alpha_i_l[i, l+1] for i in range(n_features) for l in range(n_pieces))

        # constraint 4: for each pair of X[j],Y[j], score_function(X[j]) + sigma_j[0, j] >= score_function(Y[j]) + sigma_j[1, j]
        self.model.addConstrs(score_function(X[j,:], alpha_i_l) + sigma_j[0, j] >= score_function(Y[j,:], alpha_i_l) + sigma_j[1, j] for j in range(n_samples))

        # define the objective function of the MIP
        # objective function: minimize the sum of abs(sigma_j)
        self.model.setObjective(gp.quicksum(abs(sigma_j[k,j]) for k in [0,1] for j in range(n_samples)), GRB.MINIMIZE)

        # optimize the model
        self.model.optimize()

        # get the optimal values of alpha_i_l 
        if self.model.status == GRB.OPTIMAL:
            print('Optimal found')
            alpha_i_l_opt = np.zeros((n_features, n_pieces))
            sigma_j_opt = np.zeros((2, n_samples))
            for i in range(n_features):
                for l in range(n_pieces):
                    alpha_i_l_opt[i, l] = x[i, l].x
        else:
            print('No solution')

        return alpha_i_l_opt


    def predict_utility(self, X):
        """Return Decision Function of the MIP for X. - To be completed.

        Parameters:
        -----------
        X: np.ndarray
            (n_samples, n_features) list of features of elements
        """
        # To be completed
        # Do not forget that this method is called in predict_preference (line 42) and therefor should return well-organized data for it to work.
        return



import os
import sys

# sys.path.append("../python")

import matplotlib.pyplot as plt
import numpy as np
from data import Dataloader
import metrics


# data_loader = Dataloader("../data\dataset_4") # Specify path to the dataset you want to load
# X, Y = data_loader.load()

# # model = RandomExampleModel() # Instantiation of the model with hyperparameters, if needed
# model = TwoClustersMIP(n_pieces=5, n_clusters=1)
# model.fit(X, Y) 




class HeuristicModel(BaseModel):
    """Skeleton of MIP you have to write as the first exercise.
    You have to encapsulate your code within this class that will be called for evaluation.
    """

    def __init__(self, n_pieces, n_clusters):
        """Initialization of the Heuristic Model.
        """
        self.seed = 123
        self.models = self.instantiate()
        self.n_clusters = n_clusters

    def instantiate(self):
        """Instantiation of the MIP Variables"""
        model = gp.Model("MIP")
        return

    def instantiate_clusters(self, X, Y):
        '''
        Initialization of the clusters based on (x,y) pairs coordinates
        '''
        self.clusters = dict()
        labels = apriori_clustering(X, Y, self.n_clusters)
        for k in range(self.n_clusters):
            self.clusters[k] = dict()
            self.clusters[k]["X"] = X[labels == k]
            self.clusters[k]["Y"] = Y[labels == k]

    def fit(self, X, Y):
        """Estimation of the parameters - To be completed.

        Parameters
        ----------
        X: np.ndarray
            (n_samples, n_features) features of elements preferred to Y elements
        Y: np.ndarray
            (n_samples, n_features) features of unchosen elements
        """

        # Strategy: 
        # 1. KMeans to initialize the clusters, 
        # 2. Tune utilities for each cluster and evaluate
        # 3. Reorganize clusters based on distances, 
        # 4. Repeat until convergence

        n_samples = X.shape[0]
        n_features = X.shape[1]
        n_pieces = self.n_pieces

        # define the variables of the MIP
        u = {}
        for k in range(self.n_clusters):
            for i in range(self.n_features):
                for l in range(self.n_pieces):
                    u[k, i, l] = self.model.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name=f"u_{k}{i}{l}")

        z = {}
        for k in range(self.n_clusters):
            for j in range(n_samples):
                z[j, k] = self.model.addVar(vtype=GRB.BINARY, name=f"z_{j}_{k}")

        sigma_plus = self.model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="sigma_plus") 
        sigma_minus = self.model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="sigma_minus")

        # Constraints fonction de décision croissantes 
        for i in range(self.n_features):
            for k in range(self.n_clusters):
                for l in range(self.n_pieces - 1):
                    self.model.addConstr(u[k, i, l] <= u[k, i, l + 1])
        
        #Contrainte d'égalite <-> préférence

        for j in range(n_samples):
            self.model.addConstr(np.sum([z[j, k] for k in range(self.n_clusters)])==1)


        # define some preliminary useful coefficients 
        # define the min_i and max_i for each feature i, computed with X and Y, the value will be later used to calculate the score function
        min_i = np.zeros(n_features)
        max_i = np.zeros(n_features)
        for i in range(n_features):
            min_i[i] = min(min(X[:, i]), min(Y[:, i]))
            max_i[i] = max(max(X[:, i]), max(Y[:, i]))

        # define the intervals boundaries x_i_l for each feature i, and l from 0 to n_pieces, computed with min_i and max_i
        x_i_l = np.zeros((n_features, n_pieces+1))
        for i in range(n_features):
            for l in range(n_pieces+1):
                x_i_l[i, l] = min_i[i] + l * (max_i[i] - min_i[i]) / n_pieces
        return

    def predict_utility(self, X):
        """Return Decision Function of the MIP for X. - To be completed.

        Parameters:
        -----------
        X: np.ndarray
            (n_samples, n_features) list of features of elements
        """
        # To be completed
        # Do not forget that this method is called in predict_preference (line 42) and therefor should return well-organized data for it to work.
        return
