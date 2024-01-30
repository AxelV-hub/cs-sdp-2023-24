import pickle
from abc import abstractmethod

import gurobipy as gp
from gurobipy import GRB

import numpy as np


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

        # We create the variable corresponding to the values of the utilies functions that we want to find
        u = {}
        for i in range(n_features):
            for l in range(n_pieces+1):
                for k in range(self.n_clusters):
                    u[i, l, k] = self.model.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name=f"u_{i}{l}{k}")

        # We create the variable corresponding to the clusters, the value is 1 if it is in the corresponding cluster
        z = {}
        for k in range(self.n_clusters):
            for j in range(len(X)):
                z[j, k] = self.model.addVar(vtype=GRB.BINARY, name=f"z_{j}_{k}")

        # We create the variables of surestimation and underestimation of the score function, that we want to minimize
        sigma_plus_x = {}
        sigma_minus_x = {}
        sigma_plus_y = {}
        sigma_minus_y = {}
        for j in range(n_samples):
            sigma_plus_x[j] = self.model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"sigma_plus_x_{j}")
            sigma_minus_x[j] = self.model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"sigma_minus_x_{j}")
            sigma_plus_y[j] = self.model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"sigma_plus_y_{j}")
            sigma_minus_y[j] = self.model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"sigma_minus_y_{j}")

        # We add the constraint : the utility functions are increasing
        for i in range(n_features):
            for l in range(n_pieces):
                for k in range(self.n_clusters):
                    self.model.addConstr(u[i, l, k] <= u[i, l+1, k])
        
        # Constraint for the clusters, should choose only one cluster for each sample
        for j in range(n_samples):
            self.model.addConstr(gp.quicksum(z[j, k] for k in range(self.n_clusters)) >= 1)

        # Define some preliminary useful coefficients before adding the constraint of preference with the score function
        # 1) define the min_i and max_i for each feature i, computed with X and Y, the value will be later used to calculate the score function
        min_i = np.zeros(n_features)
        max_i = np.zeros(n_features)
        for i in range(n_features):
            min_i[i] = min(min(X[:, i]), min(Y[:, i]))
            max_i[i] = max(max(X[:, i]), max(Y[:, i]))

        # 2) define the intervals boundaries x for each feature i, and l from 0 to n_pieces, computed with min_i and max_i
        x = np.zeros((n_features, n_pieces+1))
        for i in range(n_features):
            for l in range(n_pieces+1):
                x[i, l] = min_i[i] + l * (max_i[i] - min_i[i]) / n_pieces

        # Define the preference constraint, for each sample and for each feature
        # for j in range(n_samples):
        #     l_x = np.zeros(n_features,dtype=int)
        #     l_y = np.zeros(n_features,dtype=int)
        #     for i in range(n_features):
        #         l_x[i] = int(n_pieces * (X[j,i] - min_i[i]) / (max_i[i] - min_i[i]))  # define the part of the utility function it is on, value of l between 0 and n_pieces
        #         l_y[i] = int(n_pieces * (Y[j,i] - min_i[i]) / (max_i[i] - min_i[i]))  
        #     for k in range(self.n_clusters):
        #         self.model.addConstr(z[j, k] *
        #             (gp.quicksum(
        #                 min_i[i] if l_x[i] == 0 else 
        #                 max_i[i] if l_x[i] == n_pieces else 
        #                 u[i, l_x[i], k] + (X[j,i] - x[i, l_x[i]]) / (x[i, l_x[i]+1] - x[i, l_x[i]]) * (u[i, l_x[i]+1, k] - u[i, l_x[i], k]) for i in range(n_features)) +
        #             sigma_plus_y[j] - sigma_minus_y[j]) >= 
        #             z[j, k] *
        #             (gp.quicksum(
        #                 min_i[i] if l_y[i] == 0 else 
        #                 max_i[i] if l_y[i] == n_pieces else 
        #                 u[i, l_y[i], k] + (Y[j,i] - x[i, l_y[i]]) / (x[i, l_y[i]+1] - x[i, l_y[i]]) * (u[i, l_y[i]+1, k] - u[i, l_y[i], k]) for i in range(n_features)) +
        #             sigma_plus_y[j] - sigma_minus_y[j])
        #             )


        l_x = np.zeros((n_features,n_samples), dtype=int)
        l_y = np.zeros((n_features,n_samples), dtype=int)
        for j in range(n_samples):
            for i in range(n_features):
                l_x[i,j] = min(int(n_pieces * (X[j,i] - min_i[i]) / (max_i[i] - min_i[i])), n_pieces-1)
                l_y[i,j] = min(int(n_pieces * (Y[j,i] - min_i[i]) / (max_i[i] - min_i[i])), n_pieces-1)
        
        for j in range(n_samples):
            for k in range(self.n_clusters):
                self.model.addConstr(
                    z[j, k] *
                    (gp.quicksum(u[i, l_x[i,j], k] + (X[j,i] - x[i, l_x[i,j]]) / (x[i, l_x[i,j]+1] - x[i, l_x[i,j]]) * (u[i, l_x[i,j]+1, k] - u[i, l_x[i,j], k]) for i in range(n_features)) +
                    sigma_plus_y[j] - sigma_minus_y[j]) >= 
                    z[j, k] *
                    (gp.quicksum(u[i, l_y[i,j], k] + (Y[j,i] - x[i, l_y[i,j]]) / (x[i, l_y[i,j]+1] - x[i, l_y[i,j]]) * (u[i, l_y[i,j]+1, k] - u[i, l_y[i,j], k]) for i in range(n_features)) +
                    sigma_plus_y[j] - sigma_minus_y[j])
                    )

        # Adding constraint to force z to have the value 1 if X_j will be in the cluster k
        Maj = 10
        for j in range(n_samples):
            for k in range(self.n_clusters):
                self.model.addConstr(
                    gp.quicksum(u[i, l_x[i,j], k] + (X[j,i] - x[i, l_x[i,j]]) / (x[i, l_x[i,j]+1] - x[i, l_x[i,j]]) * (u[i, l_x[i,j]+1, k] - u[i, l_x[i,j], k]) for i in range(n_features)) - 
                    gp.quicksum(u[i, l_y[i,j], k] + (Y[j,i] - x[i, l_y[i,j]]) / (x[i, l_y[i,j]+1] - x[i, l_y[i,j]]) * (u[i, l_y[i,j]+1, k] - u[i, l_y[i,j], k]) for i in range(n_features)) <= 
                    Maj * z[j, k]
                )
                self.model.addConstr(
                    gp.quicksum(u[i, l_x[i,j], k] + (X[j,i] - x[i, l_x[i,j]]) / (x[i, l_x[i,j]+1] - x[i, l_x[i,j]]) * (u[i, l_x[i,j]+1, k] - u[i, l_x[i,j], k]) for i in range(n_features)) -
                    gp.quicksum(u[i, l_y[i,j], k] + (Y[j,i] - x[i, l_y[i,j]]) / (x[i, l_y[i,j]+1] - x[i, l_y[i,j]]) * (u[i, l_y[i,j]+1, k] - u[i, l_y[i,j], k]) for i in range(n_features)) >= 
                    Maj * (1 - z[j, k])
                )

        # Adding last constraint: the sum of the max values of u is equal to 1
        for k in range(self.n_clusters):
            self.model.addConstr(gp.quicksum(u[i, n_pieces, k] for i in range(n_features))== 1)

        # Define the objective function of the MIP: minimize the sum of the underestimation and overestimation of the score function
        self.model.setObjective(gp.quicksum(sigma_minus_x[j] + sigma_plus_x[j] + sigma_minus_y[j] + sigma_plus_y[j] for j in range(n_samples)), GRB.MINIMIZE)

        # optimize the model
        self.model.optimize()

        # get the optimal values of u 
        if self.model.status == GRB.OPTIMAL:
            print('Optimal found')
            u_opt = np.zeros((n_features, n_pieces, self.n_clusters))
            z_opt = np.zeros((n_samples, self.n_clusters))
            for i in range(n_features):
                for l in range(n_pieces):
                    for k in range(self.n_clusters):
                        u_opt[i, l, k] = u[i, l, k].x
            for j in range(n_samples):
                for k in range(self.n_clusters):
                    z_opt[j, k] = z[j, k].x
        else:
            print('No solution')

        return u_opt, z_opt


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

import matplotlib.pyplot as plt


data_loader = Dataloader("data\dataset_4") # Specify path to the dataset you want to load
X, Y = data_loader.load()

# model = RandomExampleModel() # Instantiation of the model with hyperparameters, if needed
model_zerocluster = TwoClustersMIP(n_pieces=5, n_clusters=2)
u_opt, z_opt = model_zerocluster.fit(X, Y) 

# ploting the utility functions for each feature and each cluster, one figure by cluster, with a grid of each utility function
num_features = np.shape(X)[1]
num_clusters = 2

for k in range(num_clusters):
    plt.subplots(num_features, 1, figsize=(10, 10))
    for i in range(num_features):
        plt.subplot(num_features, 1, i+1)
        plt.plot(u_opt[i, :, k])

plt.show()


print(u_opt)
print(z_opt)



class HeuristicModel(BaseModel):
    """Skeleton of MIP you have to write as the first exercise.
    You have to encapsulate your code within this class that will be called for evaluation.
    """

    def __init__(self):
        """Initialization of the Heuristic Model.
        """
        self.seed = 123
        self.models = self.instantiate()

    def instantiate(self):
        """Instantiation of the MIP Variables"""
        # To be completed
        return

    def fit(self, X, Y):
        """Estimation of the parameters - To be completed.

        Parameters
        ----------
        X: np.ndarray
            (n_samples, n_features) features of elements preferred to Y elements
        Y: np.ndarray
            (n_samples, n_features) features of unchosen elements
        """
        # To be completed
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
