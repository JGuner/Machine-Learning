
'''
File: kmeans.py
Project: Downloads
File Created: Feb 2021
Author: Rohit Das
'''

import numpy as np

class KMeans(object):

    def __init__(self, points, k, init='random', max_iters=10000, rel_tol=1e-5):  # No need to implement
        """
        Args:
            points: NxD numpy array, where N is # points and D is the dimensionality
            K: number of clusters
            init : how to initial the centers
            max_iters: maximum number of iterations (Hint: You could change it when debugging)
            rel_tol: convergence criteria w.r.t relative change of loss
        Return:
            none
        """
        self.points = points
        self.K = k
        if init == 'random':
            self.centers = self.init_centers()
        else:
            self.centers = self.kmpp_init()
        self.assignments = None
        self.loss = 0.0
        self.rel_tol = rel_tol
        self.max_iters = max_iters

    def init_centers(self):# [2 pts]
        """
            Initialize the centers randomly
        Return:
            self.centers : K x D numpy array, the centers.
        Hint: Please initialize centers by randomly sampling points from the dataset in case the autograder fails.
        """

        center = np.random.choice((self.points).shape[0], self.K, replace=False)
        self.centers = self.points[center]

        return self.centers

    
    def kmpp_init(self):# [3 pts]
        """
            Use the intuition that points further away from each other will probably be better initial centers
        Return:
            self.centers : K x D numpy array, the centers.
        """

        raise NotImplementedError

    def update_assignment(self):  # [5 pts]
        """
            Update the membership of each point based on the closest center
        Return:
            self.assignments : numpy array of length N, the cluster assignment for each point
        Hint: You could call pairwise_dist() function
        """        

        self.assignments = np.argmin(pairwise_dist(self.points, self.centers), axis = 1)

        return self.assignments

    def update_centers(self):  # [5 pts]
        """
            update the cluster centers
        Return:
            self.centers: new centers, a new K x D numpy array of float dtype, where K is the number of clusters, and D is the dimension.

        HINT: Points may be integer, but the centers should not have to be. Watch out for dtype casting!
        """

        for i in range(len(self.centers)):
            self.centers[i] = np.mean(self.points[i == self.assignments], axis = 0)
        return self.centers

    def get_loss(self):  # [5 pts]
        """
            The loss will be defined as the sum of the squared distances between each point and it's respective center.
        Return:
            self.loss: a single float number, which is the objective function of KMeans.
        """

        self.loss = 0

        for i in range(len(self.centers)):
            self.loss += np.sum(np.square(self.points[i == self.assignments] - self.centers[i]))
        
        return self.loss
    def train(self):    # [10 pts]
        """
            Train KMeans to cluster the data:
                0. Recall that centers have already been initialized in __init__
                1. Update the cluster assignment for each point
                2. Update the cluster centers based on the new assignments from Step 1
                3. Check to make sure there is no mean without a cluster, 
                   i.e. no cluster center without any points assigned to it.
                   - In the event of a cluster with no points assigned, 
                     pick a random point in the dataset to be the new center and 
                     update your cluster assignment accordingly.
                4. Calculate the loss and check if the model has converged to break the loop early.
                   - The convergence criteria is measured by whether the percentage difference 
                     in loss compared to the previous iteration is less than the given 
                     relative tolerance threshold (self.rel_tol). 
                5. Iterate through steps 1 to 4 max_iters times. Avoid infinite looping!
                
        Return:
            self.centers: K x D numpy array, the centers
            self.assignments: Nx1 int numpy array
            self.loss: final loss value of the objective function of KMeans.
        """

        for it1 in range(self.max_iters):
            self.update_assignment()
            self.update_centers()
            for it2 in range(self.max_iters):
                i = -1
                for j in range(self.K):
                    K_assignments = self.points[j == self.assignments]
                    if (len(K_assignments) == 0):
                        i = j

                        break

                if (i != -1):
                    selectPoint = np.random.choice(self.points.shape[0], size=1, replace=False)
                    chosenPoint = self.points[selectPoint]
                    self.centers[i] = chosenPoint
                    self.update_assignment()
                    self.update_centers()
                else:
                    break

            prev_loss = self.get_loss()
            self.update_assignment()
            self.update_centers()

            if (prev_loss != 0) and (self.rel_tol > ((prev_loss - self.get_loss()) / prev_loss)):
                return self.centers, self.assignments, self.loss

        return self.centers, self.assignments, self.loss

def pairwise_dist(x, y):  # [5 pts]
        np.random.seed(1)
        """
        Args:
            x: N x D numpy array
            y: M x D numpy array
        Return:
                dist: N x M array, where dist2[i, j] is the euclidean distance between
                x[i, :] and y[j, :]
        """

        x_norm = np.sum(x**2, axis=1)
        y_norm = np.sum(y**2, axis=1)
        xy = np.dot(x, y.T)
        dist = np.sqrt(x_norm[:, np.newaxis] + y_norm - 2 * xy)
        return dist