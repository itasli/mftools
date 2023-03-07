from .BaseAlgo import BaseAlgo
import numpy as np
import pandas as pd
from tqdm import tqdm
import numpy.linalg as linalg

class PMF(BaseAlgo):

    def __init__(self, latent_features: int, iter: int, lambd = 2, sigma = 1/10, compute_mse = False, random_state = None):
        '''
        Parameters
        ----------
        latent_features - (int) the number of latent features used

        lambd - (float) the regularization parameter

        simga - (float) the standard deviation of the normal distribution

        iters - (int) the number of iterations

        random_state - (int) the random state
        '''

        super().__init__(latent_features, iter, compute_mse, random_state)
        self.lambd = lambd
        self.sigma = sigma


    def fit(self, ratings_mat: pd.DataFrame or np.ndarray):
        '''
        Fit PMF to training data.

        Parameters
        ----------
        ratings_mat - (numpy array) a matrix of user ratings of movies
                    shape = number of users x number of movies

        Returns
        -------
        user_mat - (numpy array) a latent feature by user matrix

        movie_mat - (numpy array) a latent feature by movie matrix
        '''

        # Check if instance is pandas dataframe and convert to numpy array
        if isinstance(ratings_mat, pd.DataFrame):
            ratings_mat = ratings_mat.values
        
        # Set random state
        rs = np.random.RandomState(self.random_state)

        # Set up useful values to be used through the rest of the function
        n_users, n_movies = ratings_mat.shape
        num_ratings = np.count_nonzero(~np.isnan(ratings_mat))
        I = np.identity(self.latent_features)

        # initialize the user and movie matrices where movie matrice follow N(0, 1/lambd) and user matric zero like
        user_mat = np.zeros((n_users, self.latent_features))
        movie_mat = rs.normal(0, 1/self.lambd, (n_movies, self.latent_features))

        # Omega_ui be the index set of the observed ratings for user i
        omega_ui = np.where(~np.isnan(ratings_mat))
        # Omega_vj be the index set of the observed ratings for movie j
        omega_vj = np.where(~np.isnan(ratings_mat.T))

        
        # for each iteration
        for _ in tqdm(range(self.iter)):

            for i in range(n_users):
                omega_ui_i = omega_ui[1][omega_ui[0] == i]  # indices of movies rated by user i
                if omega_ui_i.size > 0:  # if user i has rated any movies
                    M_ij = movie_mat[omega_ui_i]
                    outer_sum = M_ij.T @ M_ij # np.sum(M_ij[:,:,None] * M_ij[:,None,:], axis=0)
                    inner_sum = M_ij.T @ ratings_mat[i, omega_ui_i] # np.sum(M_ij * ratings_mat[i, omega_ui_i][:,None], axis=0)
                    user_mat[i] = linalg.solve(self.lambd * self.sigma**2 * I + outer_sum, inner_sum)

            # loop over movie_mat:
            for j in range(n_movies):
                omega_vj_j = omega_vj[1][omega_vj[0] == j]  # indices of users who rated movie j
                if omega_vj_j.size > 0:  # if movie j has been rated by any users
                    M_ij = user_mat[omega_vj_j]
                    outer_sum = M_ij.T @ M_ij # np.sum(M_ij[:,:,None] * M_ij[:,None,:], axis=0)
                    inner_sum = M_ij.T @ ratings_mat[omega_vj_j, j] # np.sum(M_ij * ratings_mat[omega_vj_j, j][:,None], axis=0)
                    movie_mat[j] = linalg.solve(self.lambd * self.sigma**2 * I + outer_sum, inner_sum)


        
        self._user_mat = user_mat
        self._movie_mat = movie_mat
        self.pred = np.clip(np.dot(user_mat, movie_mat.T), 0, 5)

        if self.compute_mse:
            # compute the mean squared error for the trained matrices
            self.mse = np.nansum( (ratings_mat - self.pred )**2 ) / num_ratings
            # print mean squared error
            print("MSE: ", self.mse)

        return self.pred