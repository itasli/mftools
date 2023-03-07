from .BaseAlgo import BaseAlgo
import numpy as np
import pandas as pd
from tqdm import tqdm

class FunkSVD(BaseAlgo):
    
    # initialize the class using BaseAlgo
    def __init__(self, latent_features: int, iter: int, learning_rate = 0.01, batch_size = 0.1, compute_mse = False, method = 'sgd', random_state = None, reg = 5e-3):
        '''
        Parameters
        ----------
        latent_features - (int) the number of latent features used

        learning_rate - (float) the learning rate

        iters - (int) the number of iterations

        batch_size - (int) the number of user-movie pairs to use in each batch

        compute_mse - (boolean) if True, compute and print the MSE

        method - (str) 'sgd' or 'gd' for stochastic gradient descent or gradient descent

        random_state - (int) the random state

        reg - (float) the regularization parameter
        '''
        
        super().__init__(latent_features, iter, compute_mse, random_state)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.method = method
        self.reg = reg


    def fit(self, ratings_mat: pd.DataFrame or np.ndarray):
        '''
        Fit FunkSVD to training data.

        Parameters
        ----------
        ratings_mat - (numpy array) a matrix of user ratings of movies
                    shape = number of users x number of movies

        latent_features - (int) the number of latent features used

        learning_rate - (float) the learning rate

        iters - (int) the number of iterations

        batch_size - (int) the number of user-movie pairs to use in each batch

        Returns
        -------
        user_mat - (numpy array) a latent feature by user matrix

        movie_mat - (numpy array) a latent feature by movie matrix
        '''

        # Check if instance is pandas dataframe and convert to numpy array
        if isinstance(ratings_mat, pd.DataFrame):
            ratings_mat = ratings_mat.values

        rs_u = np.random.RandomState(self.random_state + 1) if self.random_state else np.random.RandomState()
        rs_m = np.random.RandomState(self.random_state - 1) if self.random_state else np.random.RandomState()

        # Set up useful values to be used through the rest of the function
        n_users = ratings_mat.shape[0]
        n_movies = ratings_mat.shape[1]
        num_ratings = np.count_nonzero(~np.isnan(ratings_mat))

        # initialize the user and movie matrices with random values
        user_mat = rs_u.rand(n_users, self.latent_features)
        movie_mat = rs_m.rand(self.latent_features, n_movies)

        # Randomly sample a subset of user-movie pairs
        pairs = np.argwhere(~np.isnan(ratings_mat))
        
        if self.method == 'sgd':

            if 0 < self.batch_size < 1:
                self.batch_size = int(self.batch_size * pairs.shape[0])

            return self._fit_sgd(pairs, ratings_mat, user_mat, movie_mat, num_ratings)
        
        elif self.method == 'gd':
            return self._fit_gd(pairs, ratings_mat, user_mat, movie_mat, num_ratings)

    
    def _fit_sgd(self, pairs, ratings_mat, user_mat, movie_mat, num_ratings):
        
        # for each iteration
        for _ in tqdm(range(self.iter)):

            # Shuffle the pairs
            np.random.shuffle(pairs)
            pairs_sample = pairs[:self.batch_size]

            # For each user-movie pair in the batch
            for i, j in pairs_sample:
                
                # compute the error as the actual minus the dot product of the user and movie latent features
                diff = ratings_mat[i, j] - np.dot(user_mat[i, :], movie_mat[:, j])

                # compute the gradient in one shot
                grad_user = 2 * diff * movie_mat[:, j]
                grad_movie = 2 * diff * user_mat[i, :]

                # update the matrices in the direction of the gradient
                user_mat[i, :] += self.learning_rate * grad_user - self.reg * user_mat[i, :]
                movie_mat[:, j] += self.learning_rate * grad_movie - self.reg * movie_mat[:, j]

        self._user_mat = user_mat
        self._movie_mat = movie_mat
        self.pred = np.clip(np.dot(user_mat, movie_mat), 0, 5)

        if self.compute_mse:
            # compute the mean squared error for the trained matrices
            self.mse = np.nansum( (ratings_mat - self.pred )**2 ) / num_ratings
            # print mean squared error
            print("MSE: ", self.mse)

        return self.pred

    def _fit_gd(self, pairs, ratings_mat, user_mat, movie_mat, num_ratings):

        # for each iteration
        for _ in tqdm(range(self.iter)):

            # For each user-movie pair
            for i, j in pairs:

                # compute the error as the actual minus the dot product of the user and movie latent features
                diff = ratings_mat[i, j] - np.dot(user_mat[i, :], movie_mat[:, j])

                # compute the gradient in one shot
                grad_user = 2 * diff * movie_mat[:, j]
                grad_movie = 2 * diff * user_mat[i, :]

                # update the matrices in the direction of the gradient
                user_mat[i, :] += self.learning_rate * grad_user - self.reg * user_mat[i, :]
                movie_mat[:, j] += self.learning_rate * grad_movie - self.reg * movie_mat[:, j]
                
        
        self._user_mat = user_mat
        self._movie_mat = movie_mat
        self.pred = np.clip(np.dot(user_mat, movie_mat), 0, 5)

        if self.compute_mse:
            # compute the mean squared error for the trained matrices
            self.mse = np.nansum( (ratings_mat - self.pred )**2 ) / num_ratings
            # print mean squared error
            print("MSE: ", self.mse)

        return self.pred