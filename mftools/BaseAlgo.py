class BaseAlgo:

    def __init__(self, latent_features: int, iter: int, compute_mse=False, random_state=None):
        '''
        Parameters
        ----------
        latent_features - (int) the number of latent features used

        iters - (int) the number of iterations

        compute_mse - (boolean) if True, compute and print the MSE

        random_state - (int) the random state
        '''

        self.latent_features = latent_features
        self.iter = iter
        self.compute_mse = compute_mse
        self.random_state = random_state
    
    def get_rating(self, user_id: int, movie_id: int):
        '''
        Parameters
        ----------
        user_id - (int) a user ID

        movie_id - (int) a movie ID

        Returns
        -------
        pred - (float) the predicted rating for the user-movie pair
        '''

        assert self.pred is not None, "Model not trained yet"
        assert 0 < user_id <= self.pred.shape[0], "User ID out of range"
        assert 0 < movie_id <= self.pred.shape[1], "Movie ID out of range"
        return self.pred[user_id-1, movie_id-1]