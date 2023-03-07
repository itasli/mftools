from .PMF import PMF
from .FunkSVD import FunkSVD
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split as sk_train_test_split


def train_test_split(df, test_size=0.2, random_state=None):
    """
    Split the data into train and test set
    """

    #check if df is a dataframe
    if isinstance(df, pd.DataFrame):
        df = df.values
         
    # set random seed
    rs = np.random.RandomState(seed=random_state)
    
    # get known ratings index using np argwhere
    known_ratings = np.argwhere(~np.isnan(df))

    # shuffle the known ratings
    rs.shuffle(known_ratings)

    # split the known ratings into train and test
    train, test = sk_train_test_split(known_ratings, test_size=test_size, random_state=random_state)

    # transform cell to nan using index in test
    train_data = df.copy()
    train_data[test[:,0], test[:,1]] = np.nan

    # transform cell to nan using index in train
    test_data = df.copy()
    test_data[train[:,0], train[:,1]] = np.nan

    return train_data, test_data


def tune_PMF(df, latent_features, num_iters, lambd_params, sigma_params, random_state, **kwargs):
    """
    Grid Search Function to select the best model based on RMSE of hold-out data
    """

    #check if test_size is in kwargs
    if 'test_size' in kwargs:
        test_size = kwargs['test_size']
    else:
        test_size = 0.2

    # split the data into train and test
    train_data, test_data = train_test_split(df, test_size, random_state=random_state)

    # initial
    min_error = float('inf')
    best_latent_feature = -1
    best_lambda = 0
    best_sigma = 0
    best_model = None

    for rank in latent_features:
        for lambd in lambd_params:
            for sigma in sigma_params:

                    # train the model
                    model = PMF(latent_features=rank, iter=num_iters, lambd=lambd, sigma=sigma, random_state=random_state)
                    
                    # fit the model
                    pred = model.fit(train_data)
                    
                    # compute mse
                    mse = np.nanmean((test_data - pred )**2)
                    error = np.sqrt(mse)

                    if error < min_error:
                        min_error = error
                        best_latent_feature = rank
                        best_lambda = lambd
                        best_sigma = sigma
                        best_model = model
                        print(f'The best model has {best_latent_feature} latent factors, lambda = {best_lambda}, sigma = {best_sigma} with validation RMSE : {min_error}')

    txt = f'The best model has {best_latent_feature} latent factors, lambda = {best_lambda}, sigma = {best_sigma}'
    return best_model, txt


def tune_FunkSVD(df, latent_features, num_iters, reg_params, lr_params, random_state, sgd=False, batch_params=None, **kwargs):
    """
    Grid Search Function to select the best model based on RMSE of hold-out data
    """

    #check if test_size is in kwargs
    if 'test_size' in kwargs:
        test_size = kwargs['test_size']
    else:
        test_size = 0.2

    # split the known ratings into train and test
    train_data, test_data = train_test_split(df, test_size=test_size, random_state=random_state)

    if sgd:
        # initial
        min_error = float('inf')
        best_latent_feature = -1
        best_regularization = 0
        best_lr = 0
        best_batch = 0
        best_model = None

        for rank in latent_features:
            for reg in reg_params:
                for lr in lr_params:
                    for batch in batch_params:

                        # train the model
                        model = FunkSVD(latent_features=rank, learning_rate=lr, iter=num_iters, method='sgd', random_state=random_state, reg=reg, batch_size=batch)
                        
                        # fit the model
                        pred = model.fit(train_data)
                        
                        # compute mse
                        mse = np.nanmean((test_data - pred )**2)
                        error = np.sqrt(mse)

                        if error < min_error:
                            min_error = error
                            best_latent_feature = rank
                            best_regularization = reg
                            best_lr = lr
                            best_batch = batch
                            best_model = model
                            print(f'The best model has {best_latent_feature} latent factors, regularization = {best_regularization}, lr = {best_lr} and batch size = {best_batch} with validation RMSE : {min_error}')

        txt = f'The best model has {best_latent_feature} latent factors, regularization = {best_regularization}, lr = {best_lr} and batch size = {best_batch}'
        return best_model, txt

    else:
        # initial
        min_error = float('inf')
        best_latent_feature = -1
        best_regularization = 0
        best_lr = 0
        best_model = None

        for rank in latent_features:
            for reg in reg_params:
                for lr in lr_params:

                    # train the model
                    model = FunkSVD(latent_features=rank, learning_rate=lr, iter=num_iters, method='gd', random_state=random_state, reg=reg)
                    
                    # fit the model
                    pred = model.fit(train_data)
                    
                    # compute mse
                    mse = np.nanmean((test_data - pred )**2)
                    error = np.sqrt(mse)

                    if error < min_error:
                        min_error = error
                        best_latent_feature = rank
                        best_regularization = reg
                        best_lr = lr
                        best_model = model
                        print(f'The best model has {best_latent_feature} latent factors and regularization = {best_regularization} and lr = {best_lr} with validation RMSE is {min_error}')


        txt = f'The best model has {best_latent_feature} latent factors and regularization = {best_regularization} and lr = {best_lr}'
        return best_model, txt