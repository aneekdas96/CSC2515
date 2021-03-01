# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 20:39:09 2017
"""
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_boston
from scipy.special import logsumexp

np.random.seed(0)

# load boston housing prices dataset
boston = load_boston()
x = boston['data']
N = x.shape[0]
x = np.concatenate((np.ones((506,1)),x),axis=1) #add constant one feature - no bias needed
d = x.shape[1]
y = boston['target']

idx = np.random.permutation(range(N))

# print(type(x)) # <class 'numpy.ndarray'>
# print(x.shape) # (506, 14)

#helper function
def l2(A,B):
    '''
    Input: A is a Nxd matrix
           B is a Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between A[i,:] and B[j,:]
    i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
    '''
    A_norm = (A**2).sum(axis=1).reshape(A.shape[0],1)
    B_norm = (B**2).sum(axis=1).reshape(1,B.shape[0])
    dist = A_norm+B_norm-2*A.dot(B.transpose())
    return dist


#to implement
def LRLS(test_datum,x_train,y_train, tau,lam=1e-5):
    '''
    Input: test_datum is a dx1 test vector
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           tau is the local reweighting parameter
           lam is the regularization parameter
    output is y_hat the prediction on test_datum
    '''
	
    test_datum = np.reshape(test_datum, (14, 1)) # reshape as  l2() requires matrix not vector 
    
    ######## CALCULATE 'A' #######
    l2_distances = -l2(np.transpose(test_datum), x_train)/(2*(tau**2))  # transpose test_datum to get (1, 14) as l2() requires Nxd format
    # l2 distances if of the shape (1, n), where the n values represent distances of the test_datum to each of our training points
    num_term = np.exp(l2_distances)
    den_term = np.exp(logsumexp(l2_distances)) # logsumexp computes the log of the sum of exponentials of input elements
    A = np.diagflat(num_term/den_term) # A is a diagonal matrix, where each diagonal element contains the weight for the i-th data point

    ######## CALCULATE 'w' #######
    term1 = np.dot(np.dot(np.transpose(x_train), A), x_train) # calculate XT*A*X
    term2 = lam*np.eye(x_train.shape[1]) # calculate lam*I
    term3 = np.dot(np.dot(np.transpose(x_train), A), y_train) # XT*A*y
    w_optimal = np.linalg.solve(term1 + term2, term3) 
    # get the predicted value of y = XT.W*
    y_hat = np.dot(np.transpose(test_datum), w_optimal)

    return y_hat


def run_validation(x,y,taus,val_frac):
    '''
    Input: x is the N x d design matrix
           y is the N x 1 targets vector    
           taus is a vector of tau values to evaluate
           val_frac is the fraction of examples to use as validation data
    output is
           a vector of training losses, one for each tau value
           a vector of validation losses, one for each tau value
    '''
    # arrays to store losses
    training_loss = np.array([])
    validation_loss = np.array([])

    # use idx to premutate the samples and labels
    randomized_samples, randomized_labels = x[idx], y[idx]

    # split the data into training and validation set
    fraction_train = 1 - val_frac
    train_lim = int(fraction_train * len(x))
    x_train, x_validation = randomized_samples[:train_lim], randomized_samples[train_lim:]
    y_train, y_validation = randomized_labels[:train_lim], randomized_labels[train_lim:]

    train_len = len(x_train)
    validation_len = len(x_validation)

    # loop to run through all values of tau
    for tau in taus:
        train_epoch_loss = 0.0 # record the training loss for particular value of tau over all samples in training set
        validation_epoch_loss = 0.0 # record the validation loss for particular value of tau over all samples in validation set
        
        # loop through all samples in training set to get loss for tau
        for sample_idx, sample in enumerate(x_train):
            lrls_val = LRLS(sample, x_train, y_train, tau)
            train_step_loss = (lrls_val - y_train[sample_idx])**2/(2*train_len)
            train_epoch_loss = train_epoch_loss + train_step_loss
        
        # loop through all samples in validation set to get loss for tau 
        for sample_idx, sample in enumerate(x_validation):
            lrls_val = LRLS(sample, x_train, y_train, tau)
            validation_step_loss = (lrls_val - y_validation[sample_idx])**2/(2*validation_len)
            validation_epoch_loss = validation_epoch_loss + validation_step_loss

        print('------------------------------------------------------')
        print('Tau : ', tau,  ' | Training loss : ', train_epoch_loss, ' | Validation loss : ', validation_epoch_loss)

        # append both losses 
        training_loss = np.append(training_loss, train_epoch_loss)
        validation_loss = np.append(validation_loss, validation_epoch_loss)

    return training_loss, validation_loss



if __name__ == "__main__":
    # In this excersice we fixed lambda (hard coded to 1e-5) and only set tau value. Feel free to play with lambda as well if you wish
    taus = np.logspace(1.0,3,200) # returns 200 values from 1.0 to 3.0 in log scale
    train_losses, test_losses = run_validation(x,y,taus,val_frac=0.3)

    # make a plot with log scaling on the x axis.
    plt.semilogx(train_losses, label="train")
    plt.semilogx(test_losses, label="test")
    plt.xlabel('tau')
    plt.ylabel('avg_loss')
    plt.legend()
    plt.show()
