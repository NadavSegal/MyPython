from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    
    N = X.shape[0]
    NumCalass = W.shape[1]
    for j in range(N):
        scores = X[j,:].dot(W)
        ExpAll = 0.0 
        for jj in range(NumCalass):
            ExpAll += np.exp(scores[jj])
            if jj == y[j]:
                Exp_y = np.exp(scores[jj])
                
        loss += -np.log(Exp_y/ExpAll)
                
        ## dW:
        # dscore / dw:
        d_scores = X[j,:]
        # dli / dscore:
        sigma_j = Exp_y/ExpAll
        d_sigma = np.zeros(W.shape[1])
        for jj in range(NumCalass):
            sigma_jj = np.exp(scores[jj])/ExpAll
            d_sigma[jj] = -sigma_jj*sigma_j
            if jj == y[j]:
                d_sigma[jj] = sigma_jj*(1-sigma_j)        
                
            dW[:,jj] += - 1/(Exp_y/ExpAll) * d_sigma[jj] * d_scores
        
        
    loss /= N
    loss += reg * np.sum(W*W)
    
    dW /= N
    dW += reg * 2 * (W)    

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    
    
    scores = X.dot(W)
    ExpAll = np.sum(np.exp(scores),1)
    Exp_y = np.exp(scores[np.arange(scores.shape[0]),y[:]])    
    loss = -np.sum(np.log(Exp_y/ExpAll))/X.shape[0]
    loss += reg * np.sum(W*W)
    
    ## dW:
    # dscore / dw:
    d_scores = X
    # d sigma / dscore:
    sigma_j = Exp_y/ExpAll
    sigma_j = np.tile(sigma_j, (10,1)) 
    sigma_j = sigma_j.T
    
    ExpAll_array = np.tile(ExpAll, (10,1))
    Exp_y_jj = np.exp(scores)  
    sigma_jj = Exp_y_jj/ExpAll_array.T
    
    #mask:
    mask = np.zeros_like(sigma_j)
    mask[np.arange(mask.shape[0]),y[:]] = 1
    
    d_sigma = sigma_jj*(mask - sigma_j)
    # dL /d sigma
    dL_dSigma = -1/ (Exp_y/ExpAll)
    dL_dSigma = np.eye(dL_dSigma.shape[0])*dL_dSigma
    # dW:  
    dW = np.dot(np.dot(dL_dSigma,d_sigma).T,d_scores)
    dW = dW.T
    
    dW /= X.shape[0]
    dW += reg * 2 * (W)      
    

    return loss, dW
