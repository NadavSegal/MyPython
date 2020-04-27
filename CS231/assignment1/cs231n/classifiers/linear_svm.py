from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape) # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if margin > 0:
                loss += margin
                dW[:,j] += X[i] ## dw            
                dW[:,y[i]] -= X[i]

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    
    dW /= num_train
    dW += reg * 2 * (W)
    
    
    
    return loss, dW



def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero
    
    scores = X.dot(W)
    y_vec = scores[np.arange(scores.shape[0]),y[:]]
    y_array = np.transpose(np.tile(y_vec,(W.shape[1],1)))
    
    mask = np.ones(y_array.shape) 
    mask[np.arange(scores.shape[0]),y[:]] = 0
    correct_class_score = y_array    
    
    margin = np.multiply(scores - correct_class_score +1,mask)
    mask[margin<0] = 0
    margin[margin<0] =0

    loss = np.sum(margin)/margin.shape[0]
    loss += reg * np.sum(W * W)    
    #################### dW:
    
    mask = np.transpose(mask)
    dW = np.dot(mask,X) 
    #################### dWy:
    
    dWy = np.zeros(dW.shape)
    y_mask = np.tile(np.sum(mask,0),(W.shape[0],1))
    y_mask = np.transpose(y_mask)
    dWy1 = np.multiply(y_mask,X)
    y_mask2 = np.zeros(y_array.shape) 
    y_mask2[np.arange(scores.shape[0]),y[:]] = 1
    y_mask2 = np.transpose(y_mask2)
    dWy = np.dot(y_mask2,dWy1)
    
    dW = np.transpose(dW-dWy)
    dW /= margin.shape[0]   
    dW += reg * 2 * (W)
    

    return loss, dW
