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

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    scores = X @ W
    scores = np.exp(scores)
    scores_for_dW = np.copy(scores)
    
    ans_score = scores[np.arange(scores.shape[0]),y]
    scores = np.sum(scores, axis = 1)
    
    L= ans_score / scores
    L = np.log(L)
    L *= -1
    loss += np.sum(L)       #3
    loss /= X.shape[0]      #2
    loss +=  reg * np.sum(W * W) # 1
    
    sum_SdW=np.sum(scores_for_dW, axis = 1)
    scores_for_dW = scores_for_dW.T
    dW = scores_for_dW / sum_SdW
    print(dW.shape)
    
    dW = dW @ X
    
    tmp = np.zeros((W.shape[1],X.shape[0]))
    tmp [y, np.arange(tmp.shape[1])] = 1 
    
    dW -= tmp @ X
    
    dW /= X.shape[0]
    #scores.
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return loss, dW



def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    scores = X @ W
    scores = np.exp(scores)
    scores_for_dW = np.copy(scores)
    
    ans_score = scores[np.arange(scores.shape[0]),y]
    scores = np.sum(scores, axis = 1)
    
    L= ans_score / scores
    L = np.log(L)
    L *= -1
    loss += np.sum(L)       #3
    loss /= X.shape[0]      #2
    loss +=  reg * np.sum(W * W) # 1
    
    sum_SdW=np.sum(scores_for_dW, axis = 1)
    scores_for_dW = scores_for_dW.T
    dW = scores_for_dW / sum_SdW
    print(dW.shape)
    
    dW = dW @ X
    
    tmp = np.zeros((W.shape[1],X.shape[0]))
    tmp [y, np.arange(tmp.shape[1])] = 1 
    
    dW -= tmp @ X
    
    dW /= X.shape[0]

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
