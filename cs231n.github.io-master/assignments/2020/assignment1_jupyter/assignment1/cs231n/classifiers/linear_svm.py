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
    dW = np.zeros(W.shape) # initialize the gradient as zero D*10

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
                dW[:,j] += X[i,:].T
                dW[:,y[i]] -= X[i,:].T
                    
    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train
    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather than first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dW += 2*reg*W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    return loss, dW



def svm_loss_vectorized0(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    # in y correct category (500,)
    scores = X @ W 
    ans_score =  X * W[:, y].T
    #print(ans_score.shape)
    #print(ans_score.T)
    ans_score = np.sum(ans_score,axis=1)
    print(ans_score.shape)
    print(ans_score)
    #print(scores.shape , ans_score.shape)
    #print((scores.T+ans_score).shape)
    #print((scores.T-ans_score) + 1 )
    L = ((scores.T-ans_score)+1 ) #+ np.ones(scores.T.shape)
    L[L<0]=0
    loss += np.sum(L)
    loss /= X.shape[0]
    loss += reg * np.sum(W * W)
    ## 정답 점수 구했다
    ## 전체 점수 구했다.
    ## 전체 점수에 필터( margin = scores[j] - correct_class_score + 1 ) 걸어줘서 다시 만듧시다.
    
    
    
    
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
def svm_loss_vectorized(W, X, y, reg):
  W=W.T
  X=X.T
  """
  Structured SVM loss function, vectorized implementation.
  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################

  scores = np.dot(W, X) # also known as f(x_i, W)

  correct_scores = np.ones(scores.shape) * scores[y, np.arange(0, scores.shape[1])]
  deltas = np.ones(scores.shape)
  
  print(np.sum(correct_scores.T,axis=1).shape)
  print(np.sum(correct_scores.T,axis=1))
  
  L = scores - np.sum(correct_scores.T,axis=1) + deltas

  L[L < 0] = 0
  L[y, np.arange(0, scores.shape[1])] = 0 # Don't count y_i
  loss = np.sum(L)

  # Average over number of training examples
  num_train = X.shape[1]
  loss /= num_train

  # Add regularization
  loss +=  reg * np.sum(W * W)

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################

  grad = np.zeros(scores.shape)

  L = scores - correct_scores + deltas

  L[L < 0] = 0
  L[L > 0] = 1
  L[y, np.arange(0, scores.shape[1])] = 0 # Don't count y_i
  L[y, np.arange(0, scores.shape[1])] = -1 * np.sum(L, axis=0)
  dW = np.dot(L, X.T)

  # Average over number of training examples
  num_train = X.shape[1]
  dW /= num_train

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW