from builtins import range
import numpy as np
from random import shuffle


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

    loss, dW = softmax_loss_vectorized(W, X, y, reg)


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

    # normalize data by dividing train std first
    N, _ = X.shape
    logits = X @ W  # shape=(N, C)
    #class_max = np.max(logits, axis=1, keepdims=True)  # shape=(N, 1)
    #logits_exp = np.exp(logits - class_max)
    logits_exp = np.exp(logits)  # shape=(N, C)
    sumexp = np.sum(logits_exp, axis=1)  # shape=(N,)
    logsumexp = np.log(sumexp)  # shape=(N,)
    loss = -np.sum(logits[np.arange(N), y] - logsumexp)  # negative prob of the correct class
    loss /= N
    loss += reg * np.sum(W * W)

    # chain rule
    dlogits = logits_exp / sumexp.reshape(N, 1)  # shape=(N, C)
    dlogits[np.arange(N), y] -= 1
    dW = X.T @ dlogits  # shape=(D, C)
    dW /= N
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
