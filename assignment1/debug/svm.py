from pathlib import Path

import numpy as np

from cs231n.classifiers import svm_loss_naive, svm_loss_vectorized
from debug.utils import get_CIFAR10_data

X_train, y_train, X_val, y_val, X_test, y_test, X_dev, y_dev = get_CIFAR10_data(num_training=4900, num_validation=100, num_test=100, num_dev=500)


# generate a random SVM weight matrix of small numbers
W = np.random.randn(3073, 10) * 0.0001

# Compute the loss and its gradient at W.
loss, grad = svm_loss_naive(W, X_dev, y_dev, 0.0)

# Numerically compute the gradient along several randomly chosen dimensions, and
# compare them with your analytically computed gradient. The numbers should match
# almost exactly along all dimensions.
from cs231n.gradient_check import grad_check_sparse
f = lambda w: svm_loss_naive(w, X_dev, y_dev, 0.0)[0]
grad_numerical = grad_check_sparse(f, W, grad)

# do the gradient check once again with regularization turned on
# you didn't forget the regularization gradient did you?
loss, grad = svm_loss_naive(W, X_dev, y_dev, 5e1)
f = lambda w: svm_loss_naive(w, X_dev, y_dev, 5e1)[0]
grad_numerical = grad_check_sparse(f, W, grad)


loss_naive, grad_naive = svm_loss_naive(W, X_dev, y_dev, 0.0005)
loss_vectorized, grad_vectorized = svm_loss_vectorized(W, X_dev, y_dev, 0.005)
print('loss difference: %f' % (loss_naive - loss_vectorized))
difference = np.linalg.norm(grad_naive - grad_vectorized, ord='fro')
print('gradient difference: %f' % difference)
