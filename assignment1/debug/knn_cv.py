from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from cs231n.classifiers import KNearestNeighbor
from cs231n.data_utils import load_CIFAR10


cifar10_dir = Path(__file__).parent.parent / 'cs231n/datasets/cifar-10-batches-py'
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

# Subsample the data for more efficient code execution in this exercise
num_training = 5000
X_train = X_train[:num_training]
y_train = y_train[:num_training]

num_test = 500
X_test = X_test[:num_test]
y_test = y_test[:num_test]

# Reshape the image data into rows
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))
print(X_train.shape, X_test.shape)

train_mean = X_train.mean(axis=0)
train_std = X_train.std(axis=0)
X_train -= train_mean
X_train /= train_std
X_test -= train_mean
X_test /= train_std
X_train.astype(np.float32)
X_test.astype(np.float32)

num_folds = 5
k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]

################################################################################
# TODO:                                                                        #
# Split up the training data into folds. After splitting, X_train_folds and    #
# y_train_folds should each be lists of length num_folds, where                #
# y_train_folds[i] is the label vector for the points in X_train_folds[i].     #
# Hint: Look up the numpy array_split function.                                #
################################################################################
# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

X_train_folds = np.array_split(X_train, num_folds)
y_train_folds = np.array_split(y_train, num_folds)

# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

# A dictionary holding the accuracies for different values of k that we find
# when running cross-validation. After running cross-validation,
# k_to_accuracies[k] should be a list of length num_folds giving the different
# accuracy values that we found when using that value of k.
k_to_accuracies = {}


################################################################################
# TODO:                                                                        #
# Perform k-fold cross validation to find the best value of k. For each        #
# possible value of k, run the k-nearest-neighbor algorithm num_folds times,   #
# where in each case you use all but one of the folds as training data and the #
# last fold as a validation set. Store the accuracies for all fold and all     #
# values of k in the k_to_accuracies dictionary.                               #
################################################################################
# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

for k in k_choices:
    k_to_accuracies[k] = []
    for i in range(num_folds):
        if i == 0:
            X_i = np.concatenate(X_train_folds[1:])
            y_i = np.concatenate(y_train_folds[1:])
        elif i == num_folds - 1:
            X_i = np.concatenate(X_train_folds[:i])
            y_i = np.concatenate(y_train_folds[:i])
        else:
            X_i = np.concatenate((X_train_folds[:i], X_train_folds[i+1:]))
            y_i = np.concatenate((y_train_folds[:i], y_train_folds[i+1:]))
        classifier = KNearestNeighbor()
        classifier.train(X_i, y_i)
        dists = classifier.compute_distances_one_loop(X_train_folds[i])
        y = classifier.predict_labels(dists, k=k)
        num_correct = np.sum(y == y_train_folds[i])
        k_to_accuracies[k].append(float(num_correct) / num_test)

# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

# Print out the computed accuracies
for k in sorted(k_to_accuracies):
    for accuracy in k_to_accuracies[k]:
        print('k = %d, accuracy = %f' % (k, accuracy))


# plot the raw observations
for k in k_choices:
    accuracies = k_to_accuracies[k]
    plt.scatter([k] * len(accuracies), accuracies)

# plot the trend line with error bars that correspond to standard deviation
accuracies_mean = np.array([np.mean(v) for k,v in sorted(k_to_accuracies.items())])
accuracies_std = np.array([np.std(v) for k,v in sorted(k_to_accuracies.items())])
plt.errorbar(k_choices, accuracies_mean, yerr=accuracies_std)
plt.title('Cross-validation on k')
plt.xlabel('k')
plt.ylabel('Cross-validation accuracy')
plt.show()


# Based on the cross-validation results above, choose the best value for k,
# retrain the classifier using all the training data, and test it on the test
# data. You should be able to get above 28% accuracy on the test data.

