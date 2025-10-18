from __future__ import print_function

import os
from builtins import range
from builtins import object
import numpy as np
from cs231n.classifiers.softmax import *  # 提供 svm_loss_vectorized / softmax_loss_vectorized
from past.builtins import xrange


class LinearClassifier(object):
    def __init__(self):
        self.W = None

    def train(
        self,
        X,
        y,
        learning_rate=1e-3,
        reg=1e-5,
        num_iters=100,
        batch_size=200,
        verbose=False,
    ):
        """
        Train this linear classifier using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) containing training data.
        - y: A numpy array of shape (N,) containing training labels.
        - learning_rate: learning rate for SGD.
        - reg: L2 regularization strength.
        - num_iters: number of SGD steps.
        - batch_size: minibatch size.
        - verbose: print progress every 100 iters if True.

        Outputs:
        - loss_history: list of loss value at each iteration.
        """
        num_train, dim = X.shape
        num_classes = np.max(y) + 1  # assume labels are 0..K-1
        if self.W is None:
            # lazily initialize W
            self.W = 0.001 * np.random.randn(dim, num_classes)

        # Run stochastic gradient descent to optimize W
        loss_history = []
        for it in range(num_iters):
            # -----------------------------
            # 采样一个 minibatch
            # -----------------------------
            idx = np.random.choice(num_train, batch_size, replace=True)
            X_batch = X[idx]                 # (batch_size, D)
            y_batch = y[idx]                 # (batch_size,)

            # evaluate loss and gradient
            loss, grad = self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)

            # -----------------------------
            # 使用梯度更新参数 (SGD step)
            # -----------------------------
            self.W -= learning_rate * grad

            if verbose and it % 100 == 0:
                print("iteration %d / %d: loss %f" % (it, num_iters, loss))

        return loss_history

    def predict(self, X):
        """
        Predict labels for data points using the trained weights.

        Inputs:
        - X: (N, D)

        Returns:
        - y_pred: (N,) predicted class indices
        """
        # 计算分类分数并取每行最大值的索引
        scores = X.dot(self.W)              # (N, C)
        y_pred = np.argmax(scores, axis=1)  # (N,)
        return y_pred

    def loss(self, X_batch, y_batch, reg):
        """
        Compute the loss function and its derivative.
        Subclasses will override this.
        """
        raise NotImplementedError

    def save(self, fname):
        """Save model parameters."""
        fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
        params = {"W": self.W}
        np.save(fpath, params)
        print(fname, "saved.")
    
    def load(self, fname):
        """Load model parameters."""
        fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
        if not os.path.exists(fpath):
            print(fname, "not available.")
            return False
        else:
            params = np.load(fpath, allow_pickle=True).item()
            self.W = params["W"]
            print(fname, "loaded.")
            return True


class LinearSVM(LinearClassifier):
    """A subclass that uses the Multiclass SVM loss function"""

    def loss(self, X_batch, y_batch, reg):
        return svm_loss_vectorized(self.W, X_batch, y_batch, reg)


class Softmax(LinearClassifier):
    """A subclass that uses the Softmax + Cross-entropy loss function"""

    def loss(self, X_batch, y_batch, reg):
        return softmax_loss_vectorized(self.W, X_batch, y_batch, reg)

