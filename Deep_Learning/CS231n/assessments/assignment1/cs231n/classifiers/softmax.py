import numpy as np
#from random import shuffle

def softmax_loss_naive(W, X, y, reg):
    loss = 0.0
    dW = np.zeros_like(W)
    N, D = X.shape
    C = W.shape[1]

    for i in range(N):
        scores = X[i].dot(W)                 # (C,)
        scores -= np.max(scores)             # 数值稳定
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores)  # (C,)

        # loss 累加
        loss += -np.log(probs[y[i]])

        # 梯度：对每个类别 j
        for j in range(C):
            dW[:, j] += (probs[j] - (j == y[i])) * X[i]

    # 平均 + 正则化
    loss = loss / N + reg * np.sum(W * W)
    dW = dW / N + 2.0 * reg * W
    return loss, dW



def softmax_loss_vectorized(W, X, y, reg):
    N = X.shape[0]
    scores = X.dot(W)                                 # (N, C)
    scores -= np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    loss = np.mean(-np.log(probs[np.arange(N), y])) + reg * np.sum(W * W)

    dp = probs.copy()
    dp[np.arange(N), y] -= 1                          # (N, C)
    dW = X.T.dot(dp) / N + 2.0 * reg * W
    return loss, dW
