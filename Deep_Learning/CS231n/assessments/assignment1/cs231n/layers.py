from builtins import range
import numpy as np

# import numexpr as ne # ~~DELETE LINE~~


def affine_forward(x, w, b):
    """
    Forward of fully-connected layer.
    x: (N, d1, ..., dk), w: (D, M), b: (M,)
    out: (N, M)
    """
    N = x.shape[0]
    x_row = x.reshape(N, -1)             # (N, D)
    out = x_row.dot(w) + b               # (N, M)
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Backward of fully-connected layer.
    dout: (N, M)
    returns dx:(N,d1..dk), dw:(D,M), db:(M,)
    """
    x, w, b = cache
    N = x.shape[0]
    x_row = x.reshape(N, -1)             # (N, D)

    db = np.sum(dout, axis=0)            # (M,)
    dw = x_row.T.dot(dout)               # (D, M)
    dx_row = dout.dot(w.T)               # (N, D)
    dx = dx_row.reshape(x.shape)         # (N, d1..dk)
    return dx, dw, db


def relu_forward(x):
    out = np.maximum(0, x)
    cache = x
    return out, cache


def relu_backward(dout, cache):
    x = cache
    dx = dout * (x > 0)
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    BatchNorm forward.
    x: (N, D), gamma:(D,), beta:(D,)
    """
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    out, cache = None, None

    if mode == "train":
        mu = np.mean(x, axis=0)                          # (D,)
        var = np.var(x, axis=0)                          # (D,) uncorrected
        std = np.sqrt(var + eps)                         # (D,)
        xhat = (x - mu) / std                            # (N, D)
        out = gamma * xhat + beta                        # (N, D)

        running_mean = momentum * running_mean + (1 - momentum) * mu
        running_var  = momentum * running_var  + (1 - momentum) * var

        cache = (x, xhat, mu, var, std, gamma, beta, eps)
    elif mode == "test":
        std = np.sqrt(running_var + eps)
        xhat = (x - running_mean) / std
        out = gamma * xhat + beta
        cache = (x, xhat, running_mean, running_var, std, gamma, beta, eps)
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var
    return out, cache


def batchnorm_backward(dout, cache):
    """
    BatchNorm backward (standard).
    dout: (N, D)
    """
    x, xhat, mu, var, std, gamma, beta, eps = cache
    N, D = x.shape

    dbeta = np.sum(dout, axis=0)                         # (D,)
    dgamma = np.sum(dout * xhat, axis=0)                 # (D,)

    dxhat = dout * gamma                                 # (N, D)

    dvar = np.sum(dxhat * (x - mu) * (-0.5) * (var + eps) ** (-3/2), axis=0)  # (D,)
    dmu = np.sum(dxhat * (-1 / std), axis=0) + dvar * np.sum(-2 * (x - mu), axis=0) / N
    dx = dxhat / std + dvar * 2 * (x - mu) / N + dmu / N

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    BatchNorm backward (optimized one-liner form).
    """
    x, xhat, mu, var, std, gamma, beta, eps = cache
    N, D = x.shape

    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(dout * xhat, axis=0)

    dxhat = dout * gamma
    inv_std = 1.0 / np.sqrt(var + eps)

    # using simplified formula:
    # dx = (1/N) * inv_std * (N*dxhat - sum(dxhat) - xhat*sum(dxhat*xhat))
    sum_dxhat = np.sum(dxhat, axis=0)
    sum_dxhat_xhat = np.sum(dxhat * xhat, axis=0)
    dx = (inv_std / N) * (N * dxhat - sum_dxhat - xhat * sum_dxhat_xhat)

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """
    LayerNorm forward (normalize per data-point across features).
    x: (N, D), gamma:(D,), beta:(D,)
    """
    eps = ln_param.get("eps", 1e-5)

    mu = np.mean(x, axis=1, keepdims=True)               # (N,1)
    var = np.var(x, axis=1, keepdims=True)               # (N,1)
    std = np.sqrt(var + eps)                             # (N,1)
    xhat = (x - mu) / std                                # (N,D)
    out = gamma * xhat + beta

    cache = (x, xhat, mu, var, std, gamma, beta, eps)
    return out, cache


def layernorm_backward(dout, cache):
    """
    LayerNorm backward.
    """
    x, xhat, mu, var, std, gamma, beta, eps = cache
    N, D = x.shape

    dbeta = np.sum(dout, axis=0)                         # (D,)
    dgamma = np.sum(dout * xhat, axis=0)                 # (D,)

    dxhat = dout * gamma                                 # (N, D)
    inv_std = 1.0 / np.sqrt(var + eps)                   # (N,1)

    # Normalize along features (D) for each sample
    # dx = (1/D) * inv_std * (D*dxhat - sum(dxhat) - xhat*sum(dxhat*xhat))
    sum_dxhat = np.sum(dxhat, axis=1, keepdims=True)                 # (N,1)
    sum_dxhat_xhat = np.sum(dxhat * xhat, axis=1, keepdims=True)     # (N,1)
    dx = (inv_std / D) * (D * dxhat - sum_dxhat - xhat * sum_dxhat_xhat)

    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Inverted dropout forward.
    """
    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])

    mask, out = None, None

    if mode == "train":
        mask = (np.random.rand(*x.shape) < p) / p
        out = x * mask
    elif mode == "test":
        out = x
    else:
        raise ValueError("Invalid dropout mode")

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)
    return out, cache


def dropout_backward(dout, cache):
    """
    Inverted dropout backward.
    """
    dropout_param, mask = cache
    mode = dropout_param["mode"]

    if mode == "train":
        dx = dout * mask
    elif mode == "test":
        dx = dout
    else:
        raise ValueError("Invalid dropout mode")
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    Naive conv forward.
    x: (N, C, H, W), w:(F, C, HH, WW), b:(F,)
    """
    stride = conv_param["stride"]
    pad = conv_param["pad"]

    N, C, H, W = x.shape
    F, C_, HH, WW = w.shape
    assert C == C_

    # Output spatial size
    H_out = 1 + (H + 2 * pad - HH) // stride
    W_out = 1 + (W + 2 * pad - WW) // stride

    # Pad input
    x_pad = np.pad(x, ((0,0),(0,0),(pad,pad),(pad,pad)), mode='constant')

    out = np.zeros((N, F, H_out, W_out), dtype=x.dtype)

    for n in range(N):
        for f in range(F):
            for i in range(H_out):
                for j in range(W_out):
                    h0 = i * stride
                    w0 = j * stride
                    window = x_pad[n, :, h0:h0+HH, w0:w0+WW]          # (C,HH,WW)
                    out[n, f, i, j] = np.sum(window * w[f]) + b[f]

    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    Naive conv backward.
    dout: (N, F, H_out, W_out)
    """
    x, w, b, conv_param = cache
    stride = conv_param["stride"]
    pad = conv_param["pad"]

    N, C, H, W = x.shape
    F, C_, HH, WW = w.shape
    H_out, W_out = dout.shape[2], dout.shape[3]

    x_pad = np.pad(x, ((0,0),(0,0),(pad,pad),(pad,pad)), mode='constant')
    dx_pad = np.zeros_like(x_pad)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)

    db = np.sum(dout, axis=(0,2,3))                      # (F,)

    for n in range(N):
        for f in range(F):
            for i in range(H_out):
                for j in range(W_out):
                    h0 = i * stride
                    w0 = j * stride
                    window = x_pad[n, :, h0:h0+HH, w0:w0+WW]          # (C,HH,WW)
                    dw[f] += window * dout[n, f, i, j]
                    dx_pad[n, :, h0:h0+HH, w0:w0+WW] += w[f] * dout[n, f, i, j]

    dx = dx_pad[:, :, pad:pad+H, pad:pad+W]
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    Naive max-pool forward.
    x: (N, C, H, W)
    pool_height, pool_width, stride in pool_param
    """
    N, C, H, W = x.shape
    pool_height = pool_param["pool_height"]
    pool_width = pool_param["pool_width"]
    stride = pool_param["stride"]

    H_out = 1 + (H - pool_height) // stride
    W_out = 1 + (W - pool_width) // stride

    out = np.zeros((N, C, H_out, W_out), dtype=x.dtype)

    for n in range(N):
        for c in range(C):
            for i in range(H_out):
                for j in range(W_out):
                    h0 = i * stride
                    w0 = j * stride
                    window = x[n, c, h0:h0+pool_height, w0:w0+pool_width]
                    out[n, c, i, j] = np.max(window)

    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    Naive max-pool backward.
    """
    x, pool_param = cache
    N, C, H, W = x.shape
    pool_height = pool_param["pool_height"]
    pool_width = pool_param["pool_width"]
    stride = pool_param["stride"]

    H_out = 1 + (H - pool_height) // stride
    W_out = 1 + (W - pool_width) // stride

    dx = np.zeros_like(x)

    for n in range(N):
        for c in range(C):
            for i in range(H_out):
                for j in range(W_out):
                    h0 = i * stride
                    w0 = j * stride
                    window = x[n, c, h0:h0+pool_height, w0:w0+pool_width]
                    m = np.max(window)
                    mask = (window == m)
                    dx[n, c, h0:h0+pool_height, w0:w0+pool_width] += mask * dout[n, c, i, j]

    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Spatial BatchNorm forward.
    x: (N, C, H, W); gamma, beta: (C,)
    """
    N, C, H, W = x.shape
    # reshape to (N*H*W, C), apply vanilla BN on channels
    x2 = x.transpose(0, 2, 3, 1).reshape(-1, C)
    out2, cache = batchnorm_forward(x2, gamma, beta, bn_param)
    out = out2.reshape(N, H, W, C).transpose(0, 3, 1, 2)
    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Spatial BatchNorm backward.
    dout: (N, C, H, W)
    """
    N, C, H, W = dout.shape
    dout2 = dout.transpose(0, 2, 3, 1).reshape(-1, C)
    dx2, dgamma, dbeta = batchnorm_backward(dout2, cache)
    dx = dx2.reshape(N, H, W, C).transpose(0, 3, 1, 2)
    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """
    Spatial GroupNorm forward.
    x: (N, C, H, W); gamma,beta: (1,C,1,1); G groups
    """
    eps = gn_param.get("eps", 1e-5)
    N, C, H, W = x.shape
    assert C % G == 0

    xg = x.reshape(N, G, C // G, H, W)                    # (N,G, C/G, H, W)
    mu = np.mean(xg, axis=(2, 3, 4), keepdims=True)
    var = np.var(xg, axis=(2, 3, 4), keepdims=True)
    std = np.sqrt(var + eps)
    xhatg = (xg - mu) / std                               # same shape as xg

    xhat = xhatg.reshape(N, C, H, W)
    out = gamma * xhat + beta

    cache = (G, x, xhat, mu, var, std, gamma, beta, eps)
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """
    Spatial GroupNorm backward.
    """
    G, x, xhat, mu, var, std, gamma, beta, eps = cache
    N, C, H, W = x.shape
    assert C % G == 0

    dbeta = np.sum(dout, axis=(0, 2, 3), keepdims=True)                 # (1,C,1,1)
    dgamma = np.sum(dout * xhat, axis=(0, 2, 3), keepdims=True)         # (1,C,1,1)

    dxhat = dout * gamma                                                # (N,C,H,W)
    dxhatg = dxhat.reshape(N, G, C // G, H, W)
    xhatg = xhat.reshape(N, G, C // G, H, W)

    inv_std = 1.0 / std
    Dg = (C // G) * H * W

    # dxg = (1/Dg) * inv_std * (Dg*dxhatg - sum(dxhatg) - xhatg*sum(dxhatg*xhatg))
    sum_dxhatg = np.sum(dxhatg, axis=(2, 3, 4), keepdims=True)
    sum_dxhatg_xhatg = np.sum(dxhatg * xhatg, axis=(2, 3, 4), keepdims=True)
    dxg = (inv_std / Dg) * (Dg * dxhatg - sum_dxhatg - xhatg * sum_dxhatg_xhatg)

    dx = dxg.reshape(N, C, H, W)
    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Multiclass SVM loss and gradient.
    x:(N,C), y:(N,)
    """
    N, C = x.shape
    correct = x[np.arange(N), y][:, None]                # (N,1)
    margins = np.maximum(0, x - correct + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N

    dx = (margins > 0).astype(x.dtype)
    row_sum = np.sum(dx, axis=1)
    dx[np.arange(N), y] -= row_sum
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Softmax loss and gradient.
    x:(N,C), y:(N,)
    """
    N, C = x.shape
    scores = x - np.max(x, axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    loss = -np.sum(np.log(probs[np.arange(N), y])) / N

    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx

