from builtins import range
from builtins import object
import os
import numpy as np
from ..layers import *
from ..layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(
        self,
        input_dim=3 * 32 * 32,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        
        # 初始化第一层（仿射层 + ReLU）
        # W1 的维度: (Input_dim, Hidden_dim)
        self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dim)
        # b1 的维度: (Hidden_dim,)
        self.params['b1'] = np.zeros(hidden_dim)
        
        # 初始化第二层（仿射层 + Softmax）
        # W2 的维度: (Hidden_dim, Num_classes)
        self.params['W2'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        # b2 的维度: (Num_classes,)
        self.params['b2'] = np.zeros(num_classes)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        # 从 self.params 中获取参数
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        
        # 架构: affine - relu - affine
        
        # 第一层: affine - relu
        # affine_relu_forward 是一个方便的层，它结合了 affine 和 relu
        hidden_layer_out, cache1 = affine_relu_forward(X, W1, b1)
        
        # 第二层: affine
        scores, cache2 = affine_forward(hidden_layer_out, W2, b2)
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        
        # 1. 计算损失（Loss）
        # 数据损失 (Softmax Loss)
        data_loss, dscores = softmax_loss(scores, y)
        
        # 正则化损失 (L2 Regularization)
        # 注意：包含 0.5 的因子
        reg_loss = 0.5 * self.reg * (np.sum(W1 * W1) + np.sum(W2 * W2))
        
        # 总损失
        loss = data_loss + reg_loss
        
        # 2. 计算梯度（Backward Pass）
        
        # 反向传播到第二层 (affine)
        # dscores 是来自 softmax 层的上游梯度
        dhidden, dW2, db2 = affine_backward(dscores, cache2)
        
        # 反向传播到第一层 (affine - relu)
        # dhidden 是来自第二层的上游梯度
        dX, dW1, db1 = affine_relu_backward(dhidden, cache1)
        
        # 3. 添加L2正则化项的梯度
        # d(0.5 * reg * W^2) / dW = reg * W
        dW1 += self.reg * W1
        dW2 += self.reg * W2
        
        # 4. 存储梯度
        grads['W1'] = dW1
        grads['b1'] = db1
        grads['W2'] = dW2
        grads['b2'] = db2

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

    def save(self, fname):
      """Save model parameters."""
      fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
      params = self.params
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
        self.params = params
        print(fname, "loaded.")
        return True



class FullyConnectedNet(object):
    """Class for a multi-layer fully connected neural network.

    Network contains an arbitrary number of hidden layers, ReLU nonlinearities,
    and a softmax loss function. This will also implement dropout and batch/layer
    normalization as options. For a network with L layers, the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional and the {...} block is
    repeated L - 1 times.

    Learnable parameters are stored in the self.params dictionary and will be learned
    using the Solver class.
    """

    def __init__(
        self,
        hidden_dims,
        input_dim=3 * 32 * 32,
        num_classes=10,
        dropout_keep_ratio=1,
        normalization=None,
        reg=0.0,
        weight_scale=1e-2,
        dtype=np.float32,
        seed=None,
    ):
        """Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout_keep_ratio: Scalar between 0 and 1 giving dropout strength.
            If dropout_keep_ratio=1 then the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
            are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
            initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
            this datatype. float32 is faster but less accurate, so you should use
            float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers.
            This will make the dropout layers deteriminstic so we can gradient check the model.
        """
        self.normalization = normalization
        self.use_dropout = dropout_keep_ratio != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        
        # 创建一个包含所有层维度的列表
        # [input_dim, hidden_dim_1, hidden_dim_2, ..., hidden_dim_L-1, num_classes]
        layer_dims = [input_dim] + hidden_dims + [num_classes]
        
        # 循环遍历所有层（从第1层到第 num_layers 层）
        for i in range(self.num_layers):
            layer_index = i + 1
            
            # W 和 b 的键
            W_key = 'W' + str(layer_index)
            b_key = 'b' + str(layer_index)
            
            # W 的维度: (dim_in, dim_out)
            # b 的维度: (dim_out,)
            dim_in = layer_dims[i]
            dim_out = layer_dims[i+1]
            
            # 初始化 W 和 b
            self.params[W_key] = weight_scale * np.random.randn(dim_in, dim_out)
            self.params[b_key] = np.zeros(dim_out)
            
            # 如果使用归一化 (batchnorm or layernorm)
            # 归一化参数在除了最后一层之外的每一层中
            if self.normalization is not None and layer_index < self.num_layers:
                gamma_key = 'gamma' + str(layer_index)
                beta_key = 'beta' + str(layer_index)
                
                # gamma (scale) 初始化为 1
                self.params[gamma_key] = np.ones(dim_out)
                # beta (shift) 初始化为 0
                self.params[beta_key] = np.zeros(dim_out)


        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout_keep_ratio}
            if seed is not None:
                self.dropout_param["seed"] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype.
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """Compute loss and gradient for the fully connected net.
        
        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
            scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
            names to gradients of the loss with respect to those parameters.
        """
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        
        current_input = X
        # caches 列表用于存储每一层的 (affine, norm, relu, dropout) cache
        caches = [] 
        
        # 循环遍历 L-1 个隐藏层
        # 架构: {affine - [norm] - relu - [dropout]}
        for i in range(self.num_layers - 1):
            layer_index = i + 1
            W = self.params['W' + str(layer_index)]
            b = self.params['b' + str(layer_index)]
            
            # 1. Affine
            affine_out, affine_cache = affine_forward(current_input, W, b)
            
            # 2. Normalization (Batchnorm / Layernorm)
            norm_cache = None
            norm_out = affine_out # 默认
            
            if self.normalization == 'batchnorm':
                gamma = self.params['gamma' + str(layer_index)]
                beta = self.params['beta' + str(layer_index)]
                bn_param = self.bn_params[i]
                norm_out, norm_cache = batchnorm_forward(affine_out, gamma, beta, bn_param)
            elif self.normalization == 'layernorm':
                gamma = self.params['gamma' + str(layer_index)]
                beta = self.params['beta' + str(layer_index)]
                ln_param = self.bn_params[i] # layernorm param 是空的
                norm_out, norm_cache = layernorm_forward(affine_out, gamma, beta, ln_param)
                
            # 3. ReLU
            relu_out, relu_cache = relu_forward(norm_out)
            
            # 4. Dropout
            dropout_cache = None
            if self.use_dropout:
                relu_out, dropout_cache = dropout_forward(relu_out, self.dropout_param)
            
            # 更新当前输入，为下一层做准备
            current_input = relu_out
            # 存储该层的缓存
            caches.append((affine_cache, norm_cache, relu_cache, dropout_cache))
            
        # 处理最后一层（第 L 层）
        # 架构: affine
        layer_index = self.num_layers
        W = self.params['W' + str(layer_index)]
        b = self.params['b' + str(layer_index)]
        
        scores, affine_cache = affine_forward(current_input, W, b)
        # 为最后一层存储缓存（只有 affine cache）
        caches.append((affine_cache, None, None, None))


        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early.
        if mode == "test":
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the   #
        # scale and shift parameters.                                              #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        
        # 1. 计算损失（Loss）
        data_loss, dscores = softmax_loss(scores, y)
        
        # L2 正则化损失
        reg_loss = 0.0
        for i in range(self.num_layers):
            layer_index = i + 1
            W = self.params['W' + str(layer_index)]
            reg_loss += np.sum(W * W)
            
        loss = data_loss + 0.5 * self.reg * reg_loss

        # 2. 计算梯度（Backward Pass）
        
        # dscores 是来自 softmax 的上游梯度
        dupstream = dscores 
        
        # 反向传播最后一层 (affine)
        layer_index = self.num_layers
        affine_cache, _, _, _ = caches[layer_index - 1] # 获取最后一层的缓存
        
        dupstream, dW, db = affine_backward(dupstream, affine_cache)
        
        # 存储梯度并添加正则化梯度
        grads['W' + str(layer_index)] = dW + self.reg * self.params['W' + str(layer_index)]
        grads['b' + str(layer_index)] = db
        
        # 反向传播 L-1 到 1 层
        # 架构: [dropout] - relu - [norm] - affine
        for i in range(self.num_layers - 2, -1, -1): # (L-2, L-3, ..., 0)
            layer_index = i + 1 # (L-1, L-2, ..., 1)
            
            affine_cache, norm_cache, relu_cache, dropout_cache = caches[i]
            
            # 4. Dropout backward
            if self.use_dropout:
                dupstream = dropout_backward(dupstream, dropout_cache)
                
            # 3. ReLU backward
            dupstream = relu_backward(dupstream, relu_cache)
            
            # 2. Normalization backward
            if self.normalization == 'batchnorm':
                dupstream, dgamma, dbeta = batchnorm_backward(dupstream, norm_cache)
                grads['gamma' + str(layer_index)] = dgamma
                grads['beta' + str(layer_index)] = dbeta
            elif self.normalization == 'layernorm':
                dupstream, dgamma, dbeta = layernorm_backward(dupstream, norm_cache)
                grads['gamma' + str(layer_index)] = dgamma
                grads['beta' + str(layer_index)] = dbeta

            # 1. Affine backward
            dupstream, dW, db = affine_backward(dupstream, affine_cache)
            
            # 存储梯度并添加正则化梯度
            grads['W' + str(layer_index)] = dW + self.reg * self.params['W' + str(layer_index)]
            grads['b' + str(layer_index)] = db

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


    def save(self, fname):
      """Save model parameters."""
      fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
      params = self.params
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
        self.params = params
        print(fname, "loaded.")
        return True