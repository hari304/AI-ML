pass
from cs231n.layers import *
from cs231n.fast_layers import *


def affine_relu_forward(x, w, b,dropout_param = None):
    """
    Convenience layer that performs an affine transform followed by a ReLU

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We drop each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.
    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    dropout_cache = None
    z, fc_cache = affine_forward(x, w, b)
    a, relu_cache = relu_forward(z)
    if dropout_param:
        out, dropout_cache = dropout_forward(a, dropout_param)
    else:
        out = a
    cache = (fc_cache, relu_cache,dropout_cache)
    return out, cache #out: a, fc_cache: x,w,b relu_cache: z


def affine_relu_backward(dout, cache):
    """
    Backward pass for the affine-relu convenience layer
    """
    fc_cache, relu_cache,dropout_cache = cache
    if dropout_cache:
        da = dropout_backward(dout, dropout_cache) #dropout_cache : dropout_param, mask
    else: 
        da = dout
    dz = relu_backward(da, relu_cache) #relu_cache: z
    dx, dw, db = affine_backward(dz, fc_cache) #fc_cache: x,w,b
    return dx, dw, db 

def affine_bn_relu_forward(x, w, b,gamma,beta,bn_param,dropout_param = None):
    """
    Convenience layer that performs an affine transform followed by a batch norm and ReLU

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We drop each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    dropout_cache = None
    z, fc_cache = affine_forward(x, w, b)
    z_bn,bn_cache = batchnorm_forward(z, gamma, beta, bn_param)
    a, relu_cache = relu_forward(z_bn)
    if dropout_param:
        out, dropout_cache = dropout_forward(a, dropout_param)
    else:
        out = a
    cache = (fc_cache,bn_cache, relu_cache,dropout_cache) 
    return out, cache
  

def affine_bn_relu_backward(dout, cache):
    """
    Backward pass for the affine-relu convenience layer
    """
    
    fc_cache,bn_cache, relu_cache,dropout_cache = cache 
    if dropout_cache:
        da = dropout_backward(dout, dropout_cache) #dropout_cache : dropout_param, mask
    else:
        da = dout
    dz = relu_backward(da, relu_cache) #relu_cache: z
    dz_bn,dgamma, dbeta = batchnorm_backward_alt(dz, bn_cache) #bn_cache : z_bn_norm,gamma,sigma,xmu,eps
    dx, dw, db = affine_backward(dz_bn, fc_cache) #fc_cache: x,w,b
    return dx, dw, db,dgamma, dbeta
    
    

def conv_relu_forward(x, w, b, conv_param):
    """
    A convenience layer that performs a convolution followed by a ReLU.

    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    out, relu_cache = relu_forward(a)
    cache = (conv_cache, relu_cache)
    return out, cache


def conv_relu_backward(dout, cache):
    """
    Backward pass for the conv-relu convenience layer.
    """
    conv_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db


def conv_bn_relu_forward(x, w, b, gamma, beta, conv_param, bn_param):
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    an, bn_cache = spatial_batchnorm_forward(a, gamma, beta, bn_param)
    out, relu_cache = relu_forward(an)
    cache = (conv_cache, bn_cache, relu_cache)
    return out, cache


def conv_bn_relu_backward(dout, cache):
    conv_cache, bn_cache, relu_cache = cache
    dan = relu_backward(dout, relu_cache)
    da, dgamma, dbeta = spatial_batchnorm_backward(dan, bn_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db, dgamma, dbeta


def conv_relu_pool_forward(x, w, b, conv_param, pool_param):
    """
    Convenience layer that performs a convolution, a ReLU, and a pool.

    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer
    - pool_param: Parameters for the pooling layer

    Returns a tuple of:
    - out: Output from the pooling layer
    - cache: Object to give to the backward pass
    """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    s, relu_cache = relu_forward(a)
    out, pool_cache = max_pool_forward_fast(s, pool_param)
    cache = (conv_cache, relu_cache, pool_cache)
    return out, cache


def conv_relu_pool_backward(dout, cache):
    """
    Backward pass for the conv-relu-pool convenience layer
    """
    conv_cache, relu_cache, pool_cache = cache
    ds = max_pool_backward_fast(dout, pool_cache)
    da = relu_backward(ds, relu_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db
