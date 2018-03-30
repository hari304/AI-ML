from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    num_train = x.shape[0]
    x_shape = x.shape
    x = x.reshape(num_train,-1) #X: NXD
    #out = relu_forward(np.dot(x,w)+b)[0] #NXD*DXM+1XM = NXM
    out = np.dot(x,w)+b #NXD*DXM+1XM = NXM
    x = x.reshape(x_shape)
    #print(out)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    num_train = x.shape[0] # N
    num_out_layer = w.shape[1] #M
    x_shape = x.shape
    x = x.reshape(num_train,-1) #X: NXD
    dout = dout.reshape((num_train,num_out_layer)) #NXM this step is to ensure incoming dout has correct shape
    dw = np.dot(x.transpose(),dout).reshape(w.shape) #DXN * NXM =DXM
    db = np.sum(dout,axis=0) #1XM
    dx = np.dot(dout,w.transpose()) # NXM * MXD = NXD
    dx = dx.reshape(x_shape) #NXD_1...D_K
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    out = np.maximum(0,x)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    mask = np.array([x>=0],dtype=int).reshape(x.shape)
    dx = dout*mask
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #######################################################################
        #sample_mean = np.mean(x, axis = 0) # 1XD
        sample_mean = (1/N)*(np.sum(x,axis = 0)) #1XD
        #sample_var = np.std(x, axis =0) #1XD
        sample_var = (1/N)*(np.sum(np.square(x - sample_mean),axis=0)) #1XD
        sigma = np.sqrt(sample_var+eps) # 1XD
        xmu = (x-sample_mean) #NXD - 1XD = NXD
        x_norm = xmu/sigma #(NXD-1XD)/1XD = NXD
        out = gamma*x_norm + beta
        running_mean = momentum*running_mean + (1-momentum)*sample_mean
        running_var = momentum*running_var + (1-momentum)*sample_var
        cache = (x_norm,gamma,sigma,xmu,eps) #x_norm : NXD gamma:1XD, sigma:1XD, xmu:NXD
        
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        x_norm = (x-running_mean)/np.sqrt(running_var) #(NXD-1XD)/1XD
        out = gamma*x_norm + beta
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    ###########################################################################
    
    x_norm,gamma,sigma,xmu,eps = cache #x_norm : NXD gamma:1XD, sigma:1XD, xmu:NXD
    N = dout.shape[0] #dout:NXD
    dbeta = np.sum(dout,axis=0) # 1XD 
    dgamma = np.sum(dout*x_norm,axis=0) #NXD * NXD = NXD
    dxnr = dout # NXD
    dxn = dxnr*gamma #NXD * NXD = NXD
    dxmu_1 = dxn*(1/sigma) # NXD / 1XD = NXD
    disigma = np.sum(dxn*xmu,axis =0) # sum(NXD*NXD) = 1XD
    #dmu = dxmu_1*(-1) #NXD 
    dsigma = disigma*(-1/np.square(sigma)) #1XD / 1XD = 1XD
    dvar = dsigma*(1/2)*(1/(sigma+eps)) #1XD / 1XD = 1XD
    dxmu2 = dvar*(1/N)*np.ones((dout.shape)) #1XD * NXD = 1XD
    dxmu_2 = dxmu2*2*xmu #1XD*NXD = NXD
    dxmu = dxmu_1+dxmu_2 #NXD + NXD = NXD
    dx_1 = dxmu #NXD
    dmu = np.sum(dxmu*(-1),axis=0) #1XD
    dx_2 = dmu*(1/N)*np.ones((dout.shape)) #1XD * NXD = NXD
    dx = dx_1 + dx_2 #NXD + NXD = NXD 
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    x_norm,gamma,sigma,xmu,eps = cache #x_norm : NXD gamma:1XD, sigma:1XD, xmu:NXD
    dout =dout.reshape(x_norm.shape) #dout takes funny shapes! soo irritating 
    N = dout.shape[0] #dout:NXD
    dgamma = np.sum(dout*x_norm,axis=0) #NXD * NXD = 1XD
    dbeta = np.sum(dout,axis=0) # 1XD 
    dx = gamma*(1/N)*(1/np.sqrt(np.square(sigma)+eps))*(N*dout - x_norm*dgamma - dbeta)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We drop each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        d = np.random.rand(x.shape[0],x.shape[1])
        mask = d < 1-p
        x = np.multiply(x,mask)
        x /=(1-p)
        out = x
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        out = x
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        p = dropout_param['p']
        dx = dout*(1/(1-p))*mask
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == 'test':
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width HH.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    pad = conv_param['pad']
    stride = conv_param['stride']
    N = x.shape[0]
    C = x.shape[1]
    H = x.shape[2]
    W = x.shape[3]
    F = w.shape[0]
    HH = w.shape[2]
    WW = w.shape[3]
    x_pad = np.pad(x,((0,0),(0,0),(pad,pad),(pad,pad)),'constant', constant_values=(0,0)) #X :(N,C,H+2pad,W+2pad
    H_out = int((H+2*pad-HH)/stride) + 1
    W_out = int((W+2*pad-WW)/stride) + 1
    out = np.zeros((N,F,H_out,W_out))
    for f in range(F):
        for hout in range(H_out):
            for wout in range(W_out):
                x_conv = x_pad[:,:,hout*stride:(hout*stride+HH),wout*stride:(wout*stride+WW)]
                conv = x_conv*w[f]
                conv_flat = conv.reshape((x_conv.shape[0],-1))
                out[:,f,hout,wout] = np.sum(conv_flat,axis = 1)+b[f]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    x, w, b, conv_param = cache
    pad = conv_param['pad']
    stride = conv_param['stride']
    N = x.shape[0]
    C = x.shape[1]
    H = x.shape[2]
    W = x.shape[3]
    F = w.shape[0]
    HH = w.shape[2]
    WW = w.shape[3]
    dw = np.zeros((F,C,HH,WW))
    db = np.zeros((F))
    dx_pad = np.zeros((N,C,(H+2*pad),(W+2*pad)))
    dx = np.zeros((N,C,H,W))
    x_pad = np.pad(x,((0,0),(0,0),(pad,pad),(pad,pad)),'constant', constant_values=(0,0)) #X :(N,C,H+2pad,W+2pad
    H_out = dout.shape[-2]
    W_out = dout.shape[-1]
    for f in range(F):
        for hout in range(H_out):
            for wout in range(W_out):
                x_conv = x_pad[:,:,hout*stride:(hout*stride+HH),wout*stride:(wout*stride+WW)]
                dconv = x_conv*dout[:,f,hout,wout].reshape(dout.shape[0],1,1,1)#dout(N,)* X_conv(N C HH WW) = dconv(N C HH WW)
                dw[f,:,:,:] +=np.sum(dconv,axis=0) # 1 C HH WW
                db[f] += np.sum(dout[:,f,hout,wout],axis = 0) #1 1 1 1
                dx_pad[:,:,hout*stride:(hout*stride+HH),wout*stride:(wout*stride+WW)] += dout[:,f,hout,wout].reshape(dout.shape[0],1,1,1)*w[f,:,:,:]
    dx = dx_pad[:,:,pad:-pad,pad:-pad]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max pooling forward pass                            #
    ###########################################################################
    N,C,H,W = x.shape
    PH = pool_param['pool_height']
    PW = pool_param['pool_width']
    stride = pool_param['stride']
    H_out = int((H-PH)/stride+1)
    W_out = int((W-PW)/stride+1)
    out = np.zeros((N,C,H_out,W_out))
    for c in range(C):
        for hout in range(H_out):
            for wout in range(W_out):
                x_pool = x[:,c,hout*stride:hout*stride+PH,wout*stride:wout*stride+PW]
                x_flat = x_pool.reshape(N,-1)
                out[:,c,hout,wout] = np.max(x_flat,axis=1)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max pooling backward pass                           #
    ###########################################################################
    x, pool_param = cache
    N,C,H,W = x.shape
    PH = pool_param['pool_height']
    PW = pool_param['pool_width']
    stride = pool_param['stride']
    H_out = int((H-PH)/stride+1)
    W_out = int((W-PW)/stride+1)
    dx = np.zeros((x.shape))
    for c in range(C):
        for hout in range(H_out):
            for wout in range(W_out):
                x_pool = x[:,c,hout*stride:hout*stride+PH,wout*stride:wout*stride+PW]
                x_flat = x_pool.reshape(x_pool.shape[0],-1)
                x_mask = (x_flat == np.max(x_flat,axis=1,keepdims=True)).reshape(x_pool.shape)
                dx[:,c,hout*stride:hout*stride+PH,wout*stride:wout*stride+PW] += x_mask*dout[:,c,hout,wout].reshape(dout.shape[0],1,1)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization using the vanilla   #
    # version of batch normalization defined above. Your implementation should#
    # be very short; ours is less than five lines.                            #
    ###########################################################################
    N, C, H, W = x.shape
    x_swapped = np.swapaxes(x,1,3)
    x_flat = x_swapped.reshape(N*H*W,C)
    out_flat, cache = batchnorm_forward(x_flat, gamma, beta, bn_param)
    out_swapped = out_flat.reshape(N,W,H,C)
    out = np.swapaxes(out_swapped,1,3)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization using the vanilla   #
    # version of batch normalization defined above. Your implementation should#
    # be very short; ours is less than five lines.                            #
    ###########################################################################
    N, C, H, W = dout.shape
    dout_swapped = np.swapaxes(dout,1,3)
    dout_flat = dout_swapped.reshape(N*H*W,C)
    dx_flat, dgamma, dbeta = batchnorm_backward_alt(dout_flat, cache)
    dx_swapped = dx_flat.reshape(N,W,H,C)
    dx = np.swapaxes(dx_swapped,1,3)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
