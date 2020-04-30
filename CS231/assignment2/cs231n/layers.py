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
    X_flat = x.reshape(x.shape[0],-1)
    a = X_flat.dot(w) +b
    out = a
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
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    x_T = np.transpose(x.reshape(x.shape[0],-1))
    w_T = np.transpose(w)    
    dout_T = np.transpose(dout)
    dx, dw, db = dout.dot(w_T).reshape(x.shape), x_T.dot(dout),             dout_T.dot(np.ones(dout.shape[0]))



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
    out = np.maximum(0,x)

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
    dx = np.multiply(np.where(x < 0, 0, 1),dout)
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
    layernorm = bn_param.get('layernorm', 0)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':

        sample_mean = np.mean(x,axis = 0)
        sample_var = np.var(x,axis = 0)
        if layernorm ==0:
            running_mean = momentum * running_mean + (1 - momentum) * sample_mean
            running_var = momentum * running_var + (1 - momentum) * sample_var
        x_n = (x - sample_mean)/(np.sqrt(sample_var + eps))
        out = gamma * x_n + beta
        
        bn_param['running_mean'] = running_mean
        bn_param['running_var'] = running_var  
        cache = (gamma, beta, x, sample_mean, sample_var, eps, x_n, layernorm)

    elif mode == 'test':

        x_n = (x - running_mean)/(np.sqrt(running_var + eps))
        out = gamma * x_n + beta

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
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # N = 4
    gamma, beta, x, sample_mean, sample_var, eps, x_n, layernorm = cache
    m = x.shape[0]
    
    dx_n = dout*gamma
    dvar = np.sum( dx_n*(x - sample_mean),0 ) * (-1/2)*(sample_var+eps)**(-3/2)
    dmean = -np.sum(dx_n/np.sqrt(sample_var+eps),0)+\
        dvar*np.sum(-2*(x - sample_mean)/m,0)
    dx = dx_n/np.sqrt(sample_var+eps) + dvar*2*(x - sample_mean)/m + dmean/m
    if layernorm:
        dgamma = np.sum(dout*x_n,1)
        dbeta = np.sum(dout,1)
    else:
        dgamma = np.sum(dout*x_n,0)
        dbeta = np.sum(dout,0)        

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
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
    See the jupyter notebook for more hints.
     
    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None

    gamma, beta, x, sample_mean, sample_var, eps, x_n, layernorm = cache
    m = x.shape[0]
    
    dx_n = dout*gamma
    dvar = np.sum( dx_n*(x - sample_mean),0 ) * (-1/2)*(sample_var+eps)**(-3/2)
    dmean = -np.sum(dx_n/np.sqrt(sample_var+eps),0)+\
        dvar*np.sum(-2*(x - sample_mean)/m,0)
    dx = dx_n/np.sqrt(sample_var+eps) + dvar*2*(x - sample_mean)/m + dmean/m
    
    dgamma = np.sum(dout*x_n,0)
    dbeta = np.sum(dout,0)

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """
    Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.
    
    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get('eps', 1e-5)


    ln_param['mode'] = 'train' # same as batch norm in train mode
    ln_param['layernorm'] = 1
    # transpose x, gamma and beta
    out, cache = batchnorm_forward(x.T, gamma.reshape(-1,1),
                                   beta.reshape(-1,1), ln_param)
    # transpose output to get original dims
    out = out.T

    return out, cache


def layernorm_backward(dout, cache):
    """
    Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None


    dx, dgamma, dbeta = batchnorm_backward(dout.T, cache)
    # transpose gradients w.r.t. input, x, to their original dims
    dx = dx.T
    

    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.

    NOTE: Please implement **inverted** dropout, not the vanilla version of dropout.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    NOTE 2: Keep in mind that p is the probability of **keep** a neuron
    output; this might be contrary to some sources, where it is referred to
    as the probability of dropping a neuron output.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':


        mask = (np.random.rand(*x.shape) < p) / p # first dropout mask
        out = mask*x
        

    elif mode == 'test':

        out = x

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

        dx = dout*mask

    elif mode == 'test':
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input. 
        

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None

    stride = conv_param['stride']    
    pad = conv_param['pad']
    
    F, C, HH, WW = w.shape
    N, C, H, W = x.shape
    H_o = np.int(1 + (H + 2 * pad - HH) / stride)
    W_o = np.int(1 + (W + 2 * pad - WW) / stride)
    out = np.zeros((N, F, H_o, W_o))

    x_1 = x.copy()
    
    PadSeq = np.array([[0,0],[0,0],[pad,pad],[pad,pad]])
    x_1 = np.pad(x_1,PadSeq,mode='constant',constant_values=0)
    
    for Nj in range(N):
        for Fj in range(F):
            for hj in range(H_o):
                for wj in range(W_o):
                    hjs = hj*stride 
                    wjs = wj*stride
                    hje = hjs + HH
                    wje = wjs + WW
                    out[Nj,Fj,hj,wj] = np.sum(np.multiply(w[Fj,:,:,:],x_1[Nj,:,hjs:hje,wjs:wje]))+b[Fj]
    
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

    x, w, b, conv_param = cache
    stride = conv_param['stride']    
    pad = conv_param['pad']
    
    x_pad = x.copy()
    PadSeq = np.array([[0,0],[0,0],[pad,pad],[pad,pad]])
    x_pad = np.pad(x_pad,PadSeq,mode='constant',constant_values=0)
    
    F, C, HH, WW = w.shape
    N, C, H, W = x.shape
    H_o = np.int(1 + (H + 2 * pad - HH) / stride)
    W_o = np.int(1 + (W + 2 * pad - WW) / stride)
    dx = np.zeros(x_pad.shape)
    dw = np.zeros(w.shape)

    for Nj in range(N):
        for Fj in range(F):
            for hj in range(H_o):
                for wj in range(W_o):                    
                    hjs = hj*stride 
                    wjs = wj*stride
                    hje = hjs + HH
                    wje = wjs + WW
                    
                    dx[Nj, :, hjs:hje, wjs:wje] += w[Fj,:,:,:]*dout[Nj, Fj, hj,wj]
                    dw[Fj, :, :, :] += x_pad[Nj, :, hjs:hje, wjs:wje]*dout[Nj, Fj, hj,wj]
                    
    dx = dx[:,:,pad:-pad,pad:-pad]
    db = np.sum(dout,axis=(0,2,3))

    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here. Output size is given by 

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None

    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']
    N, C, H, W = x.shape
    H_o = np.int(1 + (H - pool_height) / stride)
    W_o = np.int(1 + (W - pool_width) / stride)
    out = np.zeros((N,C,H_o,W_o))
    
    for Nj in range(N):
        for Cj in range(C):
            for hj in range(H_o):
                for wj in range(W_o):
                    hjs = hj*stride 
                    wjs = wj*stride
                    hje = hjs + pool_height
                    wje = wjs + pool_width
                    
                    out[Nj, Cj, hj, wj] = np.max(x[Nj, Cj, hjs:hje, wjs:wje])
    
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None

    x, pool_param = cache
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']
    
    N, C, H, W = x.shape
    H_o = np.int(1 + (H - pool_height) / stride)
    W_o = np.int(1 + (W - pool_width) / stride)
    dx = np.zeros(x.shape)
    
    for Nj in range(N):
        for Cj in range(C):
            for hj in range(H_o):
                for wj in range(W_o):
                    hjs = hj*stride 
                    wjs = wj*stride
                    hje = hjs + pool_height
                    wje = wjs + pool_width
                    
                    #out[Nj, Cj, hj, wj] = np.max(x[Nj, Cj, hjs:hje, wjs:wje]) 
                    ind = np.argmax(x[Nj, Cj, hjs:hje, wjs:wje]) 
                    c_idx = ind % pool_width
                    r_idx = ind // pool_width
                    dx[Nj, Cj, hjs+r_idx, wjs+c_idx] = dout[Nj, Cj, hj, wj]
                       
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

    N, C, H, W = x.shape
    x_in = x.swapaxes(1,3)
    x_in = x_in.reshape((N*H*W,C))
    out, cache = batchnorm_forward(x_in, gamma, beta, bn_param)
    out = out.reshape((N ,W ,H, C))
    out = out.swapaxes(3,1)

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
    
    N, C, H, W = dout.shape
    dx_in = dout.swapaxes(1,3)
    dx_in = dx_in.reshape((N*H*W,C))
    
    dx, dgamma, dbeta = batchnorm_backward(dx_in, cache)
    dx = dx.reshape((N ,W ,H, C))
    dx = dx.swapaxes(3,1)

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """
    Computes the forward pass for spatial group normalization.
    In contrast to layer normalization, group normalization splits each entry 
    in the data into G contiguous pieces, which it then normalizes independently.
    Per feature shifting and scaling are then applied to the data, in a manner identical to that of batch normalization and layer normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get('eps',1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                # 
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, C, H, W = x.shape
    x = x.reshape(G*N,C//G*H*W)
    #gamma = np.ones((C*H*W//G))
    #beta = np.zeros((C*H*W//G))
    gamma = np.repeat(gamma,H*W//G)
    gamma = gamma.reshape(-1)
    
    beta = np.repeat(beta, H*W//G)
    beta = beta.reshape(-1)
    out, cache_L_F = layernorm_forward(x, gamma, beta, gn_param)

    out = out.reshape(N,C,H,W)
    cache = G, gn_param, cache_L_F
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial group normalization.

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
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    G, gn_param, cache_L_F = cache
    
    N, C, H, W = dout.shape
    dout = dout.reshape(G*N,C//G*H*W)
    dx, dgamma, dbeta = layernorm_backward(dout, cache_L_F)
    
    dx = dx.reshape(N,C,H,W)
    dgamma = dgamma.reshape(C,H*W//G)
    dbeta = dbeta.reshape(C,H*W//G)
    dgamma = dgamma.sum(1).reshape(1,C,1,1)
    dbeta = dbeta.sum(1).reshape(1,C,1,1)
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
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
