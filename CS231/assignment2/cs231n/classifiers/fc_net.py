from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


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

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
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


        self.params['W1'] = np.random.normal(0,weight_scale,[input_dim, hidden_dim])
        self.params['b1'] = np.zeros(hidden_dim,)

        self.params['W2'] = np.random.normal(0,weight_scale,[hidden_dim, num_classes])
        self.params['b2'] = np.zeros(num_classes,)

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
        
        a1, cache1 = affine_relu_forward(X, self.params['W1'], self.params['b1'])     
        
        scores, cache2 = affine_forward(a1, self.params['W2'], self.params['b2']) # out=a, cache = (x, w, b)  
                

        if y is None:
            return scores

        loss, grads = 0, {}


        loss, dx = softmax_loss(scores, y)
        loss = loss + \
            self.reg*(np.sum(np.power(self.params['W1'],2)) +\
                np.sum(np.power(self.params['W2'],2)))/2
        
        dx, dw, db = affine_backward(dx, cache2)
        grads['W2'] = dw + (self.params['W2'])*self.reg
        grads['b2'] = db
        
        dx, dw, db = affine_relu_backward(dx, cache1) # fc_cache = (x, w, b)/ relu_cache = 
        grads['W1'] = dw + (self.params['W1'])*self.reg
        grads['b1'] = db        
        
        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=1, normalization=None, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.normalization = normalization
        self.use_dropout = dropout != 1
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
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ## first layer:       
        
        for i in range(1,self.num_layers+1):
            print(['w' + np.str(i)])
            if i == 1:
                InDim = input_dim
                OutDim = hidden_dims[i-1]
            elif i == self.num_layers:  
                InDim = hidden_dims[i-2]
                OutDim = num_classes                               
            else:
                InDim = hidden_dims[i-2]
                OutDim = hidden_dims[i-1]
            
            self.params['W'+np.str(i)] = np.random.normal(0,weight_scale,[InDim, OutDim])
            self.params['b'+np.str(i)] = np.zeros(OutDim,) 
        

        if normalization=='batchnorm':
            for i in range(1,self.num_layers):
                self.params['gamma'+np.str(i)]= np.array(1)
                self.params['beta'+np.str(i)]= np.array(0)
                
        if normalization=='layernorm':
            print('tesss')
            for i in range(1,self.num_layers):
                self.params['gamma'+np.str(i)]= np.array(1)
                self.params['beta'+np.str(i)]= np.array(0)                
        


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization=='batchnorm':
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]
        if self.normalization=='layernorm':
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.normalization=='batchnorm':
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
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

        # {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax
        
        cache = {}
        L2 = []
        L2 =0
        for i in range(1,self.num_layers):
            
            if i == 1:
                x = X
            
            w = self.params['W'+np.str(i)]
            b = self.params['b'+np.str(i)]
            L2 += self.reg*np.sum(np.power(w,2))/2                         
            
            out, cache['1l' +np.str(i)] = affine_forward(x, w, b)
            x = out
            
            if self.normalization=='batchnorm':
                gamma = self.params['gamma'+np.str(i)]
                beta = self.params['beta'+np.str(i)]
                bn_param = self.bn_params[i-1]
                out, cache['2l' +np.str(i)] = batchnorm_forward(x, gamma, beta, bn_param)
                x = out
            
            if self.normalization=='layernorm':
                gamma = self.params['gamma'+np.str(i)]
                beta = self.params['beta'+np.str(i)]
                out, cache['2l' +np.str(i)] = layernorm_forward(x, gamma, beta,{'mode': 'train'})
                x = out                        
            
            out, cache['3l' +np.str(i)] = relu_forward(x)            
            x = out
            
            if self.use_dropout:
                out, cache['4l' +np.str(i)]  = dropout_forward(x, self.dropout_param)
                x = out
        
        x = out 
        w = self.params['W'+np.str(self.num_layers)] 
        b = self.params['b'+np.str(self.num_layers)] 
        L2 += self.reg*np.sum(np.power(w,2))/2
        
        out, cache['5l' +np.str(self.num_layers)] = affine_forward(x, w, b)
        
        scores = out        



        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        loss, dx = softmax_loss(scores, y)
        #L2 = 
        loss = loss + L2
            
        dout = dx
        Cache = cache['5l' +np.str(self.num_layers)]
        dx, dw, db = affine_backward(dout, Cache)
        grads['W'+np.str(self.num_layers)] = dw + (self.params['W'+np.str(self.num_layers)])*self.reg
        grads['b'+np.str(self.num_layers)] = db
        
            
        for i in range(self.num_layers-1,0,-1):    
            
            if self.use_dropout:
                dout = dx
                Cache = cache['4l' +np.str(i)]
                dx = dropout_backward(dout, Cache)
                
            dout = dx
            Cache = cache['3l' +np.str(i)]
            dx = relu_backward(dout, Cache)    
            
            if self.normalization=='batchnorm':
                dout = dx
                Cache = cache['2l' +np.str(i)]
                dx, dgamma, dbeta = batchnorm_backward(dout, Cache)
                grads['gamma'+np.str(i)] = np.sum(dgamma)
                grads['beta'+np.str(i)] = np.sum(dbeta)                
            
            if self.normalization=='layernorm':
                dout = dx
                Cache = cache['2l' +np.str(i)]
                dx, dgamma, dbeta = layernorm_backward(dout, Cache)
                grads['gamma'+np.str(i)] = np.sum(dgamma)
                grads['beta'+np.str(i)] = np.sum(dbeta)

            dout = dx
            Cache = cache['1l' +np.str(i)]
            dx, dw, db = affine_backward(dout, Cache)
            grads['W'+np.str(i)] = dw + (self.params['W'+np.str(i)])*self.reg
            grads['b'+np.str(i)] = db                        


        return loss, grads
