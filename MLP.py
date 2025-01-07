# -*- coding: utf-8 -*-
'''
contains the Layer and MLP classes which are collectively used to 
implement feedforward neural networks

@ name:             MLP.py
@ last update:      07-29-2024
@ author:           Tyler Leake
'''

import numpy as np
from Dataset import Dataset



class Layer:
    
    def __init__(self,
                 input_size,
                 output_size,
                 initialization = ['random', 'xavier'], 
                 activation_fn = ['softmax', 'sigmoid', 'relu', 'tanh'],
                 loss_fn = ['cross-entropy', 'mean squared error']):
        
        '''
        implementation of a single fully-connected layer; collections of
        layers make up the network architecture in the MLP class
        
        params:
            
        input_size (int); number of input neurons
        output_size (int); number of outout neurons
        initialization (str); initialization method to use for weights & bias
        activation_fn (str); activation function to use for layer
        loss_fn (str); loss function to use for layer
        '''
        
        self.input_size = input_size
        self.output_size = output_size
        self.input = None
        self.output = None
        
        self.weights = None
        self.bias = None
        self.d_weights = None
        self.d_bias = None
        
        self.loss = None
        self.init = initialization
        self.activation_fn = activation_fn
        self.loss_fn = loss_fn
        
        if initialization == 'random': 
            self.random_initialization(input_size, output_size)
            
        if initialization == 'xavier': 
            self.xavier_initialization(input_size, output_size)
        
    def forward(self, input_data):
        '''linear forward pass (Z = XW + b)'''
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        self.output = self.activate(self.output)
        
    def backward(self, d_output, lr, batch_size):
        '''backpropagation of gradients'''
        
        # incoming * outgoing
        d_activation = self.compute_activation_grad() * d_output
        
        # compute the gradients of the bias and weights
        self.d_weights = np.dot(self.input.T, d_activation)
        self.d_bias = np.sum(d_activation, axis=0, keepdims=True)
        
        # compute the gradient for the next layer
        d_input = np.dot(d_activation, self.weights.T)
        
        # update the weights/bias attributes
        self.weights -= lr * self.dweights
        self.biases -= lr * self.dbiases
        
        return d_input
        
    def activate(self, input_data):
        '''activation functions: sigmoid, relu and tanh'''
        
        if self.activation_fn == 'softmax':
            ex = np.exp(input_data - np.max(input_data, axis=-1, keepdims=True))
            return ex / np.sum(ex, axis=-1, keepdims=True)
        
        if self.activation_fn == 'sigmoid':
            return 1 / (1 + np.exp(-input_data))
        
        if self.activation_fn == 'relu':
            return np.maximum(0, input_data)
        
        if self.activation_fn == 'tanh':
            return np.tanh(input_data) 
        
    def measure_loss(self, Y_true, Y_pred):
        '''
        loss function calculation; returns loss as float
        
        params:
        Y_true (np.array); ground truth values
        Y_pred (np.array); values predicted by layer
        '''
        if self.loss_fn == 'cross-entropy':
            epsilon = 1e-15  
            Y_pred = np.clip(Y_pred, epsilon, 1-epsilon)  
            loss = -np.mean(np.sum(Y_true * np.log(Y_pred), axis=1))
            
        if self.loss_fn == 'mean squared error':
            loss = np.mean((Y_true - Y_pred)**2)
            
        return loss
        
    def compute_activation_grad(self, input_data):
        '''gradient calculations for activations; gradients returned as float'''
        
        if self.activation_fn == 'softmax':
            s = input_data
            jacobian = np.diagflat(s) - np.outer(s, s)
            return jacobian
        
        if self.activation_fn == 'sigmoid':
            return self.activate(input_data) * (1 - self.activate(input_data))
            
        if self.activation_fn == 'relu':
            return (input_data > 0).astype(np.float32)
        
        if self.activation_fn == 'tanh':
            return 1 - np.tanh(input_data)**2
    
    def comnpute_loss_grad(self, Y_true, Y_pred):
        '''gradient calculations for loss functions; returns gradient as float'''
        
        if self.loss_fn == 'cross-entropy':
            epsilon = 1e-15
            Y_pred = np.clip(Y_pred, epsilon, 1-epsilon)
            gradient = Y_pred - Y_true
        
        if self.loss_fn == 'mean squared error':
            gradient = 2 * (Y_pred - Y_true)
            
        return gradient
        
    def random_initialization(self, input_size, output_size):
        '''
        random weight/bias initialization
        
        params: input_size & output_size (int), no. of neurons in input/output
        return: none, self.weights and self.bias attributes updated directly
        '''
        self.weights = np.random.rand(input_size, output_size)
        self.bias = np.zeros(output_size)
        
    def xavier_initialization(self, input_size, output_size):
        '''
        xavier initialization for weights (bias set to zero)
        
        params: input_size & output_size (int), no. of neurons in input/output
        return: none, self.weights and self.bias attributes updated directly
        '''
        limit = np.sqrt(6 / (input_size + output_size))
        self.weights = np.random.uniform(-limit, limit, (input_size, output_size))
        self.bias = np.zeros(output_size)
        
    def reset(self):
        '''reset layer attributes'''
        
        self.input = None; self.output = None
        self.weights = None; self.d_weights = None
        self.bias = None; self.d_bias = None
        self.loss = None
            
        if self.init == 'random': 
            self.random_initialization(self.input_size, self.output_size)
        if self.init == 'xavier': 
            self.xavier_initialization(self.input_size, self.output_size)



class MLP():
    
    def __init__(self, 
                 train, 
                 layer_dimensions, 
                 layer_initializations,
                 layer_activation_fns,
                 layer_loss_fns):
                 
        '''
        implementation of a multilayer perceptron with variable network size; 
        activation functions, loss functions and initialization methods can
        be adjusted for each individual layer
        
        params:
            
        train (Dataset): Dataset object to train/test on
        layer_dimensions (list of int); width of each layer to create
        layer_initializations (list of str); initialization method for each layer
        layer_activation_fns (list of str); activation function for each layer
        layer_loss_fns (list of str); loss function for each layer
        '''
        
        # self.X = training data; self.Y = ground truth values
        self.X = np.delete(train, train.response_idx, axis=1)
        self.Y = self.set_ground_truth(train)

        # Network Initialization
        # instantiate all Layer objects using the dimensions list; assert 
        # that the output layer contains the appropriate number of neurons
        
        if train.objective == 'classification': 
            assert layer_dimensions[:-1] == train.n_classes
            assert layer_activation_fns[-1] == "softmax"
            
        if train.objective == 'regression': 
            assert layer_dimensions[:-1] == 1
        
        self.layers = []
        self.layers.append(self.get_input_layer())
        
        for i in range(1, len(layer_dimensions)):
            n_in = layer_dimensions[i-1]
            n_out = layer_dimensions[i]
            initialization = layer_initializations[i]
            activation = layer_activation_fns[i]
            loss = layer_loss_fns[i]
            layer = Layer(n_in, n_out, initialization, activation, loss)
            self.layers.append(layer)
            
        self.training_losses = [] # accuracy and loss metrics
        self.using_autoencoder = False # provision for using autoencoder
        
    def set_input_layer(self):
        '''returns the input layer of the dataset'''
        
        input_layer = Layer()
        
        return input_layer
    
    def set_ground_truth(self, dataset):
        '''returns the ground truth values of the dataset'''
        hot_encoded = dataset.is_hot_encoded(dataset.response_idx)
        if hot_encoded: return self.decode(dataset, dataset.response_idx)
        if not hot_encoded: return dataset[:, dataset.response_idx]
            
    def decode(self, col_idx):
        '''
        decodes the hot encoded response column of a dataset into its 
        equivalent matrix representation if applicable
        ''' 
        hot_encoded_col = self.X.data[:, col_idx]
        hot_encoded_mat = np.array([elem for elem in hot_encoded_col])
        return hot_encoded_mat
    
    def forward(self, X, Y):
        '''forward pass through all MLP layers'''
        input_data = X
        for i in range(len(self.layers)):
            self.layers[i].forward(input_data)
            input_data = self.layers[i].output
            if (i+1) == len(self.layers): self.layers[i].measure_loss(Y)
            
    def backward(self, X, Y, lr, batch_size):
        '''backpropagation of gradients through all MLP layers'''
        d_output = self.layers[-1].compute_loss_grad(Y)
        for i in range(len(self.layers)-1, -1, -1):
            d_output = self.layers[i].backward(d_output, lr)
        
    def train(self, lr, batch_size, epochs):
        '''implementation of training method for MLP using minibatches'''
        
        # shuffle the data
        indices = np.random.permutation(self.X.shape[0])
        X_shuffled = self.X[indices]
        Y_shuffled = self.Y[indices]
    
        for i in range(epochs):
            
            for j in range(0, self.X.shape[0], batch_size):
                
                X_batch = X_shuffled[j:j + batch_size]
                Y_batch = Y_shuffled[j:j + batch_size]
            
                self.forward(X_batch, Y_batch)
                
                loss = self.layers[-1].loss
                self.training_losses.append(loss)
                if loss == 0: break
                
                self.backward(X_batch, Y_batch, lr)
            
    def eval(self, X, Y):
        '''evaluates model on test set (X) using the ground truth values (Y)'''
        
        predictions = self.forward(X)
        
        if self.X.objective == 'classification':
            predicted_classes = np.argmax(predictions, axis=1)
            true_classes = np.argmax(Y, axis=1)
            accuracy = np.mean(predicted_classes == true_classes)
            return accuracy
            
        if self.X.objective == 'regression':
            mse = np.mean((predictions - Y) ** 2)
            return mse
            
    def reset(self):
        '''reset model weights, biases gradients, loss measurements etc.'''
        for layer in self.layers: layer.reset()
        self.training_losses = []
        self.eval_loss = None

