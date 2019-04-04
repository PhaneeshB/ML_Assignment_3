import numpy as np


class Neural_Net():
    def __ini__(network, input_size, layers, output_size, batch_size,learing_rate):
        network.layers = layers.append(output_size)
        network.learning_rate = learing_rate
        # create randomly initialized weight matrix for each layer
        network.weights = [np.random.random((network.layers[0], input_size))]
        network.bias = [np.random.random((network.layers[0], 1))]

        for i in range(1, len(network.layers)):
            network.weights.append(np.random.random((network.layers[i], network.layers[i-1])))
            network.bias.append(np.random.random((network.layers[i], 1)))
        network.batch_size = batch_size
        
        network.train_error = []

    def activation(self, X, code=1):
        # code 1 = sigmoid
        # code 2 = ReLU
        if code == 1:
            return 1/(1 + np.exp(-X))
        elif code == 2:
            return np.maximum(X, 0)

    def activation_diff(self, X, code=1):
        '''Activation differentiation wrt their arguments.

        Arguments:
            X {[type]} -- Activation ndarray to be differentiated

        Keyword Arguments:
            code {int} -- To select the type of activation used (default: {1})

        Returns:
            ndarray -- processed ndarray of the same dimensions as argument
        '''
        # code 1 = sigmoid
        # code 2 = ReLU
        if code == 1:
            return np.exp(X)/((1 + np.exp(X))**2)
        elif code == 2:
            return (X >= 0).astype(int)

    def forward_prop(network, input_instance):
        return

    def back_prop(network, input_instance, output_instance):
        return

    def train_SGD(network, features, labels, iterations):
        for iter in range(iterations):
            # feed forward 
            
            # output error
            # back prop
