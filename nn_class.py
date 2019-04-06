import numpy as np
import onehot as enc


class Neural_Net():
    def __init__(network, input_size, layers, output_size, batch_size, learing_rate):
        network.layers = layers.append(output_size)
        network.learning_rate = learing_rate
        # create randomly initialized weight matrix for each layer
        network.weights = [np.random.random((network.layers[0], input_size))]
        network.bias = [np.random.random((network.layers[0], 1))]
        for i in range(1, len(network.layers)):
            network.weights.append(np.random.random((network.layers[i], network.layers[i-1])))
            network.bias.append(np.random.random((network.layers[i], 1)))
        
        network.batch_size = batch_size
        
        network.activation_type = 1


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

    def forward_prop(network, input):

        lin_layer = [0 for i in range(len(layers))]
        act_layer = [0 for i in range(len(layers))]

        # first layer
        lin_layer[0] = (network.weights[0] @ input) + network.bias[0]
        act_layer[0] = network.activation(lin_layer[0], network.network.activation_type)

        # middle layers (but the last)
        for l in range(1, len(network.layers)-1):
            lin_layer[l] = (network.weights[l] @ act_layer[l-1]) + network.bias[l]
            act_layer[l] = network.activation(lin_layer[l], network.network.activation_type)
        
        # last layer (network.activation type to be sigmoid, always)
        l = len(network.layers) - 1
        lin_layer[l] = (network.weights[l] @ act_layer[l-1]) + network.bias[l]
        act_layer[l] = network.activation(lin_layer[l], 1)    

        return lin_layer, act_layer

    def back_prop(network, input_instance, output_instance):
        return

    def train_SGD(network, features, labels, iterations):
        for iter in range(iterations):
            for batch in range(number_of_batches):

                # feed forward 
                
                # output error

                # back prop
