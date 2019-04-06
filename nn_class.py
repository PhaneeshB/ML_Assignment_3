import numpy as np
import onehot as enc
import pandas as pd
import matplotlib.pyplot as plt


class Neural_Net():
    def __init__(network, input_size, layers, output_size, batch_size,
                 learing_rate):
        network.layers = layers
        network.layers.append(output_size)
        network.learning_rate = learing_rate
        # create randomly initialized weight matrix for each layer
        network.weights = [np.random.random((network.layers[0], input_size))]
        network.bias = [np.random.random((network.layers[0], 1))]
        for i in range(1, len(network.layers)):
            network.weights.append(np.random.random((network.layers[i],
                                                     network.layers[i-1])))
            network.bias.append(np.random.random((network.layers[i], 1)))
        
        network.batch_size = batch_size
        
        network.activation_type = 1

        network.temp_linearity = [0 for i in range(len(network.layers))]
        network.temp_activation = [0 for i in range(len(network.layers))]
        network.temp_del = [0 for i in range(len(network.layers))]

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

    def _compute_acitvation(network, layer_num):
        # mention the convention
        # layer 0 : first layer in the network for which 
        # the input is feature vector(s)
        # assume the linearity to be pre-calculated for layer0
        # layer last: activation is always sigmoid therefore 
        # layers i
        if layer_num == 0:
            network.temp_activation[0] = \
            network.activation(network.temp_linearity[0], network.activation_type)
        
        elif layer_num == len(network.layers) - 1:
            wx = network.weights[layer_num] @ network.temp_activation[layer_num - 1]
            network.temp_linearity[layer_num] = wx + network.bias[layer_num]
            network.temp_activation[layer_num] = network.activation(network.temp_linearity[layer_num], 1)
        
        else:
            wx = network.weights[layer_num] @ network.temp_activation[layer_num - 1]
            network.temp_linearity[layer_num] = wx + network.bias[layer_num]
            network.temp_activation[layer_num] = network.activation(network.temp_linearity[layer_num], network.activation_type)

    def forward_prop(network, feature_input):
        #feature input is already in the correct dimensions/orientation
        layer_length = len(network.layers)

        # first layer
        network.temp_linearity[0] = (network.weights[0] @ feature_input) + network.bias[0]
        network.temp_activation[0] = network.activation(network.temp_linearity[0], network.activation_type)

        # middle layears (but the last)
        for l in range(1, layer_length-1):
            network.temp_linearity[l] = (network.weights[l] @ network.temp_activation[l-1]) + network.bias[l]
            network.temp_activation[l] = network.activation(network.temp_linearity[l], network.network.activation_type)

        # last layer (network.activation type to be sigmoid, always)
        l = layer_length - 1
        network.temp_linearity[l] = (network.weights[l] @ network.temp_activation[l-1]) + network.bias[l]
        network.temp_activation[l] = network.activation(network.temp_linearity[l], 1)    

    def back_prop(network, batch_input, prediction_error):
        # prediction error is the difference of 
        # predicted and true values for all examples
        network.temp_del = [0 for i in range(len(network.layers))]
        network.temp_del[-1] = prediction_error * \
            network.activation_diff(network.temp_linearity[-1],
                                    network.activation_type)
        
        for l in range(len(network.layers)-2, -1, -1):
            network.temp_del[l] = \
                (network.weights[l+1].T @ network.temp_del[l+1]) * \
                network.activation_diff(network.temp_linearity[l],
                                        network.activation_type)

    def update_params(network, batch_input):
        batch_size = batch_input.shape[1]

        outer_prod = network.temp_del[0] @ batch_input.T
        delta_w = (outer_prod / batch_size) * network.learning_rate
        delta_b = (network.learning_rate * np.mean(network.temp_del[0], axis=1, keepdims=True))

        network.weights[0] -= delta_w
        network.bias[0] -= delta_b

        for l in range(1, len(network.layers)):
            outer_prod = network.temp_del[l] @ network.temp_activation[l-1].T

            delta_w = (outer_prod / batch_size) * network.learning_rate
            delta_b = (network.learning_rate * np.mean(network.temp_del[l], axis=1, keepdims=True))

            network.weights[l] -= delta_w
            network.bias[l] -= delta_b

    def train_SGD(network, features, labels, epochs, train_data, test_data):
        # labels and features are already onehot encoded
        # one row is one example therefore take transpose.

        training_error = []
        train_acc= []
        test_acc = []
        pred_values = []
        for iter in range(epochs):
            number_of_batches = int(np.ceil(features.shape[0] / batch_size))
            batch_training_error = []
            
            for batch in range(number_of_batches):
                # input processing
                batch_input = features[batch * network.batch_size:(batch + 1) * network.batch_size].T
                batch_output = labels[batch * network.batch_size:(batch + 1) * network.batch_size].T
                
                # feed forward 
                network.forward_prop(batch_input)
                
                # output error
                # using the cost funtion as 0.5 sum (error^2)

                diff = network.temp_activation[-1] - batch_output
                all_error = np.sum(0.5 * ((diff) ** 2), axis=0, keepdims=True)
                cost_minibatch = np.mean(all_error)

                # back prop
                network.back_prop(batch_input, diff)

                # update weights and biases for this batch
                network.update_params(batch_input)

                # error metrics collection
                batch_training_error.append(cost_minibatch)
                print(f'epoch# = {iter}, batch# = {batch}, average error = {cost_minibatch}')

            training_error.append(np.mean(batch_training_error))
            if iter % 100 == 0:
                pred_r, acc_train = network.prediction(train_data)
                pred_s, acc_test = network.prediction(test_data)
                train_acc.append(acc_train)
                test_acc.append(acc_test)
                pred_values.append((pred_r,pred_s))



        return training_error, train_acc, test_acc, pred_values

    def prediction(network, test_input):

        test_features = test_input.iloc[:, 0:85].values.T
        test_labels = test_input.iloc[:, 85:95].values.T

        network.forward_prop(test_features)
        prediction = network.temp_activation[-1]

        if test_labels.shape[1] == 1:
            prediction_labels = np.argmax(prediction)
        else:
            x = np.argmax(prediction, axis=0)
            # print(type(x), x.shape)
            prediction_labels = list(x)

        count = 0
        for i in range(test_labels.shape[1]):
            if test_labels[:, i][prediction_labels[i]] == 1:
                count += 1

        return prediction_labels, count*100.0/test_labels.shape[1]




def plot(x, y, xlabel, ylabel, title):
    # x = range(0, len(err))
    # y = err
    plt.plot(x, y, color='g')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()
    plt.clf()
########################### MAIN #########################




file_train = './PokerDataset/poker-hand-training-true.data'
file_test = './PokerDataset/poker-hand-testing.data'

train_data, test_data= enc.onehot_neural(file_train, file_test)
train_features = train_data.iloc[:, 0:85]
train_labels = train_data.iloc[:, 85:95]
print
batch_size = 100
input_layer = 85
hidden_layer = [20]
output_layer = 10
epochs = 2000
activation_type = 2
learning_rate = 1

net = Neural_Net(input_layer, hidden_layer, output_layer, batch_size,
                 learning_rate)
print('Obj created')
train_err, train_acc, test_acc, pred_values = net.train_SGD(np.array(train_features), np.array(train_labels), epochs)
print('training done')
pred, acc = net.prediction(train_data)
print(f'Accuracy Train= {acc}')
print(f'values = {set(pred)}')
pred, acc = net.prediction(test_data)
print(f'Accuracy Test= {acc}')
print(f'values = {set(pred)}')

plot(list(range(epochs)), train_err, 'iter', 'error train', 'Hidden Layer ')
plot(list(range(0, epochs, 100)), train_acc, 'Epoch#', 'Train Accuracy', 'For hidden layer 20')
plot(list(range(0, epochs, 100)), test_acc, 'Epoch#', 'Test Accuracy', 'For Hidden layer 20')
