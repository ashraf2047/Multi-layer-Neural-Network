import numpy as np
from matplotlib import pyplot as plt 

#built-in python framework to make printing nicer
from pprint import pprint as pp 



def sigmoid(Z):
    #sigmoid function calculation
    return 1.0/(1.0 + np.exp(-Z))

def loss_function(P,Y):
    #This defines the loss function which the system hopes to minimize
    # P is the output we receive and Y is the desired output
    return -1.0 * (Y * np.log(P) + (1 - Y) * np.log(1.0 - P))

def forward_pass(X,network):

    # Make a list of layers within the network. 
    ls_layers = [key for key in network]

    # Create the bias input, which is needed for the bias weights.
    b = np.array([[1.0]])

    # Create the input vector for the layer 
    layer_input = np.concatenate((b,X))

    # Create a dict that will cache the FP results for each layer.
    cache = dict()

    #Iterate through the layers and perform the forward pass
    for layer in ls_layers:
        #matrix multiplication of weights and inputs
        Z = np.dot(network[layer], layer_input)
        A = sigmoid(Z)
        cache[layer] = (network[layer], Z, A)
        layer_input= np.concatenate((b,A))
    return cache, A[0]

def backward_pass(X, Y, cache):

    #This is the backpropagation algorithim to train the neural network, it uses gradient descent
    #The cache matrix is updated to store the new outputs

    P = cache[2][2]

    del_1_2 =  (P - Y) * cache[2][2][0] * (1 - cache[2][2][0]) 

    del_1_1 =  cache[2][0][0,1] * cache[1][2][0,0] * (1 - cache[1][2][0,0])

    del_2_1 =  cache[2][0][0,2] * cache[1][2][1,0] * (1 - cache[1][2][1,0])

    grad_W_layer2 = del_1_2 * np.array([1, cache[1][2][0], cache[1][2][1]])

    grad_W_layer1 = del_1_2 * np.array([[del_1_1],[del_2_1]]) * np.array([[1, X[0], X[1]],[1, X[0], X[1]]])

    grads = {1: grad_W_layer1, 2:grad_W_layer2}

    return grads



def update_parameters(network, grads, alpha):

    #This function updates the weights(and biases) of the neural network after each iteration of backprop
    #alpha is the learning rate of the system

    network[1] = network[1] - alpha * grads[1]
    network[2] = network[2] - alpha * grads[2]


def init_parameters(num_neurons,num_inputs):
    # This function initializes the weights of the a layer within the neural network

    # parameters:

     #    num_neurons : The number of neurons within this layer.
     #    num_inputs  : The number of neurons within the previous layer.

     #  return:
        
     #    W : A weight matrix for a given layer configuration.

     #  Example:
        
     #    num_neurons = 2
     #    num_inputs = 2

     #    W = | w_10   w_11   w_12  |
     #        | w_20   w_21   w_22  |

     #    return W

    # W_0 are the bias weights for each neuron. 
    W_0 = np.zeros((num_neurons,1))

    # W_1 are the weights connected to the inputs of the neuron.
    W_1 = np.random.randn(num_neurons, num_inputs)

    # W = [W_0, W_1] and is the weight matrix for a given layer.
    W = np.concatenate((W_0,W_1), axis = 1)
    return W

def display_network_predictions(network, epoch):
    #This function displays the output of the neural network after each training sequence
    #From this, we can see the progress of the network as it updates its weights after each iteration

    print("Results on epoch {}".format(epoch))
    _, P1 = forward_pass(np.array([[0],[0]]), network)
    _, P2 = forward_pass(np.array([[0],[1]]), network)
    _, P3 = forward_pass(np.array([[1],[0]]), network)
    _, P4 = forward_pass(np.array([[1],[1]]), network)

    print("{} --> {:.2f} --> {}".format([0,0], float(P1), (P1 > 0.5)*1))
    print("{} --> {:.2f} --> {}".format([0,1], float(P2), (P2 > 0.5)*1))
    print("{} --> {:.2f} --> {}".format([1,0], float(P3), (P3 > 0.5)*1))
    print("{} --> {:.2f} --> {}".format([1,1], float(P4), (P4 > 0.5)*1))
    print("\n\n")
    
def main():

    # Initialize the input to the hidden layer weight matrix.
    layer_1 = init_parameters(2,2)
    #pp(layer_1)

    # Initialize the hidden layer output weight matrix.
    layer_2 = init_parameters(1,2)
    #pp(layer_2)

    # Create dict to represent the network. 
    network = {1:layer_1, 2: layer_2}
    pp(network)

    #Define training set
    X = np.array([[0,0,1,1],
                  [0,1,0,1]])

    Y = np.array([0,1,1,0])

    #number of epochs
    num_epochs = 100000

    # Take average loss over each epoch
    losses = np.zeros((num_epochs, 1))

    #set learning rate (step size)
    alpha = 0.01

    #Toggle this value to change how many iterations are displayed
    epoch_display_trigger = 10

    display_network_predictions(network, 0)

    for epoch in range(num_epochs):
        loss = 0
        for i in range(len(Y)):
            x = np.array([[X[:,i][0]],[X[:,i][1]]])
            cache, P = forward_pass(x, network)
            grads = backward_pass(x, Y[i], cache)
            update_parameters(network, grads, alpha)
            loss = loss + loss_function(P, Y[i])
            losses[epoch, 0] = loss/4.0

            if epoch == epoch_display_trigger:
              epoch_display_trigger *= 10
              display_network_predictions(network, epoch)
  
    display_network_predictions(network, num_epochs)

    #Plot a graph of the loss function to see how the networks minimizes it over time
    plt.figure()
    plt.plot(losses)
    plt.show()

if __name__ == '__main__':
    main()

