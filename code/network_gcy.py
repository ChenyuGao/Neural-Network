"""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

#### Libraries
# Standard library
import random

# Third-party libraries
import numpy as np

class QuadraticCost(object):

    @staticmethod
    def fn(a, y):
        return 0.5 * np.linalg.norm(a - y) ** 2

    @staticmethod
    def delta(z, a, y):
        return (a - y) * activa_func_prime(z)

    @staticmethod
    def delta_last(z, a, y):
        return (a - y) * softmax_prime(z)


class CrossEntropyCost(object):

    @staticmethod
    def fn(a, y):
        return np.sum(np.nan_to_num( - y * np.log(a) - (1 - y) * np.log(1 - a)))

    @staticmethod
    def delta(z, a, y):
        return ((1 - y) / (1 - a) - y / a) * activa_func_prime(z)

    @staticmethod
    def delta_last(z, a, y):
        return ((1 - y) / (1 - a) - y / a) * softmax_prime(z)

class Network(object):
    def __init__(self, sizes, cost = QuadraticCost):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.cost = cost
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x) for x, y in list(zip(sizes[:-1], sizes[1:]))]

    def feedforward(self, a):
        for b, w in list(zip(self.biases, self.weights))[:-1]:
            a = activa_func(np.dot(w, a) + b)
        # use softmax() as activation function at the last layer
        a = softmax(np.dot(self.weights[-1], a) + self.biases[-1])
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print ("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test))
                if j == epochs - 1:
                    print ("Epoch complete".format(j))
                    self.save()

    def update_mini_batch(self, mini_batch, eta):
        x, y = mini_batch[0]
        for i, o in mini_batch[1:]:
            x = np.column_stack((x, i))
            y = np.column_stack((y, o))
        delta_nabla_b, delta_nabla_w = self.backprop(x, y)
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in list(zip(self.weights, delta_nabla_w))]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in list(zip(self.biases, delta_nabla_b))]

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in list(zip(self.biases, self.weights))[:-1]:
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = activa_func(z)
            activations.append(activation)
        # use softmax() as activation function at the last layer
        z = np.dot(self.weights[-1], activations[-1]) + self.biases[-1]
        zs.append(z)
        activation = softmax(z)
        activations.append(activation)
        delta = (self.cost).delta_last(zs[-1], activations[-1], y)
        nabla_b[-1] = np.sum(delta, axis = 1).reshape(self.sizes[-1], 1)
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = activa_func_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = np.sum(delta, axis = 1).reshape(self.sizes[-l], 1)
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def save(self):
        np.save("w.npy", self.weights)
        np.save("b.npy", self.biases)

#### Miscellaneous functions
def activa_func(z):
    return cos(z)

def activa_func_prime(z):
    return cos_prime(z)

# QuadraticCost: learning rate = 3.0
# CrossEntropyCost: learning rate = 3.0
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))

# QuadraticCost: learning rate = 0.2
# CrossEntropyCost: learning rate = 0.1
def cos(z):
    return np.cos(z)

def cos_prime(z):
    return -np.sin(z)

# QuadraticCost: learning rate = 0.1
# CrossEntropyCost: learning rate = 0.1
def tanh(z):
    return (np.exp(z)-np.exp(-z)) / (np.exp(z)+np.exp(-z))

def tanh_prime(z):
    return 1.0 - np.square(tanh(z))

# QuadraticCost: learning rate = 0.05
# CrossEntropyCost: learning rate = 0.05
# if learning rate is too big, it will send error
def ReLU(z):
    return np.maximum(0.01 * z, z)

def ReLU_prime(z):
    z[z<=0] = 0.0
    z[z>0] = 1.0
    return z

#  use softmax() as activation function at the last layer, then every activation function is perfect
def softmax(z):
    return np.exp(z) / np.sum(np.exp(z), axis = 0).reshape(1, z.shape[1])

def softmax_prime(z):
    return softmax(z) * (1 - softmax(z))