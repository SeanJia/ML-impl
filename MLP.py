__author__ = 'Zhiwei Jia'

import numpy as np
from numpy import *


class NeuralNet:

    def __init__(self, sizes, act=0):
        """the initialization function to create a new neural network with specified size"""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.sqrt(2.0/(x+y))*np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        self.act = act

        # these two are for momentum
        self.mom_b = [np.zeros((y, 1)) for y in sizes[1:]]
        self.mom_w = [np.zeros((y, x)) for x, y in zip(sizes[:-1], sizes[1:])]

    def forward(self, a):
        """the activation function"""
        count = 0
        for biases, weights in zip(self.biases, self.weights):
            count += 1
            if count < self.num_layers - 1:
                if self.act == 0:
                    a = Sigmoid(np.dot(weights, a) + biases)
                elif self.act == 1:
                    a = Tanh(np.dot(weights, a) + biases)
                else:
                    a = ReLU(np.dot(weights, a) + biases)
        a = Softmax(np.dot(self.weights[-1], a) + self.biases[-1])
        return a

    def training(self, train_data, test_data, store="empty", max_epoch=100, mini_batch_size=30,
                 learning_rate=1.0, momentum=0, reg=0):
        """the training function which use mini-batch stochastic gradient descent to minimize the cost"""
        num_samples = len(train_data)
        num_tests = len(test_data)
        for i in range(max_epoch):
            random.shuffle(train_data)
            mini_batches = [train_data[k:k+mini_batch_size] for k in range(0, num_samples, mini_batch_size)]
            for samples in mini_batches:
                self.mini_batch(samples, learning_rate, momentum, reg)

            # show the testing statistics after each epoch of all mini_batches
            print "Epoch {0}: on train_data, {1} % are correct".format(
                i+1, self.evaluate(train_data)/(num_samples+0.0)*100)
            print "    on test_data: {0} % are correct".format(
                self.evaluate(test_data)/(num_tests+0.0)*100)

    def mini_batch(self, mini_batch, learning_rate, m, r):
        """the mini_batch technique such that we update the weights by computing the gradient just using
           a portion of the samples for better global minimum"""

        # zero initialize current gradient (actually ahead due to momentum)
        D_biases = [np.zeros(b.shape) for b in self.biases]
        D_weights = [np.zeros(w.shape) for w in self.weights]

        # store previous momentum
        prev_m_b = [m_b + 0.0 for m_b in self.mom_b]
        prev_m_w = [m_w + 0.0 for m_w in self.mom_w]

        # compute gradient for each sample
        for (x, y) in mini_batch:
            curr_biases, curr_weights = self.backward(x, y)
            D_biases = [a + b for a, b in zip(D_biases, curr_biases)]
            D_weights = [a + b for a, b in zip(D_weights, curr_weights)]

        # update current change with regularization (actually gradient ahead)
        change_w = [learning_rate/len(mini_batch) * d + learning_rate * r * w
                       for w, d in zip(self.weights, D_weights)]
        change_b = [learning_rate/len(mini_batch) * d for d in D_biases]

        # update current momentum
        self.mom_w = [m * m_w - change for m_w, change in zip(self.mom_w, change_w)]
        self.mom_b = [m * m_b - change for m_b, change in zip(self.mom_b, change_b)]

        # update weights and biases
        self.weights = [w - m*p_m_w + (1.0+m)*m_w
                        for w, p_m_w, m_w in zip(self.weights, prev_m_w, self.mom_w)]
        self.biases = [w - m*p_m_b + (1.0+m)*m_b
                       for w, p_m_b, m_b in zip(self.biases, prev_m_b, self.mom_b)]

    def backward(self, x, y):
        """the backward propagation algorithm to compute the gradient of the cost function"""
        D_biases = [np.zeros(b.shape) for b in self.biases]
        D_weights = [np.zeros(w.shape) for w in self.weights]

        # feed-forward
        activation = x
        activations = [x]
        zs = []
        count = 0
        for b, w in zip(self.biases, self.weights):
            count += 1
            z = np.dot(w, activation)+b
            zs.append(z)
            if count < self.num_layers - 1:
                if self.act == 0:
                    activation = Sigmoid(z)
                elif self.act == 1:
                    activation = Tanh(z)
                else:
                    activation = ReLU(z)
                activations.append(activation)
            else:
                activation = Softmax(z)
                activations.append(activation)

        # backward
        # first the special case for output layer (due to soft-max)
        delta = activations[-1] - y
        D_biases[-1] = delta
        D_weights[-1] = np.dot(delta, activations[-2].T)

        # then compute the derivative of other hidden layers, from large to small
        for l in range(2, self.num_layers):
            z = zs[-l]
            if self.act == 0:
                grad = sigmoid_grad(z)
            elif self.act == 1:
                grad = tanh_grad(z)
            else:
                grad = relu_grad(z)
            delta = np.dot(self.weights[-l+1].T, delta) * grad
            D_biases[-l] = delta
            D_weights[-l] = np.dot(delta, activations[-l-1].T)

        return D_biases, D_weights

    def evaluate(self, test_data):
        """test the accuracy of the training outcome"""
        test_results = []
        for (x, y) in test_data:
            a = np.argmax(self.forward(x))
            b = np.argmax([y[i][0] for i in range(self.sizes[-1])])
            test_results.append((a, b))
        sum_correct = 0
        for (x, y) in test_results:
            if x == y:
                sum_correct += 1
        return sum_correct


def Sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))


def sigmoid_grad(z):
    """gradient of the sigmoid function."""
    return Sigmoid(z) * (1-Sigmoid(z))


def Softmax(z):
    """the softmax activation function for the output layer, best suitable for
       disjoint classes"""
    out = exp(z)
    sum_exp = sum(out)
    res = out/sum_exp
    return res


def Tanh(z):
    """the funny Tanh activation function"""
    return 1.7159 * np.tanh(2 / 3.0 * z)


def tanh_grad(z):
    """the gradient of funny Tanh(z)"""
    return 1.7159 * 2 / 3.0 * (1 - (np.tanh(2/3.0 * z)) ** 2)


def ReLU(z):
    """the ReLU activation function"""
    return np.max([z, np.zeros(z.shape)], axis=0)


def relu_grad(z):
    """the gradient of ReLU(z)"""
    index = z >= 0
    result = np.zeros(z.shape)
    result[index] = 1.0
    return result


def classification_vector(this_class, num_classes):
    """from a target value, return a vector to represent its value"""
    result = [0 for i in range(num_classes)]
    result[this_class] = 1
    return np.array([result]).T
