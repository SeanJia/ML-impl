import numpy as np

class Autoencoder:

    def __init__(self, sizes, act=0):
        self.num_layers = len(sizes) * 2 - 1
        self.sizes = sizes
        self.biases = [np.zeros((y, 1)) for y in sizes[1:]]
        self.weights = [np.sqrt(2.0/(pre+nex)) * np.random.randn(nex, pre) for pre, nex in zip(sizes[:-1], sizes[1:])]
        self.act = act

        # these two are for momentum
        self.mom_b = [np.zeros((y, 1)) for y in sizes[1:]]
        self.mom_w = [np.zeros((nex, pre)) for pre, nex in zip(sizes[:-1], sizes[1:])]

    def forward(self, a):
        """the activation function"""
        for biases, weights in zip(self.biases, self.weights):
            if self.act == 0:
                a = Sigmoid(np.dot(weights, a) + biases)
            elif self.act == 1:
                a = Tanh(np.dot(weights, a) + biases)
            else:
                a = ReLU(np.dot(weights, a) + biases)
        return a

def Sigmoid(z):
    """The sigmoid function."""
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_grad(z):
    """gradient of the sigmoid function."""
    return Sigmoid(z) * (1 - Sigmoid(z))

def Tanh(z):
    """the funny Tanh activation function"""
    return 1.7159 * np.tanh(2 / 3.0 * z)

def tanh_grad(z):
    """the gradient of funny Tanh(z)"""
    return 1.7159 * 2 / 3.0 * (1 - (np.tanh(2 / 3.0 * z)) ** 2)

def ReLU(z):
    """the ReLU activation function"""
    return np.max([z, np.zeros(z.shape)], axis=0)

def relu_grad(z):
    """the gradient of ReLU(z)"""
    index = z >= 0
    result = np.zeros(z.shape)
    result[index] = 1.0
    return result
