import numpy as np
import numpy.random
from scipy.stats import multivariate_normal as norm
import scipy.stats
import numpy.matlib
from numpy.linalg import matrix_rank

class MixedGaussian:

    def __init__(self, num_classes, dim_data, low=0, high=1, init_mean=None, init_cov=None):
        self.num_classes = num_classes
        self.dim_data = dim_data
        self.pie = np.array([1.0/num_classes for i in range(num_classes)])
        if init_mean is None:
            self.miu = np.reshape(np.random.uniform(low, high, dim_data * num_classes), (num_classes, dim_data))
        else:
            self.miu = init_mean
        if init_cov is None:
            self.sigma = np.array([np.identity(dim_data) / dim_data for i in range(num_classes)])
        else:
            self.sigma = init_cov

    def train(self, data, max_epoch=30, store=None, supervised_data=False, label=None, test_data=None, test_label=None):
        """
        Train the mixed Gaussian model in (normal) unsupervised way
        data: should be T by dim_data matrix
        """
        for i in range(max_epoch):
            p_x = self.EM_update(data)
            print "At iteration", i+1, "the log-likelihood is", self.log_likelihood(p_x)
            if supervised_data and label is not None:
                train_error = self.prediction_error(data, label, p_x)
                print "The training error is", train_error, "%"
                if test_data is not None and test_label is not None:
                    test_error = self.prediction_error(test_data.test_label, p_x)
                    print "And the testing error is", test_error, "%"
            print "------------\n"
        if store is not None:
            self.store()

    def log_likelihood(self, p_x):
        pX = self.pie.dot(p_x)
        too_small = (pX == 0.0)
        pX[too_small] = 1e-300
        return np.sum(np.log(pX))

    def prediction_error(self, data, label, p_x):
        error_count = 0
        prediction = self.prediction(data, p_x) # should be of np.array
        for i in range(self.num_classes):
            cluster = label[prediction == i]
            which_class = scipy.stats.mode(cluster)[0][0]
            error_count += np.sum(cluster != which_class)
        return error_count / (data.shape[0] + 0.0)

    def prediction(self, data, p_x):
        return np.argmax(p_x, axis=0)

    def store(self, file_name):
        pass

    def EM_update(self, data):

        print "inside em update"
        # compute p(x)
        p_x = []
        for i in range(self.num_classes):
            p = norm(mean=self.miu[i, :], cov=self.sigma[i])
            p_x.append(p.pdf(data))
        p_x = np.array(p_x)
        print np.sum(np.isnan(p_x))
        pX = self.pie.dot(p_x)
        print "after compute p(x)"

        # compute posterior
        too_small = (pX == 0.0)
        pX[too_small] = 1
        print p_x[3, 2708]
        post = np.matlib.repmat(self.pie, data.shape[0], 1).T * p_x / pX

        # update
        sum_post = np.sum(post, axis=1)
        self.pie = sum_post / (data.shape[0] - np.sum(too_small))
        print self.pie
        new_sigma = []
        print "before update sigma"
        for i in range(self.num_classes):
            Z = data.T - np.matlib.repmat(np.array([self.miu[i, :]]).T, 1, data.shape[0])
            curr_post = post[i, :]
            mat_post = np.matlib.repmat(curr_post, self.dim_data, 1)
            sigma = np.dot(mat_post * Z, Z.T) / sum_post[i]
            new_sigma.append(sigma)
            print "new sigma", i, np.linalg.matrix_rank(sigma)
        self.sigma = np.array(new_sigma)
        self.miu = post.dot(data) / np.transpose(np.array([sum_post]))
        print "finish update"

        # return p_x for future use in this iteration
        return p_x
