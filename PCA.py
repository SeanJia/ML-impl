__author__ = 'Zhiwei Jia'

import numpy as np
import numpy.linalg as la
import numpy.matlib


def PCA(data, to_dim, reconstruct=False, many_feature=False):
    """input is of the form that n (# of sample) rows and k (# of feature) cols"""
    if not many_feature:
        data = data.T
    cov = data.dot(data.T)
    values, vectors = la.eig(cov)
    if many_feature:
        vectors = np.dot(data.T, vectors)
        for i in range(vectors.shape[1]):
            vectors[:, i] /= la.norm(vectors[:, i])
        data = data.T
    order = values.argsort()[::-1]
    vectors = vectors[:, order]
    v_r = vectors[:, 0:to_dim]
    data = np.dot(v_r.T, data)
    if reconstruct:
        data = np.dot(v_r, data)
    return data.T

def PCA_test(data, start, stop, step):
    """test different paramters for PCA dimensionality reduction"""
    for i in range(start, stop, step):
        x = PCA(data, i, reconstruct=True)
        approximate = np.sum(np.sum((data.T - x.T) ** 2, axis=1), axis=0)
        real = np.sum(np.sum((data.T)**2, axis=1), axis=0)
        print 'for dimension reduced to ', i, ': ', approximate / real, ' % accuracy.'






