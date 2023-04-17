# -*- coding: utf-8 -*-
"""
@author: gualandi
"""

import numpy as np
import matplotlib.pyplot as plt


from scipy.spatial.distance import cdist

from time import perf_counter

def LoadImage(filename):
    A = plt.imread(filename).astype(np.float64) / 255
    return A

def ShowImage(A):
    fig, ax = plt.subplots()

    plt.imshow(A)
    ax.autoscale()
    ax.set_aspect('equal', 'box')
    plt.axis('off')
    plt.show()

def RandomChange(A):
    n, m, l = A.shape

    for i in range(n):
        for j in range(m):
            for h in range(l):
                if A[i, j, h] >= 0.5:
                    A[i, j, h] = 0
                else:
                    A[i, j, h] = 1
    return A

def DisplayCloud(A, samples=100):
    n, m, l = A.shape

    C = A.reshape(n * m, 3)

    s = np.random.randint(0, n * m, samples)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    plt.scatter(x=C[s, 0], y=C[s, 1], zs=C[s, 2], s=10, c=C[s])

    plt.show()


def PointSamples(A, samples=100):
    n, m, l = A.shape
    C = A.reshape(n * m, 3)
    s = np.random.randint(0, n * m, samples)
    return C[s]


def D(a, b):
    # See documentation at:
    # https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html
    return np.linalg.norm(a-b)**2

def ClosestRGB(A, B):
    # See documentation at:
    # cdist: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html
    # np.argmin: https://numpy.org/doc/stable/reference/generated/numpy.argmin.html
    return np.argmin(cdist(A, B), axis=1)


def OptimalTransport(H1, H2):
    # TODO: write an ILP model to find the optimal mapping of the color pallete H1 into H2
    # By samplying 50 pixels.
    # 1. Write an LP model
    # 2. Return the optimal mapping

    # TODO: Define a better mapping
    # ... write your LP model ....
    
    # EXAMPLE DUMMY OF MAPPING: IDENTY MAP
    return [i for i in range(len(H1))]


if __name__ == '__main__':
    A = LoadImage('urlo.jpg')
    B = LoadImage('notte.jpg')

    DisplayCloud(A)
    DisplayCloud(B)

    H1 = PointSamples(A)
    H2 = PointSamples(B)

    # Measure overall solution time
    t0 = perf_counter()
    CMAP = OptimalTransport(H1, H2)
    print("solve time: ", perf_counter() - t0)
    
    n,m,_ = A.shape
    C = A.reshape(n*m,3)

    Y = ClosestRGB(C, H1)
    H4 = np.array([H2[CMAP[i]] for i in Y])
    H5 = H4.reshape(n,m,3)

    ShowImage(H5)
    
