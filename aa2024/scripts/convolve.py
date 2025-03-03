from scipy import signal
import numpy as np
import cv2

import matplotlib.pyplot as plt

def Sign(x):
    return 1 if x >= 0.001 else -1

def Parse(filename):
    fh = open(filename, 'r')

    fh.readline()

    Xs, Ys = [], []
    for row in fh:
        line = row.replace('\n','').split(';')
        v = int(line[0])
        Ys.append(-1 if v == 4 else 1)
        Xs.append(list(map(int, line[1:])))

    return np.matrix(Xs), np.array(Ys)

def DrawDigit(A):
    plt.imshow(A)
    plt.show()

def ApplyFilter(A, F):
    k, _ = F.shape
    A = A.reshape((28,28))
    n, m = A.shape
    R = np.zeros((n//2-1, m//2-1))
    for i in range(0, n-3, 2):
        for j in range(0, m-3, 2):
            #R[i//2,j//2] = Sign(np.multiply(A[i:i+k,j:j+k], F).sum())
            #R[i//2,j//2] = Sign(signal.convolve2d(A[i:i+k,j:j+k], F, mode='valid').sum())
            R[i//2,j//2] = Sign(convolve(A[i:i+k,j:j+k], F, mode='constant', cval=0.0).sum())
            #convolve(img, kernel, mode='constant', cval=0.0)
    return R

from math import sqrt
def Filter(A, F):
    N = int(sqrt(A.shape[0]))
    #A = A.reshape((N,N)).astype(np.float32)
    C = cv2.filter2D(A, ddepth=-1, kernel=F)
    return np.sign(C[1:-1:2, 1:-1:2]-0.01+1)

Xs, Ys = Parse('../data/train_nine_four.csv')
#Xs, Ys = Parse('../data/train_three_four.csv')

print('dimension of matrix X:', Xs.shape)
print('dimension of y:', Ys.shape)

F1 = np.matrix([[-1,0,0], [0,0,0], [0,0,-1]])
F2 = np.zeros((3,3)) #np.matrix([[-1,-1,-1], [0,0,0], [1,1,1]])
from time import perf_counter
time_start = perf_counter()

for _ in range(0):
    Filter(Xs[1], F1)

# record end time
time_end = perf_counter()
# calculate the duration
time_duration = time_end - time_start
# report the duration
print(f'Took {time_duration:.3f} seconds')

DrawDigit(Filter(Filter(Xs[1].reshape(28,28).astype(float), F1), F2))
