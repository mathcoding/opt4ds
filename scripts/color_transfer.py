import numpy as np
import matplotlib.pyplot as plt

def LoadImage(filename):
    A = plt.imread(filename).astype(np.float64)/255
    return A

def ShowImage(A):
    fig, ax = plt.subplots()

    plt.imshow(A)
    ax.autoscale()
    ax.set_aspect('equal', 'box')
    plt.axis('off')
    plt.show()

def RandomChange(A):
    n,m,l = A.shape

    for i in range(n):
        for j in range(m):
            for h in range(l):
                if A[i,j,h] >= 0.5:
                    A[i,j,h] = 0
                else:
                    A[i,j,h] = 1
    return A

def DisplayCloud(A, samples=100):
    n, m, l = A.shape

    C = A.reshape(n*m, 3)

    s = np.random.randint(0, n*m, samples)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    plt.scatter(x=C[s,0], y=C[s,1], zs=C[s,2], s=10, c=C[s])

    plt.show()

def PointSamples(A, samples=100):
    n, m, l = A.shape
    C = A.reshape(n*m, 3)
    s = np.random.randint(0, n*m, samples)
    return C[s]

def OptimalTransport(H1, H2):
    # TODO: write an ILP model to find the optimal mapping of the color pallete H1 into H2
    # By samplying 50 pixels.
    # 1. Write an LP model
    # 2. Return the optimal mapping

    pass


if __name__ == '__main__':
    A = LoadImage('urlo.jpg')
    B = LoadImage('notte.jpg')

    DisplayCloud(A)
    DisplayCloud(B)

    H1 = PointSamples(A)
    H2 = PointSamples(B)
