import numpy as np
import matplotlib.pyplot as plt
from random import sample, seed
seed(13)
np.random.seed(13)

from gurobipy import Model, GRB, quicksum, Env

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

def DisplayCloud(A, samples=100):
    n, m, l = A.shape
    print('shape', n, m, l)
    C = A.reshape(n*m, 3)
    s = sample(range(0, m*n), samples)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    plt.scatter(x=C[s, 0], y=C[s, 1], zs=C[s, 2], s=100, c=C[s])
    plt.show()

def PointSamples(A, samples=100):
    n, m, l = A.shape
    C = A.reshape(n*m, 3)
    s = sample(range(0, m*n), samples)
    return C[s]

def ColorMap(H1, H2):
    # Create empty model
    model = Model()
    # Set a time limit to 60 seconds
    model.setParam('TimeLimit', 600)

    # Select the LP algorithm
    # 0: primal simplex, 1: dual simplex, 2: barrier, 
    # 3: concurrent, 4: deterministic concurrent
    model.setParam('Method', 1) 

    # For using the barrier, UNCOMMENT the following:
    # model.setParam('Method', 2) 
    # model.setParam('Crossover', 0)

    # Add flow variables
    pi = {}
    for i in range(len(H1)):
        for j in range(len(H2)):
            pi[i, j] = model.addVar()#obj=np.linalg.norm(H1[i]-H2[j]))

    # Linear objective function
    model.setObjective(quicksum(np.linalg.norm(H1[i]-H2[j])*pi[i,j] for i,j in pi))

    # For using the barrier AND a quadratic objective function, UNCOMMENT the following:
    # model.setObjective(quicksum(np.linalg.norm(H1[i]-H2[j])*pi[i,j] for i,j in pi) + quicksum(0.01*pi[i,j]*pi[i,j] for i,j in pi))
    
    # Add flow constraints on bipartite graph
    for i in range(len(H1)):
        model.addConstr(quicksum(pi[i, j] for j in range(len(H2))) >= 1)

    for j in range(len(H2)):
        model.addConstr(quicksum(pi[i, j] for i in range(len(H1))) <= 1)

    # Solve the problem
    model.optimize()

    if model.Status == 2 or model.Status == 9:
        # Optimal solution found
        # Map node i to j if pi[i,j] == 1.0
        M = []
        for i in range(len(H1)):
            T = []
            for j in range(len(H2)):
                if pi[i,j].X > 0.01:
                    T.append( (j, pi[i,j].X) )
            M.append(T)
        return M
    else:
        # No solution found
        return None
    
# Distance of cartesian product between vectors in A and B
from scipy.spatial.distance import cdist
def ClosestRGB(A, B):
    return np.argmin(cdist(A, B), axis=1)

def Wheel(Ps):
    if len(Ps) == 1:
        return Ps[0][0]
    # If more than a single element, return with probabilities
    As = [p[0] for p in Ps]
    Pr = [p[1] for p in Ps]
    Pr[0] = Pr[0] + 1.0-sum(Pr)
    Ws = np.random.choice(As, size=1, p=Pr)
    return Ws[0]


B = LoadImage('../data/notte.jpg')
A = LoadImage('../data/borgo.jpg')

H1 = PointSamples(A, 500)
H2 = PointSamples(B, 500)

CMAP = ColorMap(H1, H2)

n, m, _ = A.shape
C = A.reshape(n*m, 3)

Y = ClosestRGB(C, H1)
H4 = np.array([H2[Wheel(CMAP[i])] for i in Y])
H5 = H4.reshape(n, m, 3)

ShowImage(H5)

# NOTE:
# For an alternative demo on optimal color transfer, see the following link
# from the Python Optimal Transport library:
# https://pythonot.github.io/auto_examples/domain-adaptation/plot_otda_mapping_colors_images.html