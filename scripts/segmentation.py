import numpy as np
import matplotlib.pyplot as plt

from pyomo.environ import ConcreteModel, Var, Objective, Constraint, SolverFactory
from pyomo.environ import maximize, Binary, RangeSet, PositiveIntegers, Set, ConstraintList

import networkx as nx

def LoadImage(filename):
    A = plt.imread(filename).astype(np.float64)
    return A

def ShowImages(A, B, C):
    fig, ax = plt.subplots(1,3)


    ax[0].imshow(A, cmap='binary')
    ax[0].autoscale()
    ax[0].set_aspect('equal', 'box')
    ax[0].axis('off')

    ax[1].imshow(B, cmap='binary')
    ax[1].autoscale()
    ax[1].set_aspect('equal', 'box')
    ax[1].axis('off')

    ax[2].imshow(C, cmap='binary')
    ax[2].autoscale()
    ax[2].set_aspect('equal', 'box')
    ax[2].axis('off')

    plt.show()

def ShowImage(A):
    fig, ax = plt.subplots()

    plt.imshow(A)
    ax.autoscale()
    ax.set_aspect('equal', 'box')
    plt.axis('off')
    plt.show()


def Segmentation(G):
    n = G.number_of_nodes()
    m = G.number_of_edges()

    # Pyomo Model object
    mod = ConcreteModel()

    # Index sets
    mod.V = RangeSet(0, n-1)

    M, IM = {}, {}
    for u, p in enumerate(G.nodes()):
        M[p] = u
        IM[u] = p

    Es = [(M[i], M[j]) for i,j in G.edges()]
    print(Es)
    mod.E = Set(within=mod.V * mod.V, initialize=Es)

    # Variables
    mod.x = Var(mod.V, within=Binary)
    mod.z = Var(mod.E, within=Binary)

    # Objective function
    mod.fobj = Objective(expr=sum(d['weight']*mod.z[M[i],M[j]] for i,j,d in G.edges(data=True)))

    # Constraints
    mod.equal = ConstraintList()
    for i,j in mod.E:
        mod.equal.add( mod.x[i] - mod.x[j] <= 1 - mod.z[i,j])
        mod.equal.add( mod.x[j] - mod.x[i] <= 1 - mod.z[i,j])
        mod.equal.add( mod.x[i] + mod.x[j] + mod.z[i,j] >= 1)
        mod.equal.add( mod.x[j] + mod.x[i] <= 1 + mod.z[i,j])

    # Solve the model
    sol = SolverFactory('gurobi').solve(mod, tee=True)

    # CHECK SOLUTION STATUS

    # Get a JSON representation of the solution
    sol_json = sol.json_repn()
    # Check solution status
    if sol_json['Solver'][0]['Status'] != 'ok':
        return None
    if sol_json['Solver'][0]['Termination condition'] != 'optimal':
        return None

    sol = []
    for i, p in enumerate(G.nodes()):
        sol.append((p[0], p[1], mod.x[i]))
    return [(p, mod.x[M[p]]()) for p in G.nodes()]


def FromImage2Graph(A):
    w, h = A.shape

    G = nx.Graph()

    # Add nodes
    for i in range(w):
        for j in range(h):
            G.add_node( (i,j) )

    # Add edges
    for i in range(w):
        for j in range(h):
            for a in range(w):
                for b in range(h):
                    if i <= a and j <= b:
                        G.add_edge( (i,j), (a,b), weight=abs(A[i,j]-A[a,b]))

    return G

def Resample(A, n=16):
    w, h, _ = A.shape

    h = w//n
    h2 = h*h

    # New image to resample
    B = np.zeros( (n,n) )

    for i in range(n):
        for j in range(n):
            B[i,j] = np.sum(A[i*h:(i+1)*h, j*h:(j+1)*h]) / h2

    return B

def AddNoise(A, mu=0.2):
    n,m = A.shape

    for i in range(n):
        for j in range(m):
            A[i,j] = A[i,j] + np.random.lognormal(sigma=mu)

    return A


if __name__ == '__main__':
    A = LoadImage('../data/picture32_1007.png')


    print(A.shape)
    B = Resample(A, 10)

    B = AddNoise(B)

    G = FromImage2Graph(B)

    #ShowImages(A, B, B)
    if True:
        sol = Segmentation(G)

        n,m = B.shape
        C = np.zeros((n,m))

        for p, v in sol:
            if v != None:
                C[p] = v

        ShowImages(A, B, C)