# -*- coding: utf-8 -*-
"""
Created on Fri May 14 16:12:38 2021

@author: gualandi
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.distance import cdist

from pyomo.environ import ConcreteModel, Var, Objective, Constraint, SolverFactory
from pyomo.environ import RangeSet, NonNegativeReals, minimize

from time import perf_counter


np.random.seed(13)


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

    
def DisplayCloud(A, samples=100):
    n, m, h = A.shape 
    
    C = A.reshape(n*m, 3)
    
    s = np.random.randint(0, n*m, samples)
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    
    plt.scatter(x=C[s,0], y=C[s,1], zs=C[s,2], s=10, c=C[s])
    
    plt.show()
    

def PointSamples(A, samples=100):
    n, m, h = A.shape 
    
    C = A.reshape(n*m, 3)
    
    s = np.random.randint(0, n*m, samples)
    
    return C[s]
    
def D(a,b):
    return np.linalg.norm(a-b)**2


def OT(H1, H2):
    n = len(H1)
    
    mod = ConcreteModel()
    
    mod.I = RangeSet(0,n-1)
    mod.J = RangeSet(0,n-1)
    
    mod.x = Var(mod.I, mod.J, within=NonNegativeReals)
    
    mod.obj = Objective(expr=sum(D(H1[i], H2[j]) * mod.x[i,j] for i,j in mod.x))
    
    mod.A = Constraint(mod.I,
                       rule = lambda m, i: sum(m.x[i,j] for j in m.J) == 1)

    mod.B = Constraint(mod.J,
                       rule = lambda m, j: sum(m.x[i,j] for i in m.I) == 1)
    

    SolverFactory('gurobi').solve(mod, tee=True)
    
    ColMap = []

    for i in mod.I:
        for j in mod.J:
            if mod.x[i,j]() > 0.5:
                ColMap.append(j)
                
    return ColMap

# suppongo |A| > |B|
def ClosestRGB(A, B):
    return np.argmin(cdist(A, B), axis=1)
    


# -----------------------------------------------
#   MAIN function
# -----------------------------------------------
if __name__ == "__main__":
    A = LoadImage('notte.jpg')
    B = LoadImage('urlo.jpg')

    print(A.shape)
    print(B.shape)    
    
    ShowImage(B)
    DisplayCloud(A, 500)
    # DisplayCloud(A, 500)
    
    H1 = PointSamples(A, 50)
    H2 = PointSamples(B, 50)
    
    ColMap = OT(H1, H2)
    
    n,m,_ = A.shape
    C = A.reshape(n*m, 3)
    
    Y = ClosestRGB(C, H1)
    
    H4 = np.array([ H2[ColMap[i]] for i in Y] )
    H5 = H4.reshape(n,m,3)
    
    ShowImage(H5)
    
    
    
    