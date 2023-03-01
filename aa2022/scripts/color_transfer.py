# -*- coding: utf-8 -*-
"""
Created on Thu May 13 14:52:27 2021

@author: gualandi
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.distance import cdist

from pyomo.environ import ConcreteModel, Var, Objective, Constraint, SolverFactory
from pyomo.environ import RangeSet, NonNegativeReals, minimize

from time import perf_counter

from gurobipy import Model, GRB, quicksum


# Set a seed for random sampling
np.random.seed(13)

def LoadImage(filename):
    # Read and normalize image to [0,1]
    A = plt.imread(filename).astype(np.float64) / 255.0
    return A


def ShowImage(A):
    fig, ax = plt.subplots()
    plt.imshow(A)
    ax.autoscale()
    ax.margins(0.1)
    ax.set_aspect('equal', 'box')
    plt.axis('off')
    plt.show()
    

def DisplayCloud(A, samples=1000):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    
    n,m,_ = A.shape
    C = A.reshape(n*m,3)
    s = np.random.randint(0,n*m, samples)
    plt.scatter(x=C[s,0], y=C[s,1], zs=C[s,2], s=10.0, c=C[s] )
    plt.savefig('cloud1.pdf', bbox_inches = 'tight', pad_inches = 0)
    # plt.show()
    
    
def PointSamples(A, samples=100):
    n,m,_ = A.shape
    C = A.reshape(n*m,3)
    s = np.random.randint(0,n*m, samples)
    return C[s]
    

def D(a,b):
    return np.linalg.norm(a-b)**2


def OptimalTransport(H1, H2):
    n = len(H1)
    m = len(H2)
    
    mod = ConcreteModel()
    # Parameters
    mod.I = RangeSet(0,n-1)
    mod.J = RangeSet(0,m-1)
    # Variables
    mod.pi = Var(mod.I, mod.J, within=NonNegativeReals) 
    # Objective Function
    mod.obj = Objective(expr=sum(D(H1[i], H2[j])*mod.pi[i,j] for i,j in mod.pi))
    # Constraints on the supports
    mod.X = Constraint(mod.I, 
                          rule = lambda m, i: sum(m.pi[i,j] for j in m.J) == 1)
    mod.Y = Constraint(mod.J, 
                          rule = lambda m, j: sum(m.pi[i,j] for i in m.I) == 1)
        
    # Solve the model
    SolverFactory('gurobi').solve(mod, tee=True)
    
    ColMap = []
    for i in mod.I:
        for j in mod.J:
            if mod.pi[i,j]() > 0.5:
                ColMap.append(j)
    
    return ColMap


def OptimalTransportGurobi(H1, H2):
    n = len(H1)
    m = len(H2)
    
    mod = Model()
    # Parameters
    I = list(range(n))
    J = (range(m))
    # Variables
    x = {}
    for i in I:
        for j in J:
            x[i,j] = mod.addVar(obj=D(H1[i], H2[j]))

    # Constraints on the supports
    for i in I:
        mod.addConstr(quicksum(x[i,j] for j in J) == 1)

    for j in J:
        mod.addConstr(quicksum(x[i,j] for i in I) == 1)
        
    
    # Train: Solve the model
    mod.optimize()

    ColMap = []
    for i in I:
        for j in J:
            if x[i,j].X > 0.5:
                ColMap.append(j)
    
    return ColMap

    
# Distanza tra i punti di A e quelli di B, con |A| > |B|
# Restituisce l'indice del vettore più vicino.
def ClosestRGB(A, B):
    return np.argmin(cdist(A, B), axis=1)

    
# -----------------------------------------------
#   MAIN function
# -----------------------------------------------
if __name__ == "__main__":
    B = LoadImage('notte.jpg')
    A = LoadImage('orchidea.jpg')
    
    print(A.shape)
    print(B.shape)
    
    # ShowImage(A)
    DisplayCloud(A, 500)
    # DisplayCloud(B, 500)
    
    H1 = PointSamples(A, 200)
    H2 = PointSamples(B, 200)
    
    t0 = perf_counter()
    
    CMAP = OptimalTransportGurobi(H1, H2)
    
    print("solve time: ", perf_counter() - t0)
    
    n,m,_ = A.shape
    C = A.reshape(n*m,3)

    Y = ClosestRGB(C, H1)
    H4 = np.array([H2[CMAP[i]] for i in Y])
    H5 = H4.reshape(n,m,3)
    ShowImage(B)
    ShowImage(A)
    ShowImage(H5)
    
    