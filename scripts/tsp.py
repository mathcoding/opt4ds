# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 09:53:57 2020

@author: Gualandi
"""
import numpy as np
from math import sqrt
import time


from pyomo.environ import ConcreteModel, Var, Objective, Constraint, SolverFactory
from pyomo.environ import maximize, Binary, RangeSet, PositiveReals, ConstraintList

# Residenza Collegiali a Pavia
Rs = [(45.1882789,9.1600456, 'Del Maino'),(45.2070857,9.1382623, 'Green Campus'),
      (45.1961107,9.1395709, 'Golgi'),(45.1851618,9.1506323, 'Senatore'),
      (45.1806049,9.1691651, 'Don Bosco'),(45.1857651,9.1473637, 'CSA'),
      (45.1802511,9.1591663, 'Borromeo'),(45.1877192,9.1578934, 'Cairoli'),
      (45.1870975,9.1588276, 'Castiglioni'),(45.1871301,9.1435067, 'Santa Caterina'),
      (45.1863927,9.15947, 'Ghislieri'),(45.2007148,9.1325475, 'Nuovo'),
      (45.1787292,9.1635482, 'Cardano'),(45.1864928,9.1560687, 'Fraccaro'),
      (45.1989668,9.1775168, 'Griziotti'),(45.1838819,9.161318, 'Spallanzani'),
      (45.1823523,9.1454315, 'Valla'),(45.2007816,9.1341354, 'Volta'),
      (45.2070857,9.1382623, 'Residence Campus'),(45.2070857,9.1382623, 'Residenza Biomedica')]
 
# INSTANCES TAKE FROM THE TSPLIB:
#   http://elib.zib.de/pub/mp-testdata/tsp/tsplib/tsp/

ULYSSES = [(38.24, 20.42), (39.57, 26.15), (40.56, 25.32), (36.26, 23.12),
           (33.48, 10.54), (37.56, 12.19), (38.42, 13.11), (37.52, 20.44),
           (41.23, 9.10), (41.17, 13.05), (36.08, -5.21), (38.47, 15.13), 
           (38.15, 15.35), (37.51, 15.17), (35.49, 14.32), (39.36, 19.56)]
     
BAVIERA = [(1150.0,  1760.0), (630.0,  1660.0),  (40.0,  2090.0),    (750.0,  1100.0), 
  (1030.0,  2070.0), (1650.0,   650.0), (1490.0,  1630.0),  (790.0,  2260.0),
  (710.0,  1310.0),  (840.0,   550.0),  (1170.0,  2300.0),  (970.0,  1340.0),
  (510.0,   700.0),  (750.0,   900.0),  (1280.0,  1200.0),  (230.0,   590.0),
  (460.0,   860.0),  (1040.0,   950.0), (590.0,  1390.0),   (830.0,  1770.0),
  (490.0,   500.0),  (1840.0,  1240.0), (1260.0,  1500.0),  (1280.0,  790.0),
  (490.0,  2130.0),  (1460.0,  1420.0), (1260.0,  1910.0),  (360.0,  1980.0),
  (750.0,  2030.0)]   

    
# Mixed Integer Programming Formulation
def TSP(C):
    # Number of places
    n, n = C.shape
    # Create concrete model
    model = ConcreteModel()
    
    # Set of indices
    model.I = RangeSet(1, n)
    model.J = RangeSet(1, n)
    
    # Variables
    model.X = Var(model.I, model.J, within=Binary) 

    model.U = Var(model.I, within=PositiveReals)
    
    # Objective Function
    model.obj = Objective(
        expr=sum(C[i-1,j-1] * model.X[i,j] for i,j in model.X))
    
    # Constraints on the marginals
    model.InDegree = Constraint(model.I, 
                                rule = lambda m, i: sum(m.X[i,j] for j in m.J) == 1)

    model.OutDegree = Constraint(model.J, 
                          rule = lambda m, j: sum(m.X[i,j] for i in m.I) == 1)

    model.onedir = ConstraintList()
    for i in model.I:
         for j in model.J:
             model.onedir.add(expr=model.X[i,j]+model.X[j,i] <= 1)
             
    # # Subtour
    model.subtour = ConstraintList()
    for i in model.I:
        for j in model.J:
            if i > 1 and j > 1  and i != j:
                model.subtour.add( model.U[i] - model.U[j] + n*model.X[i,j] <= n-1)

    model.fix = Constraint(expr=model.U[1] == 1)

    # Solve the model
    sol = SolverFactory('gurobi').solve(model, tee=True)
    
    # CHECK SOLUTION STATUS
    
    # Get a JSON representation of the solution
    sol_json = sol.json_repn()
    # Check solution status
    if sol_json['Solver'][0]['Status'] != 'ok':
        return None
    if sol_json['Solver'][0]['Termination condition'] != 'optimal':
        return None
    
    return [(i-1,j-1) for i,j in model.X if model.X[i,j]() > 0.5]
    
def Subtour(Es):
    unvisited = list(range(N))
    cycle = range(N+1)  
    while unvisited:  
        thiscycle = []
        neighbors = unvisited
        while neighbors:
            current = neighbors[0]
            thiscycle.append(current)
            unvisited.remove(current)
            neighbors = [j for i, j in Es if i == current and j in unvisited]
        if len(cycle) > len(thiscycle):
            cycle = thiscycle
    return cycle


def CuttingPlane(C, Subtours):
    # Number of places
    n, n = C.shape
    # Create concrete model
    model = ConcreteModel()
    
    # Set of indices
    model.I = RangeSet(1, n)
    model.J = RangeSet(1, n)
    
    # Variables
    model.X = Var(model.I, model.J, within=Binary) 
    
    # Objective Function
    model.obj = Objective(
        expr=sum(C[i-1,j-1] * model.X[i,j] for i,j in model.X))
    
    # Constraints on the marginals
    model.InDegree = Constraint(model.I, 
                                rule = lambda m, i: sum(m.X[i,j] for j in m.J) == 1)

    model.OutDegree = Constraint(model.J, 
                          rule = lambda m, j: sum(m.X[i,j] for i in m.I) == 1)

    model.subtours = ConstraintList()
    for S in Subtours:
        P = [(i+1,j+1) for i in S for j in S if i != j]
        model.subtours.add(expr=sum(model.X[i,j] for i, j in P) <= len(S)-1)

    #opt.set_instance(model)
    solver = SolverFactory('gurobi')
    sol = solver.solve(model, tee=False)

    # CHECK SOLUTION STATUS
    # Get a JSON representation of the solution
    sol_json = sol.json_repn()
    # Check solution status
    if sol_json['Solver'][0]['Status'] != 'ok':
        return None
    if sol_json['Solver'][0]['Termination condition'] != 'optimal':
        return None
    
    return [(i-1,j-1) for i,j in model.X if model.X[i,j]() > 0.5]
        

def TSP2(C):
    global Ls
    S = []
    for i in range(N):
        for j in range(N):
            if i < j:
                S.append(set([i,j]))

    while True:
        Es = CuttingPlane(C, S)
        
        Cycle = Subtour(Es)
        if len(Cycle) == N: return Es
        
        PlotTour(Ls, Es)
        # time.sleep(3)
        
        S.append(set(Cycle))
        


def PlotTour(Ps, Ls):
    # Report solution value
    import matplotlib.pyplot as plt
    import numpy as np
    import pylab as pl
    from matplotlib import collections  as mc

    lines = [[Ps[i], Ps[j]] for i,j in Ls]

    lc = mc.LineCollection(lines, linewidths=2)
    fig, ax = pl.subplots()
    ax.add_collection(lc)
    ax.autoscale()
    ax.margins(0.1)
    plt.plot([p[0] for p in Ps], [p[1] for p in Ps], 
             'ro', color='red')
    plt.show()


def CostMatrix(Ls):
    n = len(Ls)
    C = 100000*np.ones((n,n))
    for i, (a,b) in enumerate(Ls):
        for j, (c,d) in enumerate(Ls[i+1:]):
            C[i, i+j+1] = sqrt((a-c)**2 + (b-d)**2)
            C[i+j+1, i] = C[i, i+j+1]
            
    return C
 
    
def RandomTSP(n):
    from numpy import random
    return [(x,y) for x,y in zip(random.random(n), random.random(n))]


# -----------------------------------------------
#   MAIN function
# -----------------------------------------------
if __name__ == "__main__":

    Test = 3
    N = 100
    
    # Compute Cost Matrix
    if Test == 0:
        Ls = [(a,b) for a,b,_ in Rs]
    if Test == 1:
        Ls = ULYSSES
    if Test == 2:
        Ls = BAVIERA
    if Test == 3:
        np.random.seed(13)
        Ls = RandomTSP(N)
        
    N = len(Ls)
    C = CostMatrix(Ls)
        
    # Solve problem
    tour = TSP2(C)
    
    PlotTour(Ls, tour)