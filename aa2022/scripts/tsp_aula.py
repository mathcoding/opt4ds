# -*- coding: utf-8 -*-
"""
Created on Fri May  6 14:23:24 2022

@author: gualandi
"""

# Pyomo support 
from pyomo.environ import ConcreteModel, Var, Objective, Constraint, SolverFactory
from pyomo.environ import maximize, Binary, RangeSet, PositiveReals, ConstraintList


# Import the Numerical Python library
import numpy as np
from math import sqrt

def CostMatrix(Ls):
    n = len(Ls)
    C = 100000*np.ones((n,n))
    for i, (a,b) in enumerate(Ls):
        for j, (c,d) in enumerate(Ls[i+1:]):
            C[i, i+j+1] = sqrt((a-c)**2 + (b-d)**2)
            C[i+j+1, i] = C[i, i+j+1]
            
    return C


# Residenza Collegiali a Pavia
Rs = [(45.1882789,9.1600456, 'Del Maino'),
      (45.1961107,9.1395709, 'Golgi'),(45.1851618,9.1506323, 'Senatore'),
      (45.1806049,9.1691651, 'Don Bosco'),(45.1857651,9.1473637, 'CSA'),
      (45.1802511,9.1591663, 'Borromeo'),(45.1877192,9.1578934, 'Cairoli'),
      (45.1870975,9.1588276, 'Castiglioni'),(45.1871301,9.1435067, 'Santa Caterina'),
      (45.1863927,9.15947, 'Ghislieri'),(45.2007148,9.1325475, 'Nuovo'),
      (45.1787292,9.1635482, 'Cardano'),(45.1864928,9.1560687, 'Fraccaro'),
      (45.1989668,9.1775168, 'Griziotti'),(45.1838819,9.161318, 'Spallanzani'),
      (45.1823523,9.1454315, 'Valla'),(45.2007816,9.1341354, 'Volta'),      
      (45.2070857,9.1382623, 'Residenza Biomedica')]

# (45.2070857,9.1382623, 'Residence Campus'),(45.2070857,9.1382623, 'Green Campus'),


def TSP(C):
    # Dimension of the problem
    n, n = C.shape
    
    print('city:', n)
    
    model = ConcreteModel()

    # The set of indces for our variables
    model.I = RangeSet(1, n)   # NOTE: it is different from "range(n)"
    # the RangeSet is in [1,..n], the second is [0,n(
    
    model.J = RangeSet(1, n)
    
    # Variable definition
    model.X = Var(model.I, model.J, within=Binary)
    
    # Variables for the MTZ subtour constraints
    model.U = Var(model.I, within=PositiveReals)
    
    # Objective function
    model.obj = Objective(
        expr = sum(C[i-1,j-1] * model.X[i,j] for i,j in model.X)
    )
    
    # The constraints
#    model.Indegree = Constraint(model.I, 
#                                rule =  lambda m, i: sum(m.X[i,j] for j in m.J) == 1)

    def OutDegRule(m, i):
        return sum(m.X[i,j] for j in m.J) == 1
    model.Outdegree = Constraint(model.I, rule = OutDegRule)
    
    def InDegRule(m, j):
        return sum(m.X[i,j] for i in m.I) == 1
    model.Indegree = Constraint(model.J, rule = InDegRule)
    
    # easily for forbding "pairs" tour
    model.pairs = ConstraintList()
    for i in model.I:
        for j in model.J:
            model.pairs.add(expr=model.X[i,j] + model.X[j,i] <= 1)
    
    # MTZ constraints for forbidding subtours
    model.subtour = ConstraintList()
    for i in model.I:
        for j in model.J:
            if i > 1 and j > 1 and i != j:
                model.subtour.add(model.U[i] - model.U[j] + (n-1)*model.X[i,j] <= (n-1) - 1)
    
    # Solve the model
    sol = SolverFactory('gurobi').solve(model, tee=True)
    
    sol_json = sol.json_repn()
    #print(sol_json)
    if sol_json['Solver'][0]['Status'] != 'ok':
        print("qualcosa è andato storto")
        return None

    # Retrieve the solution: as a list of edges in the optimal tour
    return [(i-1, j-1) for i, j in model.X if model.X[i,j]() > 0.0]
    

import pylab as pl
from matplotlib import collections as mc

# Ps is a list of points (x,y)
# Ls is a list of egdes
# Values are the values of the edges
def PlotTour(Ps, Ls, values):
    lines = [[Ps[i], Ps[j]] for i,j in Ls]
    
    fig, ax = pl.subplots()

    lc = mc.LineCollection(lines, linewidths=[1.5 if x > 0.501 else 1 for x in values],
                           colors=['blue' if x > 0.501 else 'orange' for x in values])
    
    ax.add_collection(lc)
    ax.scatter([i for i,j in Ps], [j for i,j in Ps], 
                s=20, alpha=0.8, color='red')
    
    ax.autoscale()
    ax.margins(0.1)
    ax.axis('equal')
    pl.show()


# -----------------------------------------------
#   MAIN function
# -----------------------------------------------
if __name__ == "__main__":
    
    Ls = [(lat,lon) for lat, lon, _ in Rs]
    
    C = CostMatrix(Ls)
    
    Es = TSP(C)
    print(Es)
    PlotTour(Ls, Es, [1 for _ in Es])
    
    
    