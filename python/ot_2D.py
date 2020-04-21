# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 21:19:45 2020

@author: Gualandi
"""
import numpy as np

from pyomo.environ import ConcreteModel, Var, Objective, Constraint, SolverFactory
from pyomo.environ import maximize, minimize, Binary, RangeSet, ConstraintList, NonNegativeReals


Normalize = lambda x: x/sum(x)

Cost = lambda x, y: (x[0] - y[0])**2 + (x[1] - y[1])**2


def OT_LP(Mu, Nu, Xm, Xn):
    # Main Pyomo model
    model = ConcreteModel()
    # Parameters
    model.I = RangeSet(len(Mu))
    model.J = RangeSet(len(Nu))
    # Variables
    model.PI = Var(model.I, model.J, within=NonNegativeReals) 
    # Objective Function
    model.obj = Objective(
        expr=sum(model.PI[i,j]*Cost(Xm[i-1], Xn[j-1]) for i,j in model.PI))
    # Constraints on the marginals
    model.Mu = Constraint(model.I, 
                          rule = lambda m, i: sum(m.PI[i,j] for j in m.J) == Mu[i-1])
    model.Nu = Constraint(model.J, 
                          rule = lambda m, j: sum(m.PI[i,j] for i in m.I) == Nu[j-1])
    
    # Solve the model
    sol = SolverFactory('glpk').solve(model)

    # Get a JSON representation of the solution
    sol_json = sol.json_repn()
    # Check solution status
    if sol_json['Solver'][0]['Status'] != 'ok':
        return None
    if sol_json['Solver'][0]['Termination condition'] != 'optimal':
        return None

    return model.obj(), dict([((i-1,j-1), model.PI[i,j]()) 
                              for i,j in model.PI if model.PI[i,j]() > 0])


def PlotOT(Mu, Xm, Nu, Xn, Plan):
    import matplotlib.pyplot as plt
    from matplotlib import collections  as mc
    
    sol = [[Xm[i], Xn[j]] for i,j in Plan]
    edge = [[xm, xn] for xm in Xm for xn in Xn]

    lc_edge = mc.LineCollection(edge, linewidths=0.2)    
    lc_sol = mc.LineCollection(sol, linewidths=1.2)

    fig, ax = plt.subplots()
    
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlim(-0.1,1.1)
    plt.ylim(-0.1,1.1)
    
    ax.scatter([i for i,_ in Xm], [j for _,j in Xm],
               s=[u*500 for u in Mu],
               c='blue', alpha=0.5)
    
    ax.scatter([i for i,_ in Xn], [j for _,j in Xn], 
               s=[u*500 for u in Nu],
               c='red', alpha=0.5)

    ax.add_collection(lc_edge)
    ax.add_collection(lc_sol)
    
    fig.tight_layout()


    plt.show()

# -----------------------------------------------
#   MAIN function
# -----------------------------------------------
if __name__ == "__main__":
    
    np.random.seed(13) # 17
    m = 7
    n = 9
    Mu = Normalize(np.random.chisquare(1, m))
    Nu = Normalize(np.random.chisquare(1, n))
    
    Xm = [(i,j) for i,j in zip(np.random.uniform(size=m), 
                               np.random.uniform(size=m))]
    
    Xn = [(i,j) for i,j in zip(np.random.uniform(size=n), 
                               np.random.uniform(size=n))]
    
    
    z, x = OT_LP(Mu, Nu, Xm, Xn)
    print(z)

    PlotOT(Mu, Xm, Nu, Xn, x)
    
    