# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 14:23:32 2021

@author: gualandi
"""

import numpy as np
import matplotlib.pyplot as plt

from pyomo.environ import ConcreteModel, Var, Objective, Constraint, SolverFactory
from pyomo.environ import maximize, minimize, Binary, RangeSet, ConstraintList, NonNegativeReals, Reals, Integers

def IdentityNN(Xs, l):
    # Main Pyomo model
    model = ConcreteModel()

    n,m = Xs.shape
    # Parameters
    model.I = RangeSet(0, m-1)

    model.K = RangeSet(0, n-1)

    model.J = RangeSet(0, l-1)

    # Variables:

    # Connection weights
#    model.W = Var(model.I, model.H, within=Integers, bounds=(-1,1)) 
#    model.U = Var(model.H, model.I, within=Integers, bounds=(-1,1)) 
    model.W = Var(model.I, model.J, within=Reals, bounds=(-1,1)) 
    model.U = Var(model.J, model.I, within=Reals, bounds=(-1,1)) 

    # Violation of the constraints
    model.s = Var(model.K, model.J, within=Binary)
    model.t = Var(model.K, model.I, within=Binary)

    model.v = Var(model.J, model.I, model.K, within=Reals)

    # norm-1 in the objective function
    model.alpha = Var(model.K, model.I, within=NonNegativeReals) 
    
    # Objective Function: minimize classification error
    model.obj = Objective(expr=sum(model.alpha[k,i] for k in model.K for i in model.I))

    # Classification constraints
    M = 1000
    model.InnerLayer = ConstraintList()
    for k in model.K:
        for j in model.J:
            model.InnerLayer.add( sum(Xs[k,i]*model.W[i,j] for i in model.I) >= 0.0 - M*(1-model.s[k,j]) )
            model.InnerLayer.add( sum(Xs[k,i]*model.W[i,j] for i in model.I) <= -0.1 + M*model.s[k,j] ) 
            
    # Norm constraint
    model.Linearize = ConstraintList()
    for i in model.I:
        for k in model.K:
            for j in model.J:
                model.Linearize.add( model.v[j,i,k] >= -M*model.s[k,j] )
                model.Linearize.add( model.v[j,i,k] <= +M*model.s[k,j] )
                model.Linearize.add( model.v[j,i,k] >= model.U[j,i] - M*(1 - model.s[k,j]) )
                model.Linearize.add( model.v[j,i,k] <= model.U[j,i] + M*(1 - model.s[k,j]) )

    model.OutputLayer = ConstraintList()
    for i in model.I:
        for k in model.K:
            model.OutputLayer.add( sum((2*model.v[j,i,k] - model.U[j,i]) for j in model.J) >= +0.0 - M*(1-model.t[k,i]) )
            model.OutputLayer.add( sum((2*model.v[j,i,k] - model.U[j,i]) for j in model.J) <= -0.1 + M*model.t[k,i] )

    model.Yhat = ConstraintList()
    for i in model.I:
        for k in model.K:
            model.Yhat.add( +(2*model.t[k,i]-1) - Xs[k,i] <= model.alpha[k,i] )
            model.Yhat.add( -(2*model.t[k,i]-1) + Xs[k,i] <= model.alpha[k,i] )
            
    # Solve the model
    sol = SolverFactory('gurobi').solve(model, tee=True)

    # Get a JSON representation of the solution
    sol_json = sol.json_repn()
    # Check solution status
    if sol_json['Solver'][0]['Status'] != 'ok':
        return None, None
    if sol_json['Solver'][0]['Termination condition'] != 'optimal':
        return None, None

    return [model.W[i,j]() for i in model.I for j in model.J]


#-----------------------------------------------
# MAIN function
#-----------------------------------------------
if __name__ == "__main__":
    from numpy.random import seed
    #seed(13)

    # AND function
    X = np.matrix([[+1,-1,-1,-1], [-1,+1,-1, -1], [-1,-1,+1,-1], [-1,-1,-1,+1]])
#    X = np.matrix([[+1,0,0,0], [0,+1,0,0], [0,0,+1,0], [0,0,0,+1]])
    print(X.shape)

    W = IdentityNN(X, 2)

    print(W)