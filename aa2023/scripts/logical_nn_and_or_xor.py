# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 14:23:32 2021

@author: gualandi
"""

import numpy as np
import matplotlib.pyplot as plt

from pyomo.environ import ConcreteModel, Var, Objective, Constraint, SolverFactory
from pyomo.environ import maximize, minimize, Binary, RangeSet, ConstraintList, NonNegativeReals, Reals, Integers

def LogicalNN(Xs, Ys):
    # Main Pyomo model
    model = ConcreteModel()

    # Parameters
    m = len(Xs[0])
    model.I = RangeSet(0, m-1)

    n = len(Xs)
    model.K = RangeSet(0, n-1)

    # Variables:

    # Connection weights
    model.W = Var(model.I, within=Integers, bounds=(-1,1)) 
    #model.W = Var(model.I, within=Reals) 

    # Violation of the constraints
    model.s = Var(model.K, within=Binary)

    # norm-1 in the objective function
    model.alpha = Var(model.K, within=NonNegativeReals) 
    
    # Objective Function: minimize classification error
    model.obj = Objective(expr=sum(model.alpha[k] for k in model.K))# - 0.01*sum(model.gamma[k] for k in model.K))
    
    # Classification constraints
    M = 1000
    def ConstrViolation(m, k):
        if Ys[k] == 1:
            return sum(Xs[k][i]*m.W[i] for i in model.I) >= 0.1 - M*model.s[k]
        return sum(Xs[k][i]*m.W[i] for i in model.I) <= -0.1 + M*(1-model.s[k])
        
    model.Margin = Constraint(model.K, rule = ConstrViolation)
    
    # Norm constraint
    model.Norm1 = ConstraintList()
    for k in model.K:
        model.Norm1.add( (1 - 2*model.s[k]) - Ys[k] <= model.alpha[k] )
        model.Norm1.add( Ys[k] - (1 - 2*model.s[k]) <= model.alpha[k] )

    # Solve the model
    sol = SolverFactory('gurobi').solve(model, tee=True)

    # Get a JSON representation of the solution
    sol_json = sol.json_repn()
    # Check solution status
    if sol_json['Solver'][0]['Status'] != 'ok':
        return None, None
    if sol_json['Solver'][0]['Termination condition'] != 'optimal':
        return None, None

    print([model.s[k]() for k in model.K])
    return [model.W[i]() for i in model.I]


def MLP(Xs, Ys, nh):
    # Main Pyomo model
    model = ConcreteModel()

    # Parameters
    m = len(Xs[0])
    model.I = RangeSet(0, m-1)

    n = len(Xs)
    model.K = RangeSet(0, n-1)

    model.H = RangeSet(0, nh-1)

    # Variable: parameters of f_W,U
    model.W = Var(model.I, model.H, within=Reals)
    model.U = Var(model.H, within=Reals)

    model.s = Var(model.K, model.H, within=Binary)
    model.z = Var(model.K, model.H, within=Reals)

    model.y_hat = Var(model.K, within=Binary)
    model.alpha = Var(model.K, within=NonNegativeReals)

    model.obj = Objective(expr=sum(model.alpha[k] for k in model.K))

    M = 1000
    model.Margin = ConstraintList()
    for k in model.K:
        for j in model.H:
            model.Margin.add( sum(Xs[k][i]*model.W[i,j] for i in model.I) >= 0.0 - M*model.s[k,j] )
            model.Margin.add( sum(Xs[k][i]*model.W[i,j] for i in model.I) <= -0.1 + M*(1-model.s[k,j]) )
        
    # Linearization constraints
    model.Linearize = ConstraintList()
    for k in model.K:
        for j in model.H:
            model.Linearize.add( model.z[k,j] >= -M*model.s[k,j] )
            model.Linearize.add( model.z[k,j] <= +M*model.s[k,j] )
            model.Linearize.add( model.z[k,j] >= model.U[j] - M*(1 - model.s[k,j]) )
            model.Linearize.add( model.z[k,j] <= model.U[j] + M*(1 - model.s[k,j]) )
    
    model.Neurons = ConstraintList()
    for k in model.K:
        model.Neurons.add( sum((model.U[j] - 2*model.z[k,j]) for j in model.H) >= +0.0 -M*model.y_hat[k])
        model.Neurons.add( sum((model.U[j] - 2*model.z[k,j]) for j in model.H) <= -0.1 +M*(1-model.y_hat[k]))

#        model.Neurons.add( sum((1 - 2*model.U[j] - 2*model.s[k,j] + 4*model.z[k,j]) for j in model.H) >= 0.1 -M*model.y_hat[k])
#        model.Neurons.add( sum((1 - 2*model.U[j] - 2*model.s[k,j] + 4*model.z[k,j]) for j in model.H) <= -0.1 +M*(1-model.y_hat[k]))

    # Norm constraint
    model.Norm1 = ConstraintList()
    for k in model.K:
        model.Norm1.add( (1 - 2*model.y_hat[k]) - Ys[k] <= model.alpha[k] )
        model.Norm1.add( Ys[k] - (1 - 2*model.y_hat[k]) <= model.alpha[k] )

    # Variables:
    # Solve the model
    sol = SolverFactory('gurobi').solve(model, tee=True)

    # Get a JSON representation of the solution
    sol_json = sol.json_repn()
    # Check solution status
    if sol_json['Solver'][0]['Status'] != 'ok':
        return None, None
    if sol_json['Solver'][0]['Termination condition'] != 'optimal':
        return None, None

    W = np.ones( (m, nh) )
    U = np.ones( nh )
    for i in model.I:
        for j in model.H:
            W[i,j] = model.W[i,j]()
    for j in model.H:
        U[j] = model.U[j]()

    return W, U


def BNN(Xs, Ys, nh):
    # Main Pyomo model
    model = ConcreteModel()

    # Variables:

    # Dimension of input
    m = len(Xs[0])
    model.I = RangeSet(0, m-1)

    # Number of samples
    n = len(Xs)
    model.K = RangeSet(0, n-1)

    # Neurons in hidden state
    model.H = RangeSet(0, nh-1)
    
    # Connection weights
    model.W = Var(model.I, model.H, within=Integers, bounds=(-1,1)) 
    model.U = Var(model.H, within=Integers, bounds=(-1,1))
    model.Ub = Var(within=Integers, bounds=(-1,1))

    model.Wabs = Var(model.I, model.H, within=Reals) 
    model.Uabs = Var(model.H, within=Reals)

    # Hidden states
    model.s = Var(model.K, model.H, within=Binary)

    model.z = Var(model.K, model.H, within=Binary)

    # Final output
    model.y_hat = Var(model.K, within=Binary)

    # norm-1 in the objective function
    model.alpha = Var(model.K, within=NonNegativeReals) 
    
    # Objective Function: minimize classification error
    model.obj = Objective(expr=10*sum(model.alpha[k] for k in model.K) )#\
       # + sum(model.Uabs[h] for h in model.H) \
       # + sum(model.Wabs[i,h] for i,h in model.Wabs)    )

#    def Bias(m, i, h):
#        if h == 0:
#            return m.W[i,h] == 0.0
#        return Constraint.Skip
#    model.FixBias = Constraint(model.I, model.H, rule = Bias)

    # Classification constraints
    M = 1000
    def ConstrViolation(m, k, h):
        if Ys[k] == 1:
            return sum(Xs[k][i]*m.W[i, h] for i in model.I) >= 0.0 - M*model.s[k,h]
        return sum(Xs[k][i]*m.W[i, h] for i in model.I) <= -0.1 + M*(1-model.s[k,h])

    model.Margin = Constraint(model.K, model.H, rule = ConstrViolation)

    # Linearization constraint for hidden state
    model.Linearize = ConstraintList()
    for k in model.K:
        for h in model.H:
            model.Linearize.add( model.z[k,h] <= model.s[k,h] )
            model.Linearize.add( model.z[k,h] <= model.U[h]   )
            model.Linearize.add( model.z[k,h] + 1 >= model.U[h] + model.s[k,h] )

    model.Neurons = ConstraintList()
    for k in model.K:
        if Ys[k] == 1:
            model.Neurons.add( model.Ub + sum((model.U[h] - 2*model.z[k,h]) for h in model.H) >= 0.0 - M*model.y_hat[k] )
        else:
            model.Neurons.add( model.Ub + sum((model.U[h] - 2*model.z[k,h]) for h in model.H) <= -0.1 + M*(1 - model.y_hat[k]) )

    # Norm constraint
    model.Norm1 = ConstraintList()
    for k in model.K:
        model.Norm1.add( (1 - 2*model.y_hat[k]) - Ys[k] <= model.alpha[k] )
        model.Norm1.add( Ys[k] - (1 - 2*model.y_hat[k]) <= model.alpha[k] )

    model.AbsH = ConstraintList()
    for h in model.H:
        model.AbsH.add( +model.U[h] <= model.Uabs[h] )
        model.AbsH.add( -model.U[h] <= model.Uabs[h] )

    model.AbsW = ConstraintList()
    for i in model.I:
        for h in model.H:
            model.AbsW.add( +model.W[i,h] <= model.Wabs[i,h] )
            model.AbsW.add( -model.W[i,h] <= model.Wabs[i,h] )

    # Solve the model
    sol = SolverFactory('gurobi').solve(model, tee=True)

    # Get a JSON representation of the solution
    sol_json = sol.json_repn()
    # Check solution status
    if sol_json['Solver'][0]['Status'] != 'ok':
        return None, None
    if sol_json['Solver'][0]['Termination condition'] != 'optimal':
        return None, None

    W = np.zeros( (m, nh) )
    U = np.zeros(nh+1)
    for i in model.I:
        for h in model.H:
            W[i,h] = model.W[i,h]()
    for h in model.H:
        U[h] = model.U[h]()
    U[-1] = model.Ub()
    return W, U

from numpy.random import normal
def AddNoise(X, mu=0.1):
    return list(map(lambda x: x+normal(0, mu), X))

def Sign(x):
    return 1 if x >= 0 else -1

def AccuracyPerceptron(Xs, Ys, w):
    m = len(Xs[0])
    acc, tot = 0, 0
    for j in range(100):
        for x, y in zip(Xs, Ys):
            x = AddNoise(x, 0.0)
            tot += 1
            acc += 1 if y == Sign(sum(x[i]*w[i] for i in range(m))) else 0

    return acc/tot

def AccuracyMLP(Xs, Ys, W, U):
    acc, tot = 0, 0
    for x, y in zip(Xs, Ys):
        tot += 1
        y_hat = np.sign( U[0] + np.matmul(U[:-1], np.sign(np.matmul(x, W))))
        print(y, y_hat)
        acc += 1 if y == y_hat else 0

    return acc/tot

#-----------------------------------------------
# MAIN function
#-----------------------------------------------
if __name__ == "__main__":
    from numpy.random import seed
    #seed(13)

    # AND function
    Xand = [(1,-1,-1), (1,-1, 1), (1,1,-1), (1,1,1)]
    Yand = [-1, -1, -1, 1]

    # OR function
    Xor = [(1,-1,-1), (1,-1, 1), (1,1,-1), (1,1,1)]
    Yor = [-1, 1, 1, 1]

    # XOR function
    Xxor = [(1,-1,-1), (1,-1, 1), (1,1,-1), (1,1,1)]
    Yxor = [-1, 1, 1, -1]

    # Select function
    Xs = Xxor
    Ys = Yxor

    # Add noise to the input
    #Xs = 1*[AddNoise(X, 0.0) for X in Xs]
    #Ys = 1*Ys
    
#    W = LogicalNN(Xs, Ys)
#    acc = AccuracyPerceptron(Xs, Ys, W)
    
    W, U = MLP(Xs, Ys, nh=2)
    print('matrices')
    print('W', W, U)
    acc = AccuracyMLP(Xs, Ys, W, U)
    print('Accuracy:', round(acc, 3))