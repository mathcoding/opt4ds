# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 14:23:32 2021

@author: gualandi
"""

import numpy as np
import matplotlib.pyplot as plt

from pyomo.environ import ConcreteModel, Var, Objective, Constraint, SolverFactory
from pyomo.environ import maximize, minimize, Binary, RangeSet, ConstraintList, NonNegativeReals, Reals, Integers


def Parse(filename):
    fh = open(filename, 'r')

    fh.readline()

    Xs, Ys = [], []
    for row in fh:
        line = row.replace('\n','').split(';')
        Ys.append(int(line[0]))
        Xs.append(list(map(int, line[1:])))

    return np.matrix(Xs), np.array(Ys)


def DrawDigit(A):
    plt.imshow(A.reshape((28,28)), cmap='binary')
    plt.show()


def BNN(Xs, Ys, nh):
    # Main Pyomo model
    model = ConcreteModel()

    # Variables:
    n, m = Xs.shape

    # Dimension of input
    model.I = RangeSet(0, m-1)

    # Number of samples
    model.K = RangeSet(0, n-1)

    # Neurons in hidden state
    model.H = RangeSet(0, nh-1)
    
    # Connection weights
    model.W = Var(model.I, model.H, within=Integers, bounds=(-1,1)) 
    model.U = Var(model.H, within=Integers, bounds=(-1,1))

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
    model.obj = Objective(expr=sum(model.alpha[k] for k in model.K) )
        #+ sum(model.Uabs[h] for h in model.H) \
        #+ sum(model.Wabs[i,h] for i,h in model.Wabs)   )

#    def Bias(m, i, h):
#        if h == 0:
#            return m.W[i,h] == 0.0
#        return Constraint.Skip
#    model.FixBias = Constraint(model.I, model.H, rule = Bias)

    # Classification constraints
    M1 = [0.1+np.sum(Xs[k, :]) for k in range(n)]
    def ConstrViolation(m, k, h):
        if Ys[k] == 1:
            return sum(Xs[k, i]*m.W[i, h] for i in model.I) >= 0.1 - M1[k]*model.s[k,h]
        return sum(Xs[k, i]*m.W[i, h] for i in model.I) <= -0.1 + M1[k]*(1-model.s[k,h])

    model.Margin = Constraint(model.K, model.H, rule = ConstrViolation)

    # Linearization constraint for hidden state
    model.Linearize = ConstraintList()
    for k in model.K:
        for h in model.H:
            model.Linearize.add( model.z[k,h] <= model.s[k,h] )
            model.Linearize.add( model.z[k,h] <= model.U[h]   )
            model.Linearize.add( model.z[k,h] + 1 >= model.U[h] + model.s[k,h] )

    model.Neurons = ConstraintList()
    M = nh+1
    for k in model.K:
        if Ys[k] == 1:
            model.Neurons.add( sum((model.U[h] - 2*model.z[k,h]) for h in model.H) >= 0.1 - M*model.y_hat[k] )
        model.Neurons.add( sum((model.U[h] - 2*model.z[k,h]) for h in model.H) <= -0.1 + M*(1 - model.y_hat[k]) )

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
    sol = SolverFactory('gurobi').solve(model, tee=True, options={'TimeLimit': 30})

    # Get a JSON representation of the solution
    sol_json = sol.json_repn()
    # Check solution status
#    if sol_json['Solver'][0]['Status'] != 'ok':
#        return None, None
#    if sol_json['Solver'][0]['Termination condition'] != 'optimal':
#        return None, None

    W = np.zeros( (m, nh) )
    U = np.zeros(nh)
    for i in model.I:
        for h in model.H:
            W[i,h] = model.W[i,h]()
    for h in model.H:
        U[h] = model.U[h]()

    return W, U


def AccuracyMLP(Xs, Ys, W, U):
    y_hat = np.sign( np.matmul( np.sign(np.matmul(Xs, W)), np.transpose(U)))

    n = len(Ys)
    return (np.sum(Ys == y_hat))/n*100

#-----------------------------------------------
# MAIN function
#-----------------------------------------------
if __name__ == "__main__":
#    Xs, Ys = Parse('../data/train_nine_four.csv')
#    Xs, Ys = Parse('../data/train_three_four.csv')
    Xs, Ys = Parse('../data/all_three_four.csv')
    Xs, Ys = Parse('../data/all_nine_four.csv')

    T = 2000
    Xtrain, Xtest = Xs[:T], Xs[T:]
    Ytrain, Ytest = Ys[:T], Ys[T:]

    Ytrain = [1 if y == 4 else -1 for y in Ytrain]
    #DrawDigit(Xs[0].reshape((28,28)))

    print(Xs.shape)
    W, U = BNN(Xtrain, Ytrain, nh=1)

    n, m = W.shape
    print('zeros', np.sum(abs(W) < 1e-09), n*m)
    Ytest = [1 if y == 4 else -1 for y in Ytest]
    acc = AccuracyMLP(Xtest, Ytest, W, U)

    print('Accuracy:', round(acc, 3))
