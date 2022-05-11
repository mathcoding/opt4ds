# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 14:23:32 2021

@author: gualandi
"""

import numpy as np

from pyomo.environ import ConcreteModel, Var, Objective, Constraint, SolverFactory
from pyomo.environ import Binary, RangeSet, NonNegativeReals


def ParseCancer(filename):
    fh = open(filename, 'r', encoding="utf-8")
    Xs = []
    Ys = []
    for line in fh:
        row = line.replace('\n','').split(',')        
        Xs.append( list(map(float, row[2:])) )
        Ys.append( int(row[1] == 'M') )
    return Xs, Ys    

        
def LinearClassifier(Xs, Ys):
    # Main Pyomo model
    model = ConcreteModel()
    # Parameters
    n = len(Xs)
    model.I = RangeSet(n)
    m = len(Xs[0])
    model.J = RangeSet(m)
    # Variables
    model.X = Var(model.J, bounds=(-float('inf'), float('inf'))) 
    model.X0 = Var(bounds=(-float('inf'), float('inf'))) 
    
    model.W = Var(model.I, within=NonNegativeReals) 
    model.U = Var(model.I, within=Binary)
    
    # Objective Function
    #model.obj = Objective(expr=sum(model.W[i] for i in model.I))
    model.obj = Objective(expr=sum(model.U[i] for i in model.I))
    
    # Constraints on the separation hyperplane
    def ConLabel(m, i):
        if Ys[i-1] == 0:
            return sum(Xs[i-1][j-1]*m.X[j] for j in m.J) >= m.X0 + 1 - m.W[i]
        else:
            return sum(Xs[i-1][j-1]*m.X[j] for j in m.J) <= m.X0 - 1 + m.W[i]
        
    model.Label = Constraint(model.I, rule = ConLabel)
    
    model.Viol = Constraint(model.I,
                            rule = lambda m, i: m.W[i] <= 10000*m.U[i])
    
    # Solve the model
    sol = SolverFactory('gurobi').solve(model, tee=True)

    # Get a JSON representation of the solution
    sol_json = sol.json_repn()
    # Check solution status
    if sol_json['Solver'][0]['Status'] != 'ok':
        return None
    if sol_json['Solver'][0]['Termination condition'] != 'optimal':
        return None

    return model.obj(), [model.X[j]() for j in model.J] + [model.X0()]


def CheckSolution(Xs, Ys, A):
    v, u = 0, 0.0
    
    for i,xs in enumerate(Xs):
        ax = sum(x*a for x,a in zip(xs, A[:-1]))
        if ax < A[-1] and Ys[i] == 0:
            v += 1
            u += abs(-A[-1] + ax)
        if ax > A[-1] and Ys[i] == 1:
            v += 1            
            u += abs(-A[-1] + ax)
    
    print("Violations: ", v, len(Xs))
    print("total:", u)
    print("Avg. ", round(u/v, 3))
        

def SplitTrainTestSet(Xs, Ys, t=0.3):
    Ax, Al = [], []  # Train sets
    Bx, Bl = [], []  # Test sets
    
    np.random.seed(13)
    
    for x, y in zip(Xs, Ys):
        if np.random.uniform(0, 1) > t:
            Ax.append(x)
            Al.append(y)
        else:
            Bx.append(x)
            Bl.append(y)
            
    return Ax, Al, Bx, Bl

    
def Accuracy(A, Bx, Bl):
    v = 0
    for xs, y in zip(Bx, Bl):
        ax = sum(x*a for x,a in zip(xs, A[:-1]))
        if ax < A[-1] and y == 0:
            v += 1
        if ax > A[-1] and y == 1:
            v += 1
    
    return round((len(Bx)-v)/len(Bx)*100,3), v, len(Bx)

    
def Confusion(A, Bx, Bl):
    tp, fp, tn, fn = 0, 0, 0, 0
    
    for xs, y in zip(Bx, Bl):
        ax = sum(x*a for x,a in zip(xs, A[:-1]))
        if ax >= A[-1] and y == 0:
            tn += 1
        if ax < A[-1] and y == 1:
            tp += 1
            
        if ax < A[-1] and y == 0:
            fn += 1
        if ax > A[-1] and y == 1:
            fp += 1
    
    return tp, fp, tn, fn


#-----------------------------------------------
# MAIN function
#-----------------------------------------------
if __name__ == "__main__":
    Xs, Ys = ParseCancer('breast_cancer_train.csv')

    Ax, Al, Bx, Bl = SplitTrainTestSet(Xs, Ys)        
    
    fobj, A = LinearClassifier(Ax, Al)

    print('-------------------------------------------')    
    print('Accuracy Train set:', Accuracy(A, Ax, Al))
    print('Accuracy Test  set:', Accuracy(A, Bx, Bl))

    print('-------------------------------------------')
    Xs, Ys = ParseCancer('breast_cancer_validate.csv')
    print('Accuracy Test  set:', Accuracy(A, Xs, Ys))
    print('Confusion Matrix: ', Confusion(A, Xs, Ys))
    
    if False:    
        fh = open("banknote_train.csv", "w")    
        for xs, y in zip(Ax, Al):
            fh.write(",".join(list(map(str,xs))+[str(y)])+"\n")
        fh.close()
        
        fh = open("banknote_validate.csv", "w")    
        for xs, y in zip(Bx, Bl):
            fh.write(",".join(list(map(str,xs))+[str(y)])+"\n")
        fh.close()
    
    