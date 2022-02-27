# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 14:23:32 2021

@author: gualandi
"""

import numpy as np
import matplotlib.pyplot as plt

from pyomo.environ import ConcreteModel, Var, Objective, Constraint, SolverFactory
from pyomo.environ import maximize, minimize, Binary, RangeSet, ConstraintList, NonNegativeReals, Reals


def Gaussian(n, mu, sigma):
    return np.random.normal(mu, sigma, (n,2))


def RandomData(n, d=0.5):
    np.random.seed(17)
    As = Gaussian(2*n, 2, d)
    Bs = Gaussian(n, 4, d)   
    
    Xs = []
    Ys = []
    
    for a in As:
        Xs.append(a)
        Ys.append(0)
    for a in Bs:
        Xs.append(a)
        Ys.append(1)
        
    return Xs, Ys


def RandomDataQ(n, d=0.5):
    np.random.seed(17)
    
    Xs = []
    Ys = []
    
    for a in Gaussian(n, (2,1), d):
        Xs.append(a)
        Ys.append(0)
    for a in Gaussian(n, (6,1), d):
        Xs.append(a)
        Ys.append(0)

    for a in Gaussian(n, (4,1), d):
        Xs.append(a)
        Ys.append(1)
        
    return Xs, Ys

 
def MaxMargin(Xs, Ys):
    # Main Pyomo model
    model = ConcreteModel()
    # Parameters
    n = len(Xs)
    model.I = RangeSet(n)
    m = len(Xs[0])
    model.J = RangeSet(m)
    # Variables
    model.X = Var(model.J, within=Reals) 
    model.X0 = Var(within=Reals) 
    
    model.Gamma = Var(within=Reals) 
    
    
    # Objective Function
    model.obj = Objective(expr=model.Gamma, sense=maximize)
    
    # Constraints on the separation hyperplane
    def ConLabel(m, i):
        if Ys[i-1] == 0:
            return sum(Xs[i-1][j-1]*m.X[j] for j in m.J) >= m.X0 + m.Gamma# - m.W[i]
        else:
            return sum(Xs[i-1][j-1]*m.X[j] for j in m.J) <= m.X0 - m.Gamma# + m.W[i]
        
    model.Margin = Constraint(model.I, rule = ConLabel)
    
    model.Norm = Constraint(expr = sum(model.X[j] for j in model.J) - model.X0 == 1)
    
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

       
def LinearClassifier(Xs, Ys):
    # Main Pyomo model
    model = ConcreteModel()
    # Parameters
    n = len(Xs)
    model.I = RangeSet(n)
    m = len(Xs[0])
    model.J = RangeSet(m)
    # Variables
    model.X = Var(model.J, within=Reals) 
    model.X0 = Var(within=Reals) 
    
    model.W = Var(model.I, within=NonNegativeReals) 
    model.U = Var(model.I, within=Binary)
    
    # Objective Function
    # model.obj = Objective(expr=sum(model.W[i] for i in model.I))
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


def QuadraticClassifier(Xs0, Ys):
    Xs = []
    for x in Xs0:
        Xs.append([x[0]*x[0], x[0], x[1]])
    
    # Main Pyomo model
    model = ConcreteModel()
    # Parameters
    n = len(Xs)
    model.I = RangeSet(n)
    m = len(Xs[0])
    model.J = RangeSet(m)
    # Variables
    model.X = Var(model.J, within=Reals) 
    model.X0 = Var(within=Reals) 
    
    model.W = Var(model.I, within=NonNegativeReals) 
    model.U = Var(model.I, within=Binary)
    
    # Objective Function
    # model.obj = Objective(expr=sum(model.W[i] for i in model.I))
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


def PlotSolution(Xs, Ys, A):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    
    ax1.scatter([x[0] for x in Xs], [x[1] for x in Xs], 
                color=['green' if y == 1 else 'blue' for y in Ys],
                alpha=0.35)
    
    ax2.scatter([x[0] for x in Xs], [x[1] for x in Xs], 
                color=['green' if y == 1 else 'blue' for y in Ys],
                alpha=0.35)
    
    xmin = min(x[0] for x in Xs)
    xmax = max(x[0] for x in Xs)
    ymin = min(x[1] for x in Xs)
    ymax = max(x[1] for x in Xs)
    x = np.linspace(xmin, xmax, 10)
    
    # Linear:
    y = -A[0]/A[1]*x + A[2]/A[1]    
    
    ax2.plot(x, y, color='red')
    
    # Punti violati
    Vs = []
    for i,x in enumerate(Xs):
        if A[0]*x[0] + A[1]*x[1] < A[2] and Ys[i] == 0:
            Vs.append(x)                        
        else:
            if A[0]*x[0] + A[1]*x[1] > A[2] and Ys[i] == 1:
                Vs.append(x)
    
    ax2.scatter([x[0] for x in Vs], [x[1] for x in Vs], color='red', alpha=0.5, marker='x')
    
    # Plot finale
    ax1.axis([xmin-0.5, xmax+0.5, ymin-0.5, ymax+0.5])
    ax2.axis([xmin-0.5, xmax+0.5, ymin-0.5, ymax+0.5])
    plt.savefig('lin_classifier.pdf') 
    plt.show()    
    

def PlotSolutionQ(Xs, Ys, A):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    
    ax1.scatter([x[0] for x in Xs], [x[1] for x in Xs], 
                color=['green' if y == 1 else 'blue' for y in Ys],
                alpha=0.35)
    
    ax2.scatter([x[0] for x in Xs], [x[1] for x in Xs], 
                color=['green' if y == 1 else 'blue' for y in Ys],
                alpha=0.35)
    
    xmin = min(x[0] for x in Xs)
    xmax = max(x[0] for x in Xs)
    ymin = min(x[1] for x in Xs)
    ymax = max(x[1] for x in Xs)
    x = np.linspace(xmin, xmax, 10)

    print(xmin,xmax)    
    # Quadratic:
    y = -A[0]/A[2]*x*x -A[1]/A[2]*x + A[3]/A[2]
    
    
    ax2.plot(x, y, color='red')
    
    # Punti violati
    Vs = []
    for i,x in enumerate(Xs):
        if A[0]*x[0]*x[0] + A[1]*x[0] + A[2]*x[1] < A[-1] and Ys[i] == 0:
            Vs.append(x)                        
        else:
            if A[0]*x[0]*x[0] + A[1]*x[0] + A[2]*x[1] > A[-1] and Ys[i] == 1:
                Vs.append(x)
    
    ax2.scatter([x[0] for x in Vs], [x[1] for x in Vs], color='red', alpha=0.5, marker='x')
    
    # Plot finale
    ax1.axis([xmin-0.5, xmax+0.5, ymin-0.5, ymax+0.5])
    ax2.axis([xmin-0.5, xmax+0.5, ymin-0.5, ymax+0.5])
    # plt.savefig('qua_classifier.pdf') 
    plt.show()   
    
    
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
        
    # Modify the following conditional expression to decide
    # which code to run
    if True:    
        Xs, Ys = RandomData(50, 0.75)
        fobj, A = MaxMargin(Xs, Ys)
        PlotSolution(Xs, Ys, A)
    
        fobj, A = LinearClassifier(Xs, Ys)
        PlotSolution(Xs, Ys, A)
    
        fobj, A = QuadraticClassifier(Xs, Ys)
        PlotSolutionQ(Xs, Ys, A)
    
    if False:    
        Xs, Ys = RandomDataQ(50, 0.25)
        
        fobj, A = LinearClassifier(Xs, Ys)
        PlotSolution(Xs, Ys, A)
        
        fobj, A = QuadraticClassifier(Xs, Ys)
        PlotSolutionQ(Xs, Ys, A)
    
        
    # print("Accuracy:", Accuracy(A, Xs, Ys)[0])
    # print("Confusion Matrix:", Confusion(A, Xs, Ys))