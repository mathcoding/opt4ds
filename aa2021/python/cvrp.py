# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 14:14:01 2021

@author: gualandi
"""

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

from gurobipy import Model, quicksum, GRB
from math import sqrt, ceil
from scipy.spatial import Delaunay

import matplotlib.pyplot as pyplot
import time
import math

import pyomo

from pyomo.environ import ConcreteModel, Var, Objective, Constraint, ConstraintList, SolverFactory
from pyomo.environ import maximize, Binary, RangeSet, PositiveIntegers, NonNegativeReals


def ParseFile(filename):
    doc = open(filename, 'r')
    # Salta le prime 3 righe
    for _ in range(3):
        doc.readline()
    # Leggi la dimensione
    n = int(doc.readline().split(' ')[2])
    # sala riga
    doc.readline()
    # Leggi la capacita
    C = int(doc.readline().split(' ')[2])
    # sala riga
    doc.readline()
    # Leggi posizioni
    Ps = {}
    for row in doc:
        row = row.rstrip().split(' ')
        if row[0] == 'DEMAND_SECTION':
            break
        row = list(map(lambda z: int(z), row))
        Ps[row[0]] = (row[1], row[2])
    # Leggi posizioni
    Ds = {}
    for row in doc:
        row = row.rstrip().split(' ')
        if row[0] == 'DEPOT_SECTION':
            break
        row = list(map(lambda z: int(z), row))
        Ds[row[0]] = row[1]
    d = int(next(doc).rstrip())

    return n, C, Ps, Ds, d


def Distance(A, B):
    return math.sqrt((A[0]-B[0])**2 + (A[1] - B[1])**2)

    

def DisegnaSegmento(A, B, ax):
    """ 
    Disegna un segmento nel piano dal punto A a al punto B
    Vedi manuale a: http://matplotlib.org/api/pyplot_api.html
    """
    # Disegna il segmento
    ax.plot([A[0], B[0]], [A[1], B[1]], 'b', lw=0.75)
    # Disegna gli estremi del segmento
    DisegnaPunto(A, ax)
    DisegnaPunto(B, ax)
    

def DisegnaPunto(A, ax):
    """
    Disegna un punto nel piano
    """
    ax.plot([A[0]], [A[1]], 'bo', alpha=0.5)
    
    
def VRPSol(n, K, C, Ps, Ds, d, F, TimeLimit):
    m = Model()
    m.setParam(GRB.Param.TimeLimit, TimeLimit)
    # Create variables
    x = {}
    for i in Ps:
        for j in Ps:
            if i != j :
                x[i,j] = m.addVar(obj=F(Ps[i], Ps[j]), vtype=GRB.BINARY,
                                  name='x'+str(i)+'_'+str(j))
    # Create Miller variables (subtour elimination)
    u = {}
    for i in Ds:
        if i != d:
            u[i] = m.addVar(obj=0.0, vtype=GRB.CONTINUOUS,
                            lb=Ds[i], ub=C, name='u'+str(i))
            
    m.update()
    # Add outdegree constraint
    for j in Ps:
        if j != d:
            Ls = list(filter(lambda z: z!=j, Ps))
            m.addConstr(quicksum(x[i,j] for i in Ls) == 1)
    # Add indegree constraint
    for i in Ps:
        if i != d:
            Ls = filter(lambda z: z!=i, Ps)
            m.addConstr(quicksum(x[i,j] for j in Ls) == 1)
    # Number of vehicles
    Ls = list(filter(lambda z: z!=d, Ps))                   
    m.addConstr(quicksum(x[i,d] for i in Ls) == K)
    m.addConstr(quicksum(x[d,j] for j in Ls) == K)
    # Subtour elimination constraints
    for i in Ls:
        for j in Ls:
            if i != j:
                m.addConstr( u[i] - u[j] +C*x[i,j] <= C - Ds[j] )

    m.update()

    # Solve the model
    m.optimize()
    solution = m.getAttr('x', x)
    selected = [(i,j) for (i,j) in solution if solution[i,j] > 0.5]

    # Xs = [p for p in Ps.values()]
    # Ws = [w for w in Ds.values()]
    
    # for i,j in selected:
    #     DisegnaSegmento(Ps[i], Ps[j])
    # plt.scatter([i for i,j in Xs[1:]], [j for i,j in Xs[1:]], 
    #             s=Ws[1:], alpha=0.3, cmap='viridis')
    # plt.plot([Xs[0][0]], [Xs[0][1]], marker='s', color='red', alpha=0.5)
    # plt.axis('square')
    # plt.axis('off')
    
    return m.objVal, selected

def VRPGu(n, K, C, Ps, Ds, d, F, TimeLimit):
    m = Model()
    m.setParam(GRB.Param.TimeLimit, TimeLimit)
    # Create variables
    x = {}
    for i in Ps:
        for j in Ps:
            if i != j :
                x[i,j] = m.addVar(obj=F(Ps[i], Ps[j]), vtype=GRB.BINARY,
                                  name='x'+str(i)+'_'+str(j))
    # Create Miller variables (subtour elimination)
    u = {}
    for i in Ds:
        if i != d:
            u[i] = m.addVar(obj=0.0, vtype=GRB.CONTINUOUS,
                            lb=Ds[i], ub=C, name='u'+str(i))
            
    m.update()
    # Add outdegree constraint
    for j in Ps:
        if j != d:
            Ls = list(filter(lambda z: z!=j, Ps))
            m.addConstr(quicksum(x[i,j] for i in Ls) == 1)
    # Add indegree constraint
    for i in Ps:
        if i != d:
            Ls = filter(lambda z: z!=i, Ps)
            m.addConstr(quicksum(x[i,j] for j in Ls) == 1)
            
    # Number of vehicles
    Ls = list(filter(lambda z: z!=d, Ps))                   
    m.addConstr(quicksum(x[i,d] for i in Ls) <= K)
    m.addConstr(quicksum(x[d,j] for j in Ls) <= K)
    
    m.addConstr(quicksum(x[i,d] for i in Ls) >= 3)
    m.addConstr(quicksum(x[d,j] for j in Ls) >= 3)

    #

    for a,b in [(15,18), (9,22), (19,20), (5,6), (8,22), (7,2), (9,10)]:
        Fs = []
        for i in Ps:
            if i == a or i == b:        
                for j in Ps:
                    if i != j and j != a and j != b:
                        Fs.append((i,j))
                        
        m.addConstr(quicksum(x[i,j] for i,j in Fs) >= 1)


    for a,b,c in [(5,6,9), (10,11,14)]:
        Fs = []
        for i in Ps:
            if i == a or i == b or i == c:        
                for j in Ps:
                    if i != j and j != a and j != b and j != c:
                        Fs.append((i,j))
                        
        m.addConstr(quicksum(x[i,j] for i,j in Fs) >= 2)

        
    # for a,b in [(15,18), (9,22), (19,20), (5,6), (8,22), (7,2)]:
    #     m.addConstr(x[a,b] + x[b,a] <= 1)
    
    m.update()

    # Solve the model
    m.optimize()
    solution = m.getAttr('x', x)
    selected = [(i,j) for (i,j) in solution if solution[i,j] > 0.5]

    # Xs = [p for p in Ps.values()]
    # Ws = [w for w in Ds.values()]
    
    # for i,j in selected:
    #     DisegnaSegmento(Ps[i], Ps[j])
    # plt.scatter([i for i,j in Xs[1:]], [j for i,j in Xs[1:]], 
    #             s=Ws[1:], alpha=0.3, cmap='viridis')
    # plt.plot([Xs[0][0]], [Xs[0][1]], marker='s', color='red', alpha=0.5)
    # plt.axis('square')
    # plt.axis('off')
    
    return m.objVal, selected


def VRP(n, K, C, Ps, Ds, d, F, TimeLimit):
    m = ConcreteModel()

    m.I = RangeSet(len(Ps))
    m.J = RangeSet(len(Ps))

    m.x = Var(m.I, m.J, domain=Binary)

    # Objective Function
    Es = []
    for i in m.I:
        for j in m.J:
            if i != j:
                Es.append((i,j))
    m.obj = Objective(expr = sum(F(Ps[i], Ps[j])*m.x[i,j] for i,j in Es))

            
    # Add outdegree constraint
    m.outdegree = ConstraintList()
    for j in m.J:
        if j != d:
            Ls = list(filter(lambda z: z!=j, Ps))
            m.outdegree.add(expr = sum(m.x[i,j] for i in Ls) == 1)

    # Add indegree constraint
    m.indegree = ConstraintList()
    for i in m.J:
        if i != d:
            Ls = list(filter(lambda z: z!=i, Ps))
            m.indegree.add(expr = sum(m.x[i,j] for j in Ls) == 1)


    Ls = list(filter(lambda z: z!=d, Ps))                   
    m.d1 = Constraint(expr = sum(m.x[i,d] for i in Ls) >= 3)
    m.d2 = Constraint(expr = sum(m.x[d,j] for j in Ls) >= 3)


    m.subtours = ConstraintList()
    for a,b in [(15,18), (9,22), (19,20), (5,6), (8,22), (7,2), (9,10)]:
        m.subtours.add(expr = m.x[a,b] + m.x[b,a] <= 1)
    

    m.triple = ConstraintList()
    for a,b,c in [(5,6,9), (10,11,14)]:
        Fs = []
        for i in Ps:
            if i == a or i == b or i == c:        
                for j in Ps:
                    if i != j and j != a and j != b and j != c:
                        Fs.append((i,j))
                        
        m.triple.add(expr = sum(m.x[i,j] for i,j in Fs) >= 2)

        
    # Solve the model
    sol = SolverFactory('gurobi').solve(m, tee=True)
    
    # CHECK SOLUTION STATUS
    
    # Get a JSON representation of the solution
    sol_json = sol.json_repn()
    
    if sol_json['Solver'][0]['Status'] != 'ok':
        print('Error in solving the model')
        return(None)
    
    # DOPO AVER CONTROLLATO LO STATUS
    print("Optimal solution value:", round(m.obj(), 3))

    selected = []
    for i in m.I:
        for j in m.J:
            if i!=j:
                if m.x[i,j]() > 0:
                    selected.append((i,j))
    return m.obj(), selected

    
def PlotSolution(Xs, Ws, Es):
    fig, ax = plt.subplots()
    for i,j in Es:
        DisegnaSegmento(Ps[i], Ps[j], ax)
    
    ax.scatter([i for i,j in Xs[1:]], [j for i,j in Xs[1:]], 
                s=Ws[1:], alpha=0.3, cmap='viridis')
    
    for i in range(len(Xs[1:])):
        ax.annotate(str(i), Xs[i])
    
    plt.plot([Xs[0][0]], [Xs[0][1]], marker='s', color='red', alpha=0.5)
    plt.axis('square')
    plt.axis('off')
    
    
#-----------------------------------------------
# MAIN function
#-----------------------------------------------
if __name__ == "__main__":
    # Xs, Ws, C, K = ParseCVRP('X-n101-k25.vrp') + (25,)
    # Xs, Ws, C, K = ParseCVRP('E-n33-k4.vrp') + (4,)
    # Xs, Ws, C, K = ParseCVRP('E-n101-k8.vrp') + (8,)
    # Xs, Ws, C, K = ParseCVRP('E-n30-k3.vrp') + (3,)
    # Xs, Ws, C, K = ParseFile('E-n23-k3.vrp') + (3,)
    # Xs, Ws, C, K = ParseCVRP('E-n51-k5.vrp') + (5,)
    # Xs, Ws, C, K = ParseCVRP('Leuven1.vrp') + (203,)

    n, C, Ps, Ds, d = ParseFile("../data/E-n23-k3.vrp")
    fobj, Es = VRP(n, 4, C, Ps, Ds, d, Distance, 600)      

    Xs = [p for p in Ps.values()]
    Ws = [w for w in Ds.values()]
    
    PlotSolution(Xs, Ws, Es)
    