# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 14:14:01 2021

@author: gualandi
"""

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

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
    return int(round(math.sqrt((A[0]-B[0])**2 + (A[1] - B[1])**2), 0))


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


def PlotSolution(Xs, Ws, Es):
    fig, ax = plt.subplots()
    for i, j in Es:
        DisegnaSegmento(Ps[i], Ps[j], ax)

    ax.scatter([i for i, j in Xs[1:]], [j for i, j in Xs[1:]],
               s=Ws[1:], alpha=0.3, cmap='viridis')

    for i in range(len(Xs[:])):
        ax.annotate(str(i+1), Xs[i])

    plt.plot([Xs[0][0]], [Xs[0][1]], marker='s', color='red', alpha=0.5)
    plt.axis('square')
    plt.axis('off')


def FlowCVRP(n, K, C, Ps, Ds, d, F, TIME_LIMIT, Pool):
    m = ConcreteModel()

    m.I = RangeSet(len(Ps))
    m.J = RangeSet(len(Ps))

    m.x = Var(m.I, m.J, domain=Binary)

    # y_ij
    m.y = Var(m.I, m.J, domain=NonNegativeReals)

    # Es = N x N \ {(i,i)}
    Es = []
    for i in m.I:
        for j in m.J:
            if i != j:
                Es.append((i, j))

    m.obj = Objective(expr=sum(F(Ps[i], Ps[j])*m.x[i, j] for i, j in Es))

    # Vincoli archi uscenti
    m.outdegree = ConstraintList()
    for i in m.I:
        if i != d:
            Ls = list(filter(lambda z: z != i, Ps))
            m.outdegree.add(expr=sum(m.x[i, j] for j in Ls) == 1)

    # Vincoli archi entranti
    m.indegree = ConstraintList()
    for j in m.J:
        if j != d:
            Ls = list(filter(lambda z: z != j, Ps))
            m.indegree.add(expr=sum(m.x[i, j] for i in Ls) == 1)

    Ls = list(filter(lambda z: z != d, Ps))
    m.d1 = Constraint(expr=sum(m.x[d, i] for i in Ls) == K)
    m.d2 = Constraint(expr=sum(m.x[i, d] for i in Ls) == K)

    # Flow balance constraints
    m.flow = ConstraintList()
    for i in m.I:
        if i == d:
            Ls = list(filter(lambda z: z != d, Ps))
            Dtot = sum(Ds[j] for j in Ds)  # TODO_: recheck for Ps (!)
            m.flow.add(expr=sum(m.y[j, d] for j in Ls) -
                       sum(m.y[i, d] for j in Ls) == -Dtot)
        else:
            Ls = list(filter(lambda z: z != i, Ps))
            m.flow.add(expr=sum(m.y[j, i] for j in Ls) -
                       sum(m.y[i, j] for j in Ls) == Ds[i])

    # Implied bounds
    m.flow_bnd = ConstraintList()
    for i, j in Es:
        m.flow_bnd.add(m.y[i, j] <= C * m.x[i, j])

    if Pool != []:
        m.pool = ConstraintList()
        for S, r in Pool:
            Es = []
            for i, j in m.x:
                if i in S and j not in S:
                    Es.append((i, j))

            m.pool.add(sum(m.x[i, j] for i, j in Es) >= r)

    # Print file for debug
    # m.write('flow.lp')

    # Solve the model
    SOLVER_NAME = 'gurobi'
    # SOLVER_NAME = 'glpk'

    solver = SolverFactory(SOLVER_NAME)

    if SOLVER_NAME == 'glpk':
        solver.options['tmlim'] = TIME_LIMIT
    elif SOLVER_NAME == 'gurobi':
        solver.options['TimeLimit'] = TIME_LIMIT

    sol = solver.solve(m, tee=True, load_solutions=False)

    # Get a JSON representation of the solution
    sol_json = sol.json_repn()
    # Check solution status
    if sol_json['Solver'][0]['Status'] != 'ok':
        return None, []

    # Load the solution
    m.solutions.load_from(sol)

    selected = []
    for i in m.I:
        for j in m.J:
            if i != j:
                if m.x[i, j]() > 0:
                    selected.append((i, j))

    return m.obj(), selected


def cMTZ(n, K, C, Ps, Ds, d, F, TIME_LIMIT):
    m = ConcreteModel()

    m.I = RangeSet(len(Ps))
    m.J = RangeSet(len(Ps))

    m.x = Var(m.I, m.J, domain=Binary)

    # Create Miller variables (subtour elimination)
    #  Ds[i] <= m.u[i] <= C
    m.u = Var(m.I, domain=NonNegativeReals, bounds=lambda m, i: (Ds[i], C))

    # Es = N x N \ {(i,i)}
    Es = []
    for i in m.I:
        for j in m.J:
            if i != j:
                Es.append((i, j))

    m.obj = Objective(expr=sum(F(Ps[i], Ps[j])*m.x[i, j] for i, j in Es))

    # Vincoli archi uscenti
    m.outdegree = ConstraintList()
    for i in m.I:
        if i != d:
            Ls = list(filter(lambda z: z != i, Ps))

            m.outdegree.add(expr=sum(m.x[i, j] for j in Ls) == 1)

    # Vincoli archi entranti
    m.indegree = ConstraintList()
    for j in m.J:
        if j != d:
            Ls = list(filter(lambda z: z != j, Ps))
            m.indegree.add(expr=sum(m.x[i, j] for i in Ls) == 1)

    Ls = list(filter(lambda z: z != d, Ps))
    m.d1 = Constraint(expr=sum(m.x[d, i] for i in Ls) >= K)
    m.d2 = Constraint(expr=sum(m.x[i, d] for i in Ls) >= K)

    # Subtour elimination constraints with MTZ
    m.subtour = ConstraintList()
    for i in Ls:
        for j in Ls:
            if i != j:
                m.subtour.add(m.u[i] - m.u[j] + C*m.x[i, j] <= C - Ds[j])

    # Solve the model
    SOLVER_NAME = 'gurobi'
    # SOLVER_NAME = 'glpk'

    solver = SolverFactory(SOLVER_NAME)

    if SOLVER_NAME == 'glpk':
        solver.options['tmlim'] = TIME_LIMIT
    elif SOLVER_NAME == 'gurobi':
        solver.options['TimeLimit'] = TIME_LIMIT

    sol = solver.solve(m, tee=True)

    # Get a JSON representation of the solution
    sol_json = sol.json_repn()
    print(sol_json)
    # Check solution status
    if not (sol_json['Solver'][0]['Status'] == 'ok'
            or sol_json['Solver'][0]['Status'] == 'aborted'):
        return None

    selected = []
    for i in m.I:
        for j in m.J:
            if i != j:
                if m.x[i, j]() > 0:
                    selected.append((i, j))

    return m.obj(), selected


def VRP(n, K, C, Ps, Ds, d, F, TimeLimit):
    m = ConcreteModel()

    m.I = RangeSet(len(Ps))
    m.J = RangeSet(len(Ps))

    m.x = Var(m.I, m.J, domain=Binary)

    # Es = N x N \ {(i,i)}
    Es = []
    for i in m.I:
        for j in m.J:
            if i != j:
                Es.append((i, j))

    m.obj = Objective(expr=sum(F(Ps[i], Ps[j])*m.x[i, j] for i, j in Es))

    # Vincoli archi uscenti
    m.outdegree = ConstraintList()
    for i in m.I:
        if i != d:
            Ls = []
            for z in Ps:
                if i != z:
                    Ls.append(z)

            Ls = list(filter(lambda z: z != i, Ps))

            m.outdegree.add(expr=sum(m.x[i, j] for j in Ls) == 1)

    # Vincoli archi entranti
    m.indegree = ConstraintList()
    for j in m.J:
        if j != d:
            Ls = list(filter(lambda z: z != j, Ps))
            m.indegree.add(expr=sum(m.x[i, j] for i in Ls) == 1)

    Ls = list(filter(lambda z: z != d, Ps))
    m.d1 = Constraint(expr=sum(m.x[d, i] for i in Ls) >= 3)
    m.d2 = Constraint(expr=sum(m.x[i, d] for i in Ls) >= 3)

    m.pairs = ConstraintList()
    # for a, b in [(19,20),(15,18),(7,2)]:
    #     m.pairs.add( expr = m.x[a,b] + m.x[b,a] <= 1 )

    for a, b in [(19, 20), (15, 18), (7, 2), (5, 6), (22, 8), (14, 12), (11, 14)]:
        Fs = []
        for i in Ps:
            if i == a or i == b:
                for j in Ps:
                    if i != j and j != a and j != b:
                        Fs.append((i, j))
        m.pairs.add(expr=sum(m.x[i, j] for i, j in Fs) >= 1)

    for a, b, c in [(5, 6, 9)]:
        Fs = []
        for i in Ps:
            if i == a or i == b or i == c:
                for j in Ps:
                    if i != j and j != a and j != b and j != c:
                        Fs.append((i, j))
        m.pairs.add(expr=sum(m.x[i, j] for i, j in Fs) >= 1)

    for S in [(10, 11, 14), (5, 6, 9, 10), (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23), (2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23),
              (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 20, 21, 22, 23)]:
        Fs = []
        for i in Ps:
            if i in S:
                for j in Ps:
                    if i != j and (j not in S):
                        Fs.append((i, j))
        K = ceil(sum(Ds[i] for i in S)/C)
        m.pairs.add(expr=sum(m.x[i, j] for i, j in Fs) >= K)

    # Fissiamo un sottoinsieme S \subset N = {2,...n} di nodi

    # Troviamo tutti le coppie (i,j) con i in S, j non in S (abbiamo chiamato Fs)

    # Poniamo il vincolo:
    #   \sum_{ij in Fs} x_ij >= "stima per difetto del numero di veicoli che mi servono
    #                            " per serivere tutti i nodi in S"
    #                            " intero più grande( \sum_{i in S} d_i/C)

    # Solve the model
    sol = SolverFactory('gurobi').solve(m, tee=True)

    # Get a JSON representation of the solution
    sol_json = sol.json_repn()
    # Check solution status
    if sol_json['Solver'][0]['Status'] != 'ok':
        return None

    selected = []
    for i in m.I:
        for j in m.J:
            if i != j:
                if m.x[i, j]() > 0:
                    selected.append((i, j))

    return m.obj(), selected


# Library for graphs


def VRPCut(n, K, C, Ps, Ds, d, F, TimeLimit):
    # Build a directed graph out of the data
    G = nx.DiGraph()

    n = len(Ps)
    for i in range(1, n+1):
        for j in range(1, n+1):
            if i != j and Ds[i] + Ds[j] <= C:
                G.add_edge(i, j, weight=F(Ps[i], Ps[j]))

    a = G.number_of_edges()

    # Build ILP Model
    m = ConcreteModel()

    m.N = RangeSet(n)

    # TODO: introduce only usefull variables (no arc, no variable)
    m.x = Var(m.N, m.N, domain=Binary)

    # Objective function of arc variables
    m.obj = Objective(expr=sum(G[i][j]['weight']*m.x[i, j]
                      for i, j in G.edges()))

    # Vincoli archi uscenti
    m.outdegree = ConstraintList()
    for i in m.N:
        if i != d:
            m.outdegree.add(expr=sum(m.x[v, w]
                            for v, w in G.out_edges(i)) == 1)

    # Vincoli archi entranti
    m.indegree = ConstraintList()
    for j in m.N:
        if j != d:
            m.indegree.add(expr=sum(m.x[v, w] for v, w in G.in_edges(j)) == 1)

    m.d1 = Constraint(expr=sum(m.x[v, w] for v, w in G.out_edges(d)) >= 3)
    m.d2 = Constraint(expr=sum(m.x[v, w] for v, w in G.in_edges(d)) >= 3)

    # Arc constraint
    m.arcs = ConstraintList()
    for i, j in G.edges():
        if i < j:
            m.arcs.add(m.x[i, j] + m.x[j, i] <= 1)

    # Fissiamo un sottoinsieme S \subset N = {2,...n} di nodi

    # Troviamo tutti le coppie (i,j) con i in S, j non in S (abbiamo chiamato Fs)

    # Poniamo il vincolo:
    #   \sum_{ij in Fs} x_ij >= "stima per difetto del numero di veicoli che mi servono
    #                            " per serivere tutti i nodi in S"
    #                            " intero più grande( \sum_{i in S} d_i/C)

    #                     Fs.append( (i,j) )
    #     K = ceil(sum(Ds[i] for i in S)/C)
    #     m.pairs.add( expr = sum(m.x[i,j] for i,j in Fs) >= K )

    # Solve the model
    solver = SolverFactory('gurobi')

    it = 0
    Pool = []
    while it <= 50:
        it += 1

        sol = solver.solve(m, tee=False)

        # Get a JSON representation of the solution
        sol_json = sol.json_repn()
        # Check solution status
        if sol_json['Solver'][0]['Status'] != 'ok':
            return None

        selected = []
        for i, j in m.x:
            if m.x[i, j]() and m.x[i, j]() > 0:
                selected.append((i, j))

        Cuts = SubtourElimin(selected, Ds, C)
        print("LB:", m.obj(), "Cuts: ", Cuts)

        if Cuts == []:
            print("Optimal solution found")
            break

        for S, r in Cuts:
            Pool.append((S, r))
            Es = []
            for i, j in G.edges():
                if i in S and j not in S:
                    Es.append((i, j))

            m.arcs.add(sum(m.x[i, j] for i, j in Es) >= r)

    return m.obj(), selected, Pool


def SubtourElimin(Es, d, C, depot=1):
    G = nx.Graph()

    for i, j in Es:
        G.add_edge(i, j)

    Cs = nx.cycle_basis(G)

    Subtours = []
    for cycle in Cs:
        c = sum(d[i] for i in cycle)
        S = list(filter(lambda z: z != depot, cycle))
        if c > C or depot not in cycle:
            Subtours.append((S, ceil(c/C)))

    return Subtours


# -----------------------------------------------
# MAIN function
# -----------------------------------------------
if __name__ == "__main__":
    # CONTROLLARE DOVE AVETE I FILE DI DATI
    # n, C, Ps, Ds, d = ParseFile("./E-n30-k3.vrp")
    n, C, Ps, Ds, d = ParseFile("./E-n23-k3.vrp")

    # fobj, Es = FlowCVRP(n, 3, C, Ps, Ds, d, Distance, 60, [])

    # fobj, Es = cMTZ(n, 3, C, Ps, Ds, d, Distance, 60)

    # fobj, Es = VRP(n, 3, C, Ps, Ds, d, Distance, 60)

    fobj, Es, Pool = VRPCut(n, 3, C, Ps, Ds, d, Distance, 10)
    # fobj, Es = FlowCVRP(n, 3, C, Ps, Ds, d, Distance, 60, Pool)

    print('valore funzione obiettivo:', fobj)

    Xs = [p for p in Ps.values()]
    Ws = [w for w in Ds.values()]

    PlotSolution(Xs, Ws, Es)
