# -*- coding: utf-8 -*-
"""
@author: Gualandi
"""

from pyomo.environ import ConcreteModel, Var, Objective, Constraint, SolverFactory
from pyomo.environ import maximize, Binary, RangeSet, PositiveIntegers, Set, ConstraintList

import networkx as nx

def RNAfolding(seq):
    # Complements: {A, U}, {C, G}

    # Input size
    n = len(seq)

    # Create concrete model
    model = ConcreteModel()

    # Set of indices
    model.V = RangeSet(0, n-1)

    E = []
    for i in range(n):
        for j in range(n):
            if i < j:
                E.append( (i,j) )
    model.A = Set(within=model.V * model.V, initialize=E)

    # Variables
    model.x = Var(model.A, within=Binary)

    # Objective Function
    model.obj = Objective(expr=sum(model.x[a] for a in model.A), sense=maximize)

    # Set variable to zero
    def SetZero(m, i, j):
        if seq[i] == 'A' and seq[j] == 'U' or seq[i] == 'U' and seq[j] == 'A':
            return Constraint.Skip
        elif seq[i] == 'C' and seq[j] == 'G' or seq[i] == 'G' and seq[j] == 'C':
            return Constraint.Skip
        return model.x[i,j] == 0
        
    model.fixed = Constraint(model.A, rule=SetZero)

    # At most a pair
    def AtMost(m, i):
        return sum(m.x[k,i] for k in range(i)) + sum(m.x[i,k] for k in range(i+1, n)) <= 1

    model.atMost = Constraint(model.V, rule=AtMost)

    # Crossing edges
    model.crossing = ConstraintList() 
    for i,j in model.A:
        for v,w in model.A:
            if i < v < j < w:
                model.crossing.add( model.x[i,j] + model.x[v,w] <= 1)

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

    sol = []
    for i, j in model.A:
        if model.x[i,j]() > 0.5:
            sol.append( (i,j,seq[i],seq[j]) )

    return model.obj(), sol


def BuildGraph(seq):
    # Build a graph from scratch
    G = nx.Graph()

    # First, add all nodes
    n = len(seq)
    for i in range(n):
        for j in range(i+1, n):
            if (seq[i] == 'A' and seq[j] == 'U') or (seq[i] == 'U' and seq[j] == 'A'):
                G.add_node( (i, j) )
            if (seq[i] == 'C' and seq[j] == 'G') or (seq[i] == 'G' and seq[j] == 'C'):
                G.add_node( (i, j) )

    # Now add all arcs
    for i, j in G.nodes():
        for v, w in G.nodes():
            if (i,j) != (v,w) and i <= v <= j <= w:
                G.add_edge( (i,j), (v, w) )

    print(G.number_of_nodes(), G.number_of_edges())

    H = nx.Graph()
    M = {}
    for v, (i,j) in enumerate(G.nodes()):
        H.add_node(v, label=(i,j, seq[i], seq[j]))
        M[i,j] = v

    for (a,b) in G.edges():
        H.add_edge(M[a], M[b])

    return H


def MaxStableSet(G):
    # Number of nodes
    n = G.number_of_nodes()

    # Create concrete model
    model = ConcreteModel()

    # Set of indices
    model.V = RangeSet(0, n-1)

    # Variables
    model.z = Var(model.V, within=Binary)

    # Objective Function
    model.obj = Objective(expr=sum(model.z[v] for v in model.V), sense=maximize)

    # Conflict arcs
    def Conflict(m, i, j):
        if (i,j) in G.edges():
             return m.z[i] + m.z[j] <= 1.0
        return Constraint.Skip

    model.conflict = Constraint(model.V, model.V, rule=Conflict)

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

    sol = []
    for i in model.V:
        if model.z[i]() > 0.5:
            sol.append( (i, G.nodes[i]['label']) )

    return model.obj(), sol


def MaxClique(G):
    H = nx.complement(G)

    for clique in list(sorted(nx.find_cliques(H), key=len, reverse=True))[:3]:
        print(len(clique), [G.nodes[v]['label'] for v in clique])


# -----------------------------------------------
#   MAIN function
# -----------------------------------------------
if __name__ == "__main__":
    seq = "ACGUGCCCGAU"
    seq2 = "GCGGGAUGGCCCAAGUGCCAUUGAUGACCUGA"

    zbar, xbar = RNAfolding(seq2)

    print('Value:', zbar)
    print('Solution:', xbar)

    G = BuildGraph(seq2)

    w, sol = MaxStableSet(G)
    print(w, sol)

    print([G.nodes[v]['label'] for v in nx.maximal_independent_set(G)])

    MaxClique(G)