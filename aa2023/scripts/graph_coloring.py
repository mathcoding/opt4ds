# -*- coding: utf-8 -*-
"""
@author: Gualandi
"""

from pyomo.environ import ConcreteModel, Var, Objective, Constraint, SolverFactory
from pyomo.environ import maximize, Binary, RangeSet, PositiveIntegers

import networkx as nx


def GraphColoring(G):
    # Number of nodes
    n = G.number_of_nodes()
    m = G.number_of_edges()

    # Create concrete model
    model = ConcreteModel()

    # Set of indices
    model.V = RangeSet(0, n-1)
    model.E = RangeSet(0, m-1)
    model.K = RangeSet(0, n-1)

    # Variables
    model.x = Var(model.V, model.K, within=Binary)
    model.y = Var(model.K, within=Binary)

    # Objective Function
    model.obj = Objective(expr=sum(model.y[k] for k in model.K))

    # Every must receive a single color
    def Unique(model, i):
        return sum(model.x[i, k] for k in model.K) == 1

    model.assign = Constraint(model.V, rule=Unique)

    # Conflict arcs
    def Conflict(m, i, j, k):
        if (i,j) in G.edges():
            return m.x[i,k] + m.x[j,k] <= m.y[k]
        return Constraint.Skip

    model.conflict = Constraint(model.V, model.V, model.K, rule=Conflict)

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
        for k in model.K:
            if model.x[i,k]() == 1:
                sol.append( (i,k) )

    return model.obj(), sol


def PlotSolution(G, sol):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    limits = plt.axis("off")  # turn off axis
    
    # Learn to search in StackOverflow:
    # https://stackoverflow.com/questions/13517614/draw-different-color-for-nodes-in-networkx-based-on-their-node-value
    
    color_lookup = {v: k for v,k in sol}

    low, *_, high = sorted(color_lookup.values())
    norm = mpl.colors.Normalize(vmin=low, vmax=high, clip=True)
    mapper = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.coolwarm)
    colors = [mapper.to_rgba(i) for i in color_lookup.values()]

    nx.draw(G, pos = nx.kamada_kawai_layout(G), nodelist=color_lookup, node_size=1000, node_color=colors, with_labels=True)

    plt.show()

# -----------------------------------------------
#   MAIN function
# -----------------------------------------------
if __name__ == "__main__":

    G = nx.petersen_graph()
    #G = nx.chvatal_graph()

    G = nx.erdos_renyi_graph(20, 0.3, seed=13)

    xhi, sol = GraphColoring(G)

    print("Minimum number of colors:", xhi)

    print(sol)
    PlotSolution(G, sol)