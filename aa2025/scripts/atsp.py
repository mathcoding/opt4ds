# Import the Numerical Python library
import numpy as np
# Import the NetworkX library
import networkx as nx

from math import sqrt

import pylab as pl
from matplotlib import collections as mc

from gurobipy import Model, GRB, quicksum


# Residenza Collegiali a Pavia
Rs = [(45.1882789,9.1600456, 'Del Maino', 0),
      (45.1961107,9.1395709, 'Golgi', 1), (45.1851618,9.1506323, 'Senatore', 2),
      (45.1806049,9.1691651, 'Don Bosco', 3), (45.1857651,9.1473637, 'CSA', 4),
      (45.1802511,9.1591663, 'Borromeo', 5), (45.1877192,9.1578934, 'Cairoli', 6),
      (45.1870975,9.1588276, 'Castiglioni', 7), (45.1871301,9.1435067, 'Santa Caterina', 8),
      (45.1863927,9.15947, 'Ghislieri', 9), (45.2007148,9.1325475, 'Nuovo', 10),
      (45.1787292,9.1635482, 'Cardano', 11), (45.1864928,9.1560687, 'Fraccaro', 12),
      (45.1989668,9.1775168, 'Griziotti', 13), (45.1838819,9.161318, 'Spallanzani', 14),
      (45.1823523,9.1454315, 'Valla', 15), (45.2007816,9.1341354, 'Volta', 16),
      (45.2070857,9.1382623, 'Residenza Biomedica', 17)]

ULYSSES = [(38.24, 20.42), (39.57, 26.15), (40.56, 25.32), (36.26, 23.12),
           (33.48, 10.54), (37.56, 12.19), (38.42, 13.11), (37.52, 20.44),
           (41.23, 9.10), (41.17, 13.05), (36.08, -5.21), (38.47, 15.13), 
           (38.15, 15.35), (37.51, 15.17), (35.49, 14.32), (39.36, 19.56)]
     
BAVIERA = [(1150.0,  1760.0), (630.0,  1660.0),  (40.0,  2090.0),    (750.0,  1100.0), 
  (1030.0,  2070.0), (1650.0,   650.0), (1490.0,  1630.0),  (790.0,  2260.0),
  (710.0,  1310.0),  (840.0,   550.0),  (1170.0,  2300.0),  (970.0,  1340.0),
  (510.0,   700.0),  (750.0,   900.0),  (1280.0,  1200.0),  (230.0,   590.0),
  (460.0,   860.0),  (1040.0,   950.0), (590.0,  1390.0),   (830.0,  1770.0),
  (490.0,   500.0),  (1840.0,  1240.0), (1260.0,  1500.0),  (1280.0,  790.0),
  (490.0,  2130.0),  (1460.0,  1420.0), (1260.0,  1910.0),  (360.0,  1980.0),
  (750.0,  2030.0)] 

def CostMatrix(Ls):
    n = len(Ls)
    C = 100000*np.ones((n,n))
    for i, (a,b) in enumerate(Ls):
        for j, (c,d) in enumerate(Ls[i+1:]):
            C[i, i+j+1] = sqrt((a-c)**2 + (b-d)**2)
            C[i+j+1, i] = C[i, i+j+1]
    return C

def BuildDiGraph(C):
    # Build a directed graph out of the data
    G = nx.DiGraph()
    n,n = C.shape
    for i in range(n):
        for j in range(n):
            if i != j:
                G.add_edge(i, j, weight=C[i,j])
    return G

def PlotTour(Ps, Ls, values):
    lines = [[Ps[i], Ps[j]] for i,j in Ls]
    fig, ax = pl.subplots()

    lc = mc.LineCollection(lines, linewidths=[1.5 if x > 0.501 else 1 for x in values],
                           colors=['blue' if x > 0.501 else 'orange' for x in values])
    
    ax.add_collection(lc)
    ax.scatter([i for i,j in Ps], [j for i,j in Ps], 
                s=20, alpha=0.8, color='red')
    
    ax.autoscale()
    ax.margins(0.1)
    ax.axis('equal')
    pl.show()


def SolveTSP(G, TIME_LIMIT=5):
    model = Model()
    model.setParam('TimeLimit', TIME_LIMIT)
    model.setParam('OutputFlag', 0)
    model.setParam('Threads', 1)
    n = G.number_of_nodes()
    
    # Create variables
    x = {}
    for i, j in G.edges():
        x[i,j] = model.addVar(obj=G[i][j]['weight'], vtype=GRB.CONTINUOUS, name=f'x_{i}_{j}')
        #x[i,j] = model.addVar(obj=G[i][j]['weight'], vtype=GRB.BINARY, name=f'x_{i}_{j}')

    # Create constraints
    for i in G.nodes():
        model.addConstr(quicksum(x[i,j] for i,j in G.out_edges(i)) == 1, name=f'out_{i}')

    for j in G.nodes():
        model.addConstr(quicksum(x[i,j] for i,j in G.in_edges(j)) == 1, name=f'in_{j}')

    # for i,j in G.edges():
    #     model.addConstr(x[i,j] + x[j,i] <= 1, name=f'no_subtour_{i}_{j}')

    while True:
        model.optimize()
        if model.status != GRB.OPTIMAL and model.status != GRB.TIME_LIMIT:
            print('Model status:', model.status)
            return 0, []
        xbar = [x[i,j].x for i,j in x if x[i,j].x > 0.001]
        arcs = [(i,j) for i,j in x if x[i,j].x > 0.001]

        # Support graph
        H = nx.Graph()
        for i,j in arcs:
            H.add_edge(i, j)
        Subtours = list(nx.connected_components(H))

        # Tour length: 73.67366520030248 Number of subtours: 1
        # Tour length: 73.98761804517501 Number of subtours: 1
        print('Tour length:', model.ObjVal, 'Number of subtours:', len(Subtours))

        if len(Subtours) == 1:
            break 

        for S in Subtours:
            model.addConstr(quicksum(x[i,j] for i in S for j in range(n) if j not in S) >= 1, 
                            name=f'subtour_{S}')
            
        PlotTour(Ps, arcs, xbar)
                                     

    return model.ObjVal, arcs, xbar


def SolveTSP_MKT(G, TIME_LIMIT=60):
    model = Model()
    model.setParam('TimeLimit', TIME_LIMIT)
    model.setParam('OutputFlag', 1)
    model.setParam('Threads', 1)
    n = G.number_of_nodes()
    
    # Create variables
    x = {}
    for i, j in G.edges():
        x[i,j] = model.addVar(obj=G[i][j]['weight'], vtype=GRB.BINARY, name=f'x_{i}_{j}')

    # Position variables
    u = {}
    for i in G.nodes():
        if i != 0:
            u[i] = model.addVar(vtype=GRB.INTEGER, obj=0, name=f'u_{i}')

    # Create constraints
    for i in G.nodes():
        model.addConstr(quicksum(x[i,j] for i,j in G.out_edges(i)) == 1, name=f'out_{i}')

    for j in G.nodes():
        model.addConstr(quicksum(x[i,j] for i,j in G.in_edges(j)) == 1, name=f'in_{j}')

    # Position constraints
    for i,j in G.edges():
        if i != 0 and j != 0:
            model.addConstr(u[i] - u[j] + 1 <= (n-1)*(1-x[i,j]), name=f'pos_{i}_{j}')

    model.optimize()

    if model.status != GRB.OPTIMAL and model.status != GRB.TIME_LIMIT:
        print('Model status:', model.status)
        return 0, []
    xbar = [x[i,j].x for i,j in x if x[i,j].x > 0.001]
    arcs = [(i,j) for i,j in x if x[i,j].x > 0.001]

    return model.ObjVal, arcs, xbar


if __name__ == '__main__':
    #Ps = [(x,y) for x,y,_,_ in Rs[:-1]]
    
    Ps = ULYSSES
    C = CostMatrix(Ps)
    G = BuildDiGraph(C)

    # n = len(Ps)
    # values = [1 for _ in range(n)]
    # RandomTour = [(i, (i+1)%n) for i in range(n)]

#    fobj, arcs, xbar = SolveTSP(G, TIME_LIMIT=60)
    fobj, arcs, xbar = SolveTSP_MKT(G, TIME_LIMIT=60)
    PlotTour(Ps, arcs, xbar)


