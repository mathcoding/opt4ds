from gurobipy import Model, GRB, quicksum


# Residenze Collegiali a Pavia
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

# INSTANCES TAKE FROM THE TSPLIB:
#   http://elib.zib.de/pub/mp-testdata/tsp/tsplib/tsp/

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

def RandomTSP(n, _seed=13):
    from numpy import random
    random.seed(_seed)
    return [(x,y) for x,y in zip(random.random(n), random.random(n))]

# Import the Numerical Python library
import numpy as np
from math import sqrt

def CostMatrix(Ls):
    n = len(Ls)
    C = 100000*np.ones((n,n))
    for i, (a,b) in enumerate(Ls):
        for j, (c,d) in enumerate(Ls[i+1:]):
            C[i, i+j+1] = sqrt((a-c)**2 + (b-d)**2)
            C[i+j+1, i] = C[i, i+j+1]
            
    return C

# Import the NetworkX library
import networkx as nx

def BuildGraph(C):
    # Build an undirected graph out of the data
    G = nx.Graph()
    
    n,n = C.shape
    for i in range(n):
        for j in range(n):
            if i < j:
                G.add_edge(i, j, weight=C[i,j])

    return G

import pylab as pl
from matplotlib import collections as mc

def PlotTour(Ps, Ls, values):
    lines = [[Ps[i], Ps[j]] for i,j in Ls]
    fig, ax = pl.subplots()

    lc = mc.LineCollection(lines, linewidths=[1.5 if x > 0.99 else 1 for x in values],
                           colors=['blue' if x > 0.99 else 'orange' for x in values])
    
    ax.add_collection(lc)
    ax.scatter([i for i,j in Ps], [j for i,j in Ps], 
                s=20, alpha=0.8, color='red')
    
    for l, (i,j) in enumerate(Ps):
        ax.annotate(str(l), (i, j))

    ax.autoscale()
    ax.margins(0.1)
    ax.axis('equal')
    pl.show()

# Exercise: write a function to check the feasibility of your solution
def CheckFeasibility(sol):
    # TODO: complete this function
    return True

# TODO: complete the following script with your solution
def SolveTSP_SEC(G, TIME_LIMIT=120):
    return None

if __name__ == "__main__":
    Test = 2

    # Compute Cost Matrix
    if Test == 0:
        Ls = [(b,a) for a,b,_ in Rs]
    if Test == 1:
        Ls = ULYSSES
    if Test == 2:
        Ls = BAVIERA
    if Test == 3:
        N = 100
        Ls = RandomTSP(N)

    # Compute cost matrix
    C = CostMatrix(Ls)

    # Solve problem
    G = BuildGraph(C)

    z_lp, tour, values = SolveTSP_SEC(G) # 9.074148047873e+03
    #z_lp, tour, values = SolveTSP_MKT(G) # 9.074148047873e+03

    Ps = [(p[0],p[1]) for p in Ls]
    PlotTour(Ps, tour, values)
    