# -*- coding: utf-8 -*-
"""
@author: Gualandi
"""

import numpy as np
import networkx as nx

from math import sqrt
from time import sleep

from pyomo.environ import ConcreteModel, Var, Objective, Constraint, SolverFactory, Set
from pyomo.environ import maximize, Binary, RangeSet, PositiveReals, ConstraintList, NonNegativeReals

# Residenza Collegiali a Pavia
Rs = [(45.1882789,9.1600456, 'Del Maino'),(45.2070857,9.1382623, 'Green Campus'),
      (45.1961107,9.1395709, 'Golgi'),(45.1851618,9.1506323, 'Senatore'),
      (45.1806049,9.1691651, 'Don Bosco'),(45.1857651,9.1473637, 'CSA'),
      (45.1802511,9.1591663, 'Borromeo'),(45.1877192,9.1578934, 'Cairoli'),
      (45.1870975,9.1588276, 'Castiglioni'),(45.1871301,9.1435067, 'Santa Caterina'),
      (45.1863927,9.15947, 'Ghislieri'),(45.2007148,9.1325475, 'Nuovo'),
      (45.1787292,9.1635482, 'Cardano'),(45.1864928,9.1560687, 'Fraccaro'),
      (45.1989668,9.1775168, 'Griziotti'),(45.1838819,9.161318, 'Spallanzani'),
      (45.1823523,9.1454315, 'Valla'),(45.2007816,9.1341354, 'Volta'),
      (45.2070857,9.1382623, 'Residence Campus'),(45.2070857,9.1382623, 'Residenza Biomedica')]
 
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

    


def PlotTour(Ps, Ls, values):
    # Report solution value
    import pylab as pl
    from matplotlib import collections  as mc

    lines = [[Ps[i], Ps[j]] for i,j in Ls]

    lc = mc.LineCollection(lines, linewidths=[1.5
                                              if x > 0.501 else 1 for x in values],
                           colors=['blue' if x > 0.501 else 'orange' for x in values])
    
    fig, ax = pl.subplots()
    ax.add_collection(lc)
    ax.scatter([i for i,j in Ps], [j for i,j in Ps], 
                s=20, alpha=0.8, color='red')
    
    ax.autoscale()
    ax.margins(0.1)
    ax.set_aspect('equal', 'box')

    # pl.savefig('tspFrac.pdf')
    pl.show()


def CostMatrix(Ls):
    n = len(Ls)
    C = 100000*np.ones((n,n))
    for i, (a,b) in enumerate(Ls):
        for j, (c,d) in enumerate(Ls[i+1:]):
            C[i, i+j+1] = sqrt((a-c)**2 + (b-d)**2)
            C[i+j+1, i] = C[i, i+j+1]
            
    return C
 
    
def RandomTSP(n):
    from numpy import random
    random.seed(13)
    return [(x,y) for x,y in zip(random.random(n), random.random(n))]

def BuildDiGraph(C):
    # Build a directed graph out of the data
    G = nx.DiGraph()
    
    n,n = C.shape
    for i in range(1,n+1):
        for j in range(1,n+1):
            if i != j:
                G.add_edge(i, j, weight=C[i-1,j-1])

    return G


def BuildGraph(C):
    # Build a directed graph out of the data
    G = nx.Graph()
    
    n,n = C.shape
    for i in range(1,n+1):
        for j in range(1,n+1):
            if i < j:
                G.add_edge(i, j, weight=C[i-1,j-1])

    return G


# Mixed Integer Programming Formulation
def TSP(G, TIME_LIMIT=600):
    # Number of places
    n = G.number_of_nodes()
    
    # TODO: Implement the model of your choice
    m = ConcreteModel()
    
    # 1. Data and ranges
    m.N = RangeSet(n)
    
    m.A = Set(initialize=((i,j) for i,j in G.edges()), dimen=2) 
    
    # 2. Variables    
    # TODO: introduce only usefull variables (no arc, no variable)
    m.x = Var(m.A, domain=NonNegativeReals, bounds=lambda m: (0,1))

    # 3. Objective function
    # Objective function of arc variables
    m.obj = Objective(expr = sum(G[i][j]['weight']*m.x[i,j] for i,j in G.edges()))

    # 4. Constraints    
    # Vincoli archi uscenti 
    m.outdegree = ConstraintList()
    for i in m.N:
        m.outdegree.add(expr = sum(m.x[v,w] for v,w in G.out_edges(i)) == 1)

    # Vincoli archi entranti
    m.indegree = ConstraintList()
    for j in m.N:
        m.indegree.add(expr = sum(m.x[v,w] for v,w in G.in_edges(j)) == 1)


    # Arc constraint
    m.arcs = ConstraintList()
    for i,j in G.edges():
        if i < j:
             m.arcs.add( m.x[i,j] + m.x[j,i] <= 1 )              
             
    m.subtour = ConstraintList()
    
    solver = SolverFactory('gurobi')
        
    # 5. Solution
    # Solve the model
    SOLVER_NAME = 'gurobi'
    # SOLVER_NAME = 'glpk'
    
    solver = SolverFactory(SOLVER_NAME)
    
    if SOLVER_NAME == 'glpk':         
        solver.options['tmlim'] = TIME_LIMIT
    elif SOLVER_NAME == 'gurobi':           
        solver.options['TimeLimit'] = TIME_LIMIT
    
    it = 0
    Cold = []
    while it <= 100:
        it += 1
        sol = solver.solve(m, tee=False, load_solutions=False)
        
        # Get a JSON representation of the solution
        sol_json = sol.json_repn()
        # Check solution status
        if sol_json['Solver'][0]['Status'] != 'ok':
            return None, []
    
        # Load the solution    
        m.solutions.load_from(sol)   

        print(it, m.obj())
        
        selected = []
        values = []
        for i in m.N:
            for j in m.N:
                if i < j:
                    if m.x[i,j]() > 0 or  m.x[j,i]() > 0:
                        selected.append( (i-1, j-1) )
                        values.append(m.x[i,j]()+m.x[j,i]())
        
        PlotTour(Ls, selected, values) 
        
        # Build graph
        H = nx.Graph()
        
        for i in m.N:
            for j in m.N:
                if i < j:
                    if m.x[i,j]() > 0.00001 or m.x[j,i]() > 0.00001:
                        H.add_edge(i,j, weight=m.x[i,j])
            
        Cs = nx.cycle_basis(H)

        
        if Cs != Cold:
            Cold = Cs
            for cycle in Cs:
                Es = []
                for i in cycle:
                    for j in G.nodes():
                        if j not in cycle:
                            Es.append( (i,j) )
    
                if len(Es) > 0:
                    m.subtour.add( sum(m.x[i,j] for i,j in Es ) >= 1 )
        else:
            break
       
    selected = []
    values = []
    for i in m.N:
        for j in m.N:
            if i < j:
                if m.x[i,j]() > 0 or  m.x[j,i]() > 0:
                    selected.append( (i-1, j-1) )
                    values.append(m.x[i,j]()+m.x[j,i]())
    
    PlotTour(Ls, selected, values) 
             
               
    return m.obj(), selected


# Mixed Integer Programming Formulation
def TSPSYM(G, TIME_LIMIT=600):
    # Number of places
    n = G.number_of_nodes()
    
    # TODO: Implement the model of your choice
    m = ConcreteModel()
    
    # 1. Data and ranges
    m.N = RangeSet(n)
    
    m.A = Set(initialize=((i,j) for i,j in G.edges()), dimen=2) 
    
    # 2. Variables    
    # TODO: introduce only usefull variables (no arc, no variable)
    m.x = Var(m.A, domain=NonNegativeReals, bounds=lambda m: (0,1))

    # 3. Objective function
    # Objective function of arc variables
    m.obj = Objective(expr = sum(G[i][j]['weight']*m.x[i,j] for i,j in m.A))

    # 4. Constraints    
    # Vincoli archi uscenti 
    m.degree = ConstraintList()
    for i in m.N:
        Es = []
        for v,w in G.edges(i):
            if v > w:
                v, w = w, v
            Es.append( (v,w) )
        m.degree.add(expr = sum(m.x[v,w] for v, w in Es) == 2)

    m.subtour = ConstraintList()
    
    solver = SolverFactory('gurobi')
        
    # 5. Solution
    # Solve the model
    SOLVER_NAME = 'gurobi'
    # SOLVER_NAME = 'glpk'
    
    solver = SolverFactory(SOLVER_NAME)
    
    if SOLVER_NAME == 'glpk':         
        solver.options['tmlim'] = TIME_LIMIT
    elif SOLVER_NAME == 'gurobi':           
        solver.options['TimeLimit'] = TIME_LIMIT
    
    it = 0
    Cold = []
    while it <= 100:
        it += 1
        sol = solver.solve(m, tee=False, load_solutions=False)
        
        # Get a JSON representation of the solution
        sol_json = sol.json_repn()
        # Check solution status
        if sol_json['Solver'][0]['Status'] != 'ok':
            return None, []
    
        # Load the solution    
        m.solutions.load_from(sol)   

        selected = []
        values = []
        for i, j in m.A:
            if m.x[i,j]() > 0:
                selected.append( (i-1, j-1) )
                values.append(m.x[i,j]())
        
        PlotTour(Ls, selected, values) 
        
        # Build graph
        H = nx.Graph()
        
        for i,j in m.A:
            H.add_edge(i,j, weight=m.x[i,j]())
            
        # Cs = nx.connected_components(H)
        # Cs = list(Cs)
        # Cs = nx.cycle_basis(H)
        
        cut_value, S = nx.stoer_wagner(H)
        print(it, m.obj(), sum(values))
        flag = True

        if cut_value >= 2:
            print(cut_value)
            # Separate blossom
            H = nx.Graph()
            for i,j in m.A:
                if m.x[i,j]() > 0.1 and m.x[i,j]() < 0.9:
                    if i < j:
                        H.add_edge(i, j)
                    else:
                        H.add_edge(j, i)
                        
            selected = []
            values = []
            for i,j in m.A:
                selected.append( (i-1, j-1) )
                values.append(m.x[i,j]())
            Cs = nx.cycle_basis(H)
            for cycle in Cs:
                NS = len(cycle)                      
                if NS == 3:
                    S = set()
                    for i in range(NS):
                        if cycle[i-1] < cycle[i]:
                            S.add( (cycle[i-1], cycle[i]) )
                        else:
                            S.add( (cycle[i], cycle[i-1]) )
                    
                    for i in cycle:
                        for j in G.neighbors(i):
                            if (i,j) not in S:
                                v,w = i,j
                                if i > j:
                                    v,w = j,i
                                if m.x[v,w]() > 0.9:
                                    S.add( (v,w) )

                    if False and len(S) > NS+2:
                        m.subtour.add( sum(m.x[i,j] for i,j in S ) <= NS+1 )
                        flag = False
                        print('added', S)
                            
        else:  
            Es = []
            for i in S[0]:
                for j in S[1]:
                    if i < j:
                        Es.append( (i,j) )
                    else:
                        Es.append( (j,i) )
    
            if len(Es) > 0:
                m.subtour.add( sum(m.x[i,j] for i,j in Es ) >= 2 )
                flag = False
        if flag:
            break
  
        # sleep(1)
        
        # if Cs == Cold:
        #     break
        
        # Cold = Cs
        # for cycle in Cs:
        #     Es = []
        #     for i,j in m.A:
        #         if (i in cycle and j not in cycle) or (i not in cycle and j in cycle):
        #             if i < j:
        #                 Es.append( (i,j) )
        #             else:
        #                 Es.append( (j,i) )
        #             Es.append( (i,j) )
    
        #     if len(Es) > 0:
        #         m.subtour.add( sum(m.x[i,j] for i,j in Es ) >= 2 )
       
    selected = []
    values = []
    for i,j in m.A:
        if m.x[i,j]() > 0:
            selected.append( (i-1, j-1) )
            values.append(m.x[i,j]())
    print(values)
    PlotTour(Ls, selected, values) 
             
               
    return m.obj(), selected

# -----------------------------------------------
#   MAIN function
# -----------------------------------------------
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
    if True:
        G = BuildGraph(C)
        z_lp, tour = TSPSYM(G)
    if False:
        G = BuildDiGraph(C)
        z_lp, tour = TSP(G)