# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 11:12:06 2021

@author: gualandi
"""

import pyomo

from pyomo.environ import ConcreteModel, Var, Objective, Constraint, SolverFactory
from pyomo.environ import maximize, Binary, RangeSet, PositiveIntegers, NonNegativeReals

# Problem Description
"""
The industrial steel can be easily recycled, since it is possible to burn any
scrap to get only liquid steel (without plastics, glasses, ...).
However, it is hard to separate each single metal presents in the scrap,
and as a consequence, beside iron, we can get also chromium, nichel, and
other impurities in the liquid steel.
Depending on the type of production, some metals are desiderable, while others
are not. For example, the stainless steel 18/10 must have 18% of chromium and
10% of nichel (consider that chromium and nichel are very expensive, much more 
than the steel itself). 

Suppose that the Rossi's Steel company of Voghera can choose to buy some iron
scrap block with different properties regarding the different metals contained in 
each block. The company want to produce at minimum cost 100 quintals of stainless 
steel 18/10, which must have at least 65% of iron and at most 1% of 
impurity materials. Which fraction of each block is going to buy?

The data of the problem are given below.

"""

# Data of the problem (in theory, read data from .csv or excel file)

# Blocks you can byu
Blocks = ['Block1', 'Block2', 'Block3', 'Block4', 'Block5', 'Block6']

Weights = [30, 90, 50, 70, 60, 50]  # In quintal
Costs = [50, 100, 80, 85, 92, 115]  # Thousand of euros

# Componets of metal in each block (given in percetange)
Cs = [
    [93, 76, 74, 65, 72, 68],  # Ferro
    [5, 13, 11, 16, 6, 23],  # Cromo
    [0, 11, 12, 14, 20, 8],  # Nichel
    [2, 0, 3, 5, 2, 1]
]  # Impurità

# Create concrete model
m = ConcreteModel()

# Set of indices
m.I = RangeSet(0, len(Blocks) - 1)


# Variables
def fb(m, i):
    return 0, Weights[i]


m.x = Var(m.I, domain=NonNegativeReals, bounds=fb)

# Objective Function
m.obj = Objective(expr=sum(Costs[i] * m.x[i] for i in m.I))

# Production Constraints
m.c1 = Constraint(expr=sum(Cs[0][i] / 100 * m.x[i] for i in m.I) >= 65)

m.c2 = Constraint(expr=sum(Cs[1][i] / 100 * m.x[i] for i in m.I) == 18)

m.c3 = Constraint(expr=sum(Cs[2][i] / 100 * m.x[i] for i in m.I) == 10)

m.c4 = Constraint(expr=sum(Cs[3][i] / 100 * m.x[i] for i in m.I) <= 1)

# Overall production
m.c5 = Constraint(expr=sum(m.x[i] for i in m.I) == 100)

# Write the LP model in standard format
m.write("misc.lp")

# Solve the model
sol = SolverFactory('glpk').solve(m, tee=True)
#sol = SolverFactory('gurobi').solve(m, tee=True)

# CHECK SOLUTION STATUS

# Get a JSON representation of the solution
sol_json = sol.json_repn()

if sol_json['Solver'][0]['Status'] == 'ok':
    print("Optimal solution value:", round(m.obj(), 1))

    print("\tValues of the decision variables:")
    for i, b in enumerate(Blocks):
        print(b, m.x[i]())
else:
    print('Error in solving the model')
