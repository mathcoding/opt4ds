# -*- coding: utf-8 -*-
"""
Solution to Exercise 2.5: Steel Recycle Bleending Problem (notebook: Python and Gurobi)

@author: Gualandi
"""

import numpy as np

# Data of the problem (in theory, read data from .csv or excel file)

# Blocks you can buy
Blocks = ['Block1','Block2','Block3','Block4','Block5','Block6']

Weights = [30, 90, 50, 70, 60, 50]  # In quintals
Costs = [50, 100, 80, 85, 92, 115]  # Thousand of euros

# Components of metal in each block (given in percentage)
Cs = np.matrix([[93, 76, 74, 65, 72, 68],  # Ferro
                [5, 13, 11, 16, 6, 23],    # Cromo
                [0, 11, 12, 14, 20, 8],    # Nichel
                [2, 0, 3, 5, 2, 1]])       # Impurità


from gurobipy import Model, GRB, quicksum, Env

# LP model        
model = Model()

# Variables and objective function
x = [model.addVar(ub=Weights[i], obj=0.0) for i, c in enumerate(Costs)]

z = [model.addVar(vtype=GRB.BINARY, obj=c*Weights[i]) for i, c in enumerate(Costs)]

# Constraints
model.addConstr(quicksum(x[i] for i in range(len(x))) == 100)

# Constraints for the components
model.addConstr(quicksum(Cs[0, i]*x[i]/100 for i in range(len(x))) >= 65)
model.addConstr(quicksum(Cs[1, i]*x[i]/100 for i in range(len(x))) == 18)
model.addConstr(quicksum(Cs[2, i]*x[i]/100 for i in range(len(x))) == 10)
model.addConstr(quicksum(Cs[3, i]*x[i]/100 for i in range(len(x))) <= 1)

# Logical constraints
for i in range(len(x)):
    model.addConstr(x[i] <= Weights[i]*z[i])

model.optimize()

print('Status:', model.status)
print('Objective value:', model.objVal)
print('Solution x:', [x[i].x for i in range(len(x))])
print('Solution y:', [z[i].x for i in range(len(x))])
