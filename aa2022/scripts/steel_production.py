# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 11:15:51 2022

@author: Gualandi
"""

from pyomo.environ import ConcreteModel, Var, Objective, Constraint, SolverFactory, NonNegativeReals, maximize

model = ConcreteModel()

# Homework. Data
pB = 25
pC = 30

rB = 200
rC = 140

dB = 6000
dC = 4000

T = 40

# Declare decision variables
model.xB = Var(domain=NonNegativeReals, bounds=(0,dB))
model.xC = Var(domain=NonNegativeReals, bounds=(0,dC))

# Declare objective
model.cost = Objective(
    expr = pB*model.xB + pC*model.xC,
    sense = maximize)

# Declare constraints
model.cnstr1 = Constraint(expr = 1/rB*model.xB + 1/rC*model.xC <= T)

# Solve
sol = SolverFactory('glpk').solve(model, tee=True)

# Basic info about the solution process
for info in sol['Solver']:
    print(info)
    
# Report solution value
print("Optimal solution value: z =", model.cost())
print("Decision variables:")
print("\tProduction of bands:", model.xB())
print("\tProduction of coils:", model.xC())

