from pyomo.environ import ConcreteModel, Var, Objective, Constraint, SolverFactory
from pyomo.environ import maximize, minimize, Binary, RangeSet, ConstraintList, NonNegativeReals, Reals, Integers

# Main Pyomo model
model = ConcreteModel()

model.x1 = Var(within=NonNegativeReals, bounds=(0.5,1))
model.x2 = Var(within=NonNegativeReals, bounds=(0.5,1))
model.y = Var(within=NonNegativeReals)

model.obj = Objective(expr=model.y)

model.c1 = Constraint( expr=model.y <= model.x1 )
model.c2 = Constraint( expr=model.y <= model.x2 )
model.c3 = Constraint( expr=model.x1 + model.x2 <= model.y + 1)

model.c4 = Constraint( expr=2*model.x1 + 2*model.x2 - 1 >= 0.1 - (1+0.1)*(1- model.y ) )
model.c5 = Constraint( expr=2*model.x1 + 2*model.x2 - 1 <= -0.1 + (3+0.1)*model.y)

# Solve the model
sol = SolverFactory('gurobi').solve(model, tee=True, options={'TimeLimit': 600})


print(model.x1(), model.x2(), model.y())