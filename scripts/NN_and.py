import numpy as np
import matplotlib.pyplot as plt

from pyomo.environ import ConcreteModel, Var, Objective, Constraint, SolverFactory
from pyomo.environ import maximize, Binary, RangeSet, Reals, Set, ConstraintList, PositiveReals

def MIPand(Xs, Ys):
    # Pyomo Model object
    mod = ConcreteModel()

    mod.K = RangeSet(0, len(Xs)-1)
    mod.I = RangeSet(0, len(Xs[0])-1)

    # Decision variables
#    mod.W = Var(mod.I, within=Reals)
    mod.W = Var(mod.I, within=Binary)

    mod.S = Var(mod.K, within=Binary)

    mod.alpha = Var(mod.K, within=PositiveReals)

    # Objective function
    mod.obj = Objective(expr=sum(mod.alpha[k] for k in mod.K))

    # Constraints
    mod.abs = ConstraintList()
    for k in mod.K:
        mod.abs.add( +Ys[k] - (1 - 2*mod.S[k]) <= mod.alpha[k])
        mod.abs.add( -Ys[k] + (1 - 2*mod.S[k]) <= mod.alpha[k])

    M = 1000
    def OutputCon(m, k):
        if Ys[k] == 1:
            return sum((1 - 2*m.W[i])*Xs[k][i] for i in m.I) >= 0.01 - M*mod.S[k]
        return sum((1 - 2*m.W[i])*Xs[k][i] for i in m.I) <= -0.01 + M*(1 - mod.S[k])
#        if Ys[k] == 1:
#            return sum(m.W[i]*Xs[k][i] for i in m.I) >= 0.01 - M*mod.S[k]
#        return sum(m.W[i]*Xs[k][i] for i in m.I) <= -0.01 + M*(1 - mod.S[k])
    mod.output = Constraint(mod.K, rule = OutputCon)

    # Solve the model
    sol = SolverFactory('gurobi').solve(mod, tee=True)

    # Get a JSON representation of the solution
    sol_json = sol.json_repn()
    # Check solution status
    if sol_json['Solver'][0]['Status'] != 'ok':
        return None, None
    if sol_json['Solver'][0]['Termination condition'] != 'optimal':
        return None, None

    # Return the final solution
    return [(1- 2*mod.W[i]()) for i in mod.I]


#-----------------------------------------------
# MAIN function
#-----------------------------------------------
if __name__ == "__main__":
    from numpy.random import seed
    #seed(13)

    Xand = [(1, -1, -1), (1, -1, +1), (1, +1, -1), (1, +1, +1)]
    Yand = [-1, +1, +1, -1]

    W = MIPand(Xand, Yand)

    print('optimal weight:', W)
    # To verify the model 
    for x, y in zip(Xand, Yand):
        yhat = +1 if sum(W[i]*x[i] for i in range(3)) >= 0 else -1
        print(y, yhat)