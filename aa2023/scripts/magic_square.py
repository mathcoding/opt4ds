# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 09:53:57 2020

@author: Gualandi
"""

from pyomo.environ import ConcreteModel, Var, Objective, Constraint, SolverFactory
from pyomo.environ import maximize, Binary, RangeSet, PositiveIntegers


def MagicSquareSolver(n):
    # Create concrete model
    model = ConcreteModel()

    # Set of indices
    model.I = RangeSet(1, n)
    model.J = RangeSet(1, n)
    model.K = RangeSet(1, n * n)

    # Variables
    model.z = Var(within=PositiveIntegers)
    model.x = Var(model.I, model.J, model.K, within=Binary)

    # Objective Function
    model.obj = Objective(expr=model.z)

    # Every number "k" can appear in a single cell "(i,j)"
    def Unique(model, k):
        return sum(model.x[i, j, k] for j in model.J for i in model.I) == 1

    model.unique = Constraint(model.K, rule=Unique)

    # Every cell "(i,j)" can contain a single number "k"
    def CellUnique(model, i, j):
        return sum(model.x[i, j, k] for k in model.K) == 1

    model.cellUnique = Constraint(model.I, model.J, rule=CellUnique)

    # The sum of the numbers over each row "i" must be equal to "z"
    def Row(model, i):
        return sum(k * model.x[i, j, k] for j in model.J
                   for k in model.K) == model.z

    model.row = Constraint(model.I, rule=Row)

    # The sum of the numbers over a column "j" must be equal to "z"
    def Col(model, j):
        return sum(k * model.x[i, j, k] for i in model.I
                   for k in model.K) == model.z

    model.column = Constraint(model.J, rule=Col)

    # The sum over the main diagonal and the off-diagonal must be equal
    model.diag1 = Constraint(expr=sum(k * model.x[i, i, k] for i in model.I
                                      for k in model.K) == model.z)
    model.diag2 = Constraint(expr=sum(k * model.x[i, n - i + 1, k]
                                      for i in model.I
                                      for k in model.K) == model.z)

    # Write the LP model in standard format
    model.write("magic_{}.lp".format(n))

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

    return model.x


def PlotMagicSquare(x, n):
    # Report solution value
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    sol = np.zeros((n, n), dtype=int)

    for i, j, k in x:
        if x[i, j, k]() > 0.5:
            sol[i - 1, j - 1] = k

    cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(6, 6))
    plt.imshow(sol, interpolation='nearest', cmap=cmap)
    plt.title("Magic Square, Size: {}".format(n))
    plt.axis('off')

    for i, j in itertools.product(range(n), range(n)):
        plt.text(j,
                 i,
                 "{:d}".format(sol[i, j]),
                 fontsize=24,
                 ha='center',
                 va='center')

    plt.tight_layout()
    plt.show()


# -----------------------------------------------
#   MAIN function
# -----------------------------------------------
if __name__ == "__main__":

    # Solve Magic Square of size 4
    n = 4

    x = MagicSquareSolver(n)
    if x:
        PlotMagicSquare(x, n)
    else:
        print("Unable to solve Magic Square of size", n)
