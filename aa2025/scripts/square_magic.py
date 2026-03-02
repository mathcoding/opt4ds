# -*- coding: utf-8 -*-
"""
Solution to Exercise 2.6: Magic Square

@author: Gualandi
"""

from gurobipy import Model, GRB, quicksum
import numpy as np

def MagicSquare(n):
    model = Model()

    # Variables x_ijk
    z = model.addVar(vtype=GRB.INTEGER, name="z")

    x = {}
    for i in range(n):
        for j in range(n):
            for k in range(n*n):
                x[i, j, k] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}_{k}")

    # Constraints: unique value in each position
    for k in range(n*n):
        model.addConstr(quicksum(x[i, j, k] for i in range(n) for j in range(n)) == 1)
    
    # Constraints: every cell contains exactly one value
    for i in range(n):
        for j in range(n):
            model.addConstr(quicksum(x[i, j, k] for k in range(n*n)) == 1)

    # Sum over each row
    for i in range(n):
        model.addConstr(quicksum(k*x[i, j, k] for j in range(n) for k in range(n*n)) == z)

    # Sum over each column
    for j in range(n):
        model.addConstr(quicksum(k*x[i, j, k] for i in range(n) for k in range(n*n)) == z)

    # Main diagonal
    model.addConstr(quicksum(k*x[i, i, k] for i in range(n) for k in range(n*n)) == z)

    # Anti-diagonal
    model.addConstr(quicksum(k*x[i, n-i-1, k] for i in range(n) for k in range(n*n)) == z)

    # Solve model
    model.optimize()

    if model.Status == GRB.OPTIMAL:
        S = np.zeros((n, n), dtype=int)
        for i in range(n):
            for j in range(n):
                S[i, j] = sum(k*x[i, j, k].X for k in range(n*n))

        return S
    else:
        return None


def PlotMagicSquare(sol):
    # Report solution value
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    n,n = sol.shape

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
    n = 6

    S = MagicSquare(n)
    PlotMagicSquare(S)
    