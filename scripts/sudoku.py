# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 09:07:35 2020

@author: Gualandi
"""

from pyomo.environ import ConcreteModel, Var, Objective, SolverFactory
from pyomo.environ import Binary, RangeSet, ConstraintList

def SudokuSolver(Data):
    # Create concrete model
    model = ConcreteModel()
    
    # Sudoku of size 9x9, with subsquare 3x3
    n = 9
    model.I = RangeSet(1, n)
    model.J = RangeSet(1, n)
    model.K = RangeSet(1, n)
    
    # Variables
    model.x = Var(model.I, model.J, model.K, within=Binary)
    
    # Objective Function
    model.obj = Objective(
        expr = sum(model.x[i,j,k] for i in model.I for j in model.J for k in model.K))
    
    # 1. A single digit for each position
    model.unique = ConstraintList()
    for i in model.I:
        for j in model.J:
            expr = 0
            for k in model.K:
                expr += model.x[i,j,k]
            model.unique.add( expr == 1 )
    
    # 2. Row constraints
    model.rows = ConstraintList()
    for i in model.I:
        for k in model.K:
            expr = 0
            for j in model.J:
                expr += model.x[i,j,k]
            model.rows.add( expr == 1 )
    
    # 3. Column constraints
    model.columns = ConstraintList()
    for j in model.J:
        for k in model.K:
            expr = 0
            for i in model.I:
                expr += model.x[i,j,k]
            model.columns.add( expr == 1 )
                    
    # 4. Submatrix constraints
    model.blocks = ConstraintList()
    S = [1, 4, 7]
    for i0 in S:
        for j0 in S:
            for k in model.K:
                expr = 0
                for i in range(3):
                    for j in range(3):
                        expr += model.x[i0+i, j0+j,k]
                model.blocks.add( expr == 1 )
       
    # 5. Fix input data
    for i in range(n):
        for j in range(n):
            if Data[i][j] > 0:
                model.x[i+1,j+1,Data[i][j]].fix(1)
                
    # Solve the model
    sol = SolverFactory('glpk').solve(model, tee=True)
    
    # CHECK SOLUTION STATUS
    
    # Get a JSON representation of the solution
    sol_json = sol.json_repn()
    # Check solution status
    if sol_json['Solver'][0]['Status'] != 'ok':
        return None
    if sol_json['Solver'][0]['Termination condition'] != 'optimal':
        return None
    
    # Print useful information of the solution
    print("objective value:", model.obj())

    for i in model.I:
        for j in model.J:
            for k in model.K:
                if model.x[i,j,k]() > 0:
                    print(k, end="  ")
        print()
    
    return model.x

    
def PlotSudoku(x, size=6):
    import matplotlib.pyplot as plt
    import numpy as np

    boardgame = np.zeros((9, 9))

    plt.figure(figsize=(size, size))
    plt.imshow(boardgame, cmap='binary')

    for i, j, k in x:
        if x[i,j,k]() > 0:
            if Data[i-1][j-1] == k:
                plt.text(i-1, j-1, k, fontsize=4*size, color='red',
                     ha='center', va='center')
            else:                
                plt.text(i-1, j-1, k, fontsize=4*size, color='darkblue',
                         ha='center', va='center')
             
    # Prettify output
    for i in range(9):
        plt.axhline(y=i+0.5, color='grey', linestyle='--', alpha=0.5)
        plt.axvline(x=i+0.5, color='grey', linestyle='--', alpha=0.5)
    for i in range(3):
        plt.axhline(y=i*3+2.5, color='grey', linestyle='-', lw=2)
        plt.axvline(x=i*3+2.5, color='grey', linestyle='-', lw=2)

    plt.xticks([])
    plt.yticks([])
    plt.show()
    
    
# -----------------------------------------------
#   MAIN function
# -----------------------------------------------
if __name__ == "__main__":

    # INPUT DATA
    Data= [[0, 0, 0, 0, 9, 4, 8, 0, 0],
            [0, 2, 0, 0, 1, 7, 5, 0, 0],
            [0, 0, 6, 0, 0, 0, 0, 1, 0],
            [0, 6, 2, 0, 0, 8, 0, 0, 7],
            [0, 0, 0, 3, 0, 2, 0, 0, 0],
            [3, 0, 0, 9, 0, 0, 4, 2, 0],
            [0, 9, 0, 0, 0, 0, 6, 0, 0],
            [0, 0, 1, 7, 8, 0, 0, 9, 0],
            [0, 0, 3, 4, 5, 0, 0, 0, 0]]

    x = SudokuSolver(Data)
    
    # If x is "None" there is no feasible solution
    if x:
        PlotSudoku(x)
    else:
        print("Error in solving the problem. Is the problem feasible?")