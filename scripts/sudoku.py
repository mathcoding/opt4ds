# -*- coding: utf-8 -*-
"""
@author: Gualandi
"""
from gurobipy import Model, GRB, quicksum
import numpy as np

def SudokuSolver(Data):
    # Create concrete model
    model = Model()
    
    # Sudoku of size 9x9, with subsquare 3x3
    N = 9
    n = 3
    
    # Variables
    
    # Objective Function
    
    # 1. A single digit for each position
    
    # 2. Row constraints
    
    # 3. Column constraints
                    
    # 4. Submatrix constraints
       
    # 5. Fix input data

    # Solve model
    model.optimize()

    # if model.Status == GRB.OPTIMAL:
    #     S = np.zeros((N, N), dtype=int)
    #     for i in range(N):
    #         for j in range(N):
    #             S[i, j] = sum(k*x[i, j, k].X for k in range(N))+1

    #     for i in range(N):
    #          for j in range(N):
    #              for k in range(N):
    #                  if x[i,j,k].X > 0:
    #                      print(k+1, end="  ")
    #          print()
         sol = {}
        # for i in range(N):
        #      for j in range(N):
        #          for k in range(N):
        #                 if x[i,j,k].X > 0:
        #                     sol[i,j,k] = k+1
        return sol
    else:
        return None
    
def PlotSudoku(x, size=6):
    import matplotlib.pyplot as plt
    import numpy as np

    boardgame = np.zeros((9, 9))

    plt.figure(figsize=(size, size))
    plt.imshow(boardgame, cmap='binary')

    for i, j, k in x:
        if x[i,j,k] > 0:
            if Data[i][j] == k:
                plt.text(i, j, k, fontsize=4*size, color='red',
                     ha='center', va='center')
            else:                
                plt.text(i, j, k, fontsize=4*size, color='darkblue',
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

    sol = SudokuSolver(Data)
    print(sol)
    # If x is "None" there is no feasible solution
    if sol is not None:
        PlotSudoku(sol)
    else:
        print("Error in solving the problem. Is the problem feasible?")