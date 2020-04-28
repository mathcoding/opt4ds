# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 13:50:28 2020

EXERCISE OBJECTIVE: IMPLEMENT A TEXTBOOK PRIMAL SIMPLEX WITH DICTIONARIES

@author: Gualandi
"""

import numpy as np

Inf = 1e+10


class LinearProblem(object):
    def __init__(self, c, A, b):
        pass

        
    def getObj(self):
        return Inf


    def checkOptimality(self):
        return True

        
    def selectEntering(self):
        return -1

    
    def selectLeaving(self, k):
        return -1
    
    
    def pivoting(self, k, h):
        pass
    
    
    def solve(self):
        pass

    
    def __str__(self):
        return str(self.D)
    
    
# -----------------------------------------------
#   MAIN function
# -----------------------------------------------
if __name__ == "__main__":
    # Cost
    c = [2, 1]
    # Constraint Coefficients
    A = [[2, 1],
         [2, 3],
         [4, 1],
         [1, 5]]
    # R.H.S.
    b = [4, 3, 5, 1]
    
    # LP problem
    # P = LinearProblem(c, A, b)
    # print(P)
    
    # print('Is optimal?', P.checkOptimality())
    # k = P.selectEntering()
    # print('k =', k)
    # h = P.selectLeaving(k)
    # print('h =', h)