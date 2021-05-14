# -*- coding: utf-8 -*-
"""
Created on Thu May 13 14:52:27 2021

@author: gualandi
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.distance import cdist

from pyomo.environ import ConcreteModel, Var, Objective, Constraint, SolverFactory
from pyomo.environ import RangeSet, NonNegativeReals, minimize

from time import perf_counter

from gurobipy import Model, GRB, quicksum

# Set a seed for random sampling
np.random.seed(13)


def LoadImage(filename):
    return None


def ShowImage(A):
    pass


def DisplayCloud(A, samples=1000):
    pass


def PointSamples(A, samples=100):
    return []


def D(a, b):
    return 0


def OptimalTransport(H1, H2):

    return []


def OptimalTransportGurobi(H1, H2):
    return []


# Distanza tra i punti di A e quelli di B, con |A| > |B|
# Restituisce l'indice del vettore più vicino.
def ClosestRGB(A, B):
    pass


# -----------------------------------------------
#   MAIN function
# -----------------------------------------------
if __name__ == "__main__":
    A = LoadImage('borgo.jpg')
    B = LoadImage('notte.jpg')

    print(A.shape)
    print(B.shape)

    # ShowImage(A)
    DisplayCloud(A, 500)
    # DisplayCloud(B, 500)

    # H1 = PointSamples(A, 2000)
    # H2 = PointSamples(B, 2000)

    # t0 = perf_counter()

    # CMAP = OptimalTransportGurobi(H1, H2)

    # print("solve time: ", perf_counter() - t0)

    # n,m,_ = A.shape
    # C = A.reshape(n*m,3)

    # Y = ClosestRGB(C, H1)
    # H4 = np.array([H2[CMAP[i]] for i in Y])
    # H5 = H4.reshape(n,m,3)
    # ShowImage(B)
    # ShowImage(A)
    # ShowImage(H5)
