# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 14:14:01 2021

@author: gualandi
"""

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

from math import sqrt, ceil
from scipy.spatial import Delaunay

import matplotlib.pyplot as pyplot
import time
import math

import pyomo

from pyomo.environ import ConcreteModel, Var, Objective, Constraint, ConstraintList, SolverFactory
from pyomo.environ import maximize, Binary, RangeSet, PositiveIntegers, NonNegativeReals


def ParseFile(filename):
    doc = open(filename, 'r')
    # Salta le prime 3 righe
    for _ in range(3):
        doc.readline()
    # Leggi la dimensione
    n = int(doc.readline().split(' ')[2])
    # sala riga
    doc.readline()
    # Leggi la capacita
    C = int(doc.readline().split(' ')[2])
    # sala riga
    doc.readline()
    # Leggi posizioni
    Ps = {}
    for row in doc:
        row = row.rstrip().split(' ')
        if row[0] == 'DEMAND_SECTION':
            break
        row = list(map(lambda z: int(z), row))
        Ps[row[0]] = (row[1], row[2])
    # Leggi posizioni
    Ds = {}
    for row in doc:
        row = row.rstrip().split(' ')
        if row[0] == 'DEPOT_SECTION':
            break
        row = list(map(lambda z: int(z), row))
        Ds[row[0]] = row[1]
    d = int(next(doc).rstrip())

    return n, C, Ps, Ds, d


def Distance(A, B):
    return math.sqrt((A[0]-B[0])**2 + (A[1] - B[1])**2)

    
def DisegnaSegmento(A, B, ax):
    """ 
    Disegna un segmento nel piano dal punto A a al punto B
    Vedi manuale a: http://matplotlib.org/api/pyplot_api.html
    """
    # Disegna il segmento
    ax.plot([A[0], B[0]], [A[1], B[1]], 'b', lw=0.75)
    # Disegna gli estremi del segmento
    DisegnaPunto(A, ax)
    DisegnaPunto(B, ax)
    

def DisegnaPunto(A, ax):
    """
    Disegna un punto nel piano
    """
    ax.plot([A[0]], [A[1]], 'bo', alpha=0.5)
    
    

def VRP(n, K, C, Ps, Ds, d, F, TimeLimit):
    return 0, []

    
def PlotSolution(Xs, Ws, Es):
    fig, ax = plt.subplots()
    for i,j in Es:
        DisegnaSegmento(Ps[i], Ps[j], ax)
    
    ax.scatter([i for i,j in Xs[1:]], [j for i,j in Xs[1:]], 
                s=Ws[1:], alpha=0.3, cmap='viridis')
    
    for i in range(len(Xs[1:])):
        ax.annotate(str(i), Xs[i])
    
    plt.plot([Xs[0][0]], [Xs[0][1]], marker='s', color='red', alpha=0.5)
    plt.axis('square')
    plt.axis('off')
    
    
#-----------------------------------------------
# MAIN function
#-----------------------------------------------
if __name__ == "__main__":
    # CONTROLLARE DOVE AVETE I FILE DI DATI
    n, C, Ps, Ds, d = ParseFile("../data/E-n23-k3.vrp")
    fobj, Es = VRP(n, 4, C, Ps, Ds, d, Distance, 600)      

    Xs = [p for p in Ps.values()]
    Ws = [w for w in Ds.values()]
    
    PlotSolution(Xs, Ws, Es)
    