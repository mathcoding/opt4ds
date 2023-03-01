# -*- coding: utf-8 -*-
"""
Created on Mon May 25 19:08:09 2020

@author: Gualandi
"""

import numpy as np
import time
from math import sqrt

# Residenza Collegiali a Pavia
Rs = [(45.1882789,9.1600456, 'Del Maino'),(45.2070857,9.1382623, 'Green Campus'),
      (45.1961107,9.1395709, 'Golgi'),(45.1851618,9.1506323, 'Senatore'),
      (45.1806049,9.1691651, 'Don Bosco'),(45.1857651,9.1473637, 'CSA'),
      (45.1802511,9.1591663, 'Borromeo'),(45.1877192,9.1578934, 'Cairoli'),
      (45.1870975,9.1588276, 'Castiglioni'),(45.1871301,9.1435067, 'Santa Caterina'),
      (45.1863927,9.15947, 'Ghislieri'),(45.2007148,9.1325475, 'Nuovo'),
      (45.1787292,9.1635482, 'Cardano'),(45.1864928,9.1560687, 'Fraccaro'),
      (45.1989668,9.1775168, 'Griziotti'),(45.1838819,9.161318, 'Spallanzani'),
      (45.1823523,9.1454315, 'Valla'),(45.2007816,9.1341354, 'Volta'),
      (45.2070857,9.1382623, 'Residence Campus'),(45.2070857,9.1382623, 'Residenza Biomedica')]

ULYSSES = [(38.24, 20.42), (39.57, 26.15), (40.56, 25.32), (36.26, 23.12),
           (33.48, 10.54), (37.56, 12.19), (38.42, 13.11), (37.52, 20.44),
           (41.23, 9.10), (41.17, 13.05), (36.08, -5.21), (38.47, 15.13), 
           (38.15, 15.35), (37.51, 15.17), (35.49, 14.32), (39.36, 19.56)]


def TourLength(Tour):
    global C
    c = 0
    for i in range(len(Tour)):
        c += C[Tour[i-1], Tour[i]]
    return c

def Swap(Tour, i, j):
    Tour[j], Tour[i] = Tour[i], Tour[j]
    
    
def TourFlip(x, y):
    global N
    while x < y:
        Tour[y%N], Tour[x%N] = Tour[x%N], Tour[y%N]
        x += 1
        y -= 1

def TD(i,j):
    return C[Tour[i%N], Tour[j%N]]


def TwoOpt():
    while True:
        Improve = False
        for a in range(N):
            for c in range(a+2, a+N-2):
                if TD(a, a+1) + TD(c,c+1) > TD(a, c) + TD(a+1, c+1):
                    TourFlip(a+1, c)
                    Improve = True

        if not Improve:
            return                


def NN():
    global Tour
    ll = 0
    Tour = [i for i in range(N)]
    for i in range(1, N):
        best = 2*C.max()
        for j in range(i, N):
            if C[Tour[i-1], Tour[j]] < best:
                best = C[Tour[i-1], Tour[j]]
                bestj = j
        ll += best
        Swap(Tour, i, bestj)
    
    return ll + C[Tour[N-1], Tour[0]]
    

def MST(count):
    global C, Tour, N, HashTable
    if count <= 1: return 0
    ll = 0
    
    MAXCOST = 2*C.max()
    pcity = [Tour[i] for i in range(count)]
    pdist = [MAXCOST for _ in range(count)]

    if count != N:
        count += 1
        pcity.append(Tour[N-1])
        
    tcity = tuple(pcity)
    if tcity in HashTable:
        return HashTable[tcity]
    
    newcity = pcity[count-1]
    m = count - 1
    while m > 0:
        mindist = MAXCOST
        mini = -1
        for i in range(m):
            thisdist = C[pcity[i], newcity]
            if thisdist < pdist[i]:
                pdist[i] = thisdist
            if pdist[i] < mindist:
                mindist = pdist[i]
                mini = i
        
        newcity = pcity[mini]
        ll += mindist
        pcity[mini] = pcity[m-1]
        pdist[mini] = pdist[m-1]
        m -= 1
        
    HashTable[tcity] = ll
    
    return ll
        
    
        
        
def Permute(k, tourlen, cheapsum):
    global C, N, Tour, BestTour, best

    # if tourlen + cheapsum >= best: return
    if tourlen + MST(k+1) >= best: return
    
    if k == 1:
        tourlen += C[Tour[0], Tour[1]] + C[Tour[N-1], Tour[0]]
        if tourlen < best:
            best = tourlen
            print("best sol: ", best)
            BestTour = [i for i in Tour]
    else:
        for i in range(k):
            Swap(Tour, i, k-1)
            Permute(k-1, tourlen + C[Tour[k-1], Tour[k]], 
                    cheapsum - Cheapest[Tour[k-1]])
            Swap(Tour, i, k-1)
            
            
def Permute2(k, tourlen):
    global C, Tour, BestTour, best

    #if tourlen >= best: return
    
    if k == 1:
        tourlen += C[Tour[0], Tour[1]] + C[Tour[N-1], Tour[0]]
        if tourlen < best:
            best = tourlen
            BestTour = [i for i in Tour]
    else:
        for i in range(k):
            Swap(Tour, i, k-1)
            Permute(k-1, tourlen + C[Tour[k-1], Tour[k]])
            Swap(Tour, i, k-1)
            
            
def Permute1(k, tourlen=0):
    global C, Tour, BestTour, best, Cheapest
    if k == 1:
        cost = TourLength(Tour)
        if cost < best:
            best = cost
            print("best sol: ", best)
            BestTour = [i for i in Tour]
    else:
        for i in range(k):
            Swap(Tour, i, k-1)
            Permute1(k-1)
            Swap(Tour, i, k-1)
                

    
def PlotTour(Ps, Tour):
    # Report solution value
    import matplotlib.pyplot as plt
    import numpy as np
    import pylab as pl
    from matplotlib import collections  as mc

    lines = [[Ps[Tour[i-1]], Ps[Tour[i]]] for i in range(len(Tour))]

    lc = mc.LineCollection(lines, linewidths=2)
    fig, ax = pl.subplots()
    ax.add_collection(lc)
    ax.autoscale()
    ax.margins(0.1)
    
    
def CostMatrix(Ls):
    n = len(Ls)
    C = 100000*np.ones((n,n))
    for i, (a,b) in enumerate(Ls):
        for j, (c,d) in enumerate(Ls[i+1:]):
            C[i, i+j+1] = sqrt((a-c)**2 + (b-d)**2)
            C[i+j+1, i] = C[i, i+j+1]
            
    return C
 
    
def RandomTSP(n):
    from numpy import random
    return [(x,y) for x,y in zip(random.random(n), random.random(n))]


# -----------------------------------------------
#   MAIN function
# -----------------------------------------------
if __name__ == "__main__":
    N = 22
    np.random.seed(13)
    
    Ps = RandomTSP(N)    
    # Ps = [(a,b) for a,b,_ in Rs]
    # Ps = ULYSSES

    N = len(Ps)
    C = CostMatrix(Ps)
    
    best = NN()
    TwoOpt()
    best = TourLength(Tour)
    
    HashTable = {}
    
    Tour = [i for i in range(N)]
    c0 = TourLength(Tour)
    
    BestTour = [i for i in range(N)]
    # best = TourLength(BestTour)
    
    
    Cheapest = C.min(0)
    cheapsum = np.sum(Cheapest)
    
    print("LB: {}, UB: {}".format(max(cheapsum, MST(N)), best))
    
    t1 = time.time()
    cost = Permute(N-1, 0, cheapsum) 
    t2 = time.time()
    
    print("N: {}, Best: {:.3f}, Start: {:.3f}, time: {:.2f}"
          .format(N, best, c0, t2-t1))
    
    PlotTour(Ps, BestTour)
    
    
    