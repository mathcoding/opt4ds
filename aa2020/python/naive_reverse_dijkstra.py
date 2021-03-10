# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 10:42:53 2020

@author: Gualandi
"""
import sys

Inf = sys.maxsize


class Digraph(object):
    def __init__(self):
        self.N = set()  # Set of nodes
        self.A = {}     # Dictionary of arcs (i,j) -> c_ij
        
    def add_arc(self, v, w, c):
        """ Store Backward Star (instead of classic Forward Star) """
        self.N.add(v)
        self.N.add(w)
        
        if w not in self.A:
            self.A[w] = {}
        self.A[w][v] = c
        
    def get_cost(self, v, w):
        if w not in self.A or v not in self.A[w]:
            return Inf
        return self.A[w][v]
    
    def __str__(self):
        s = ""
        for w in self.A:
            s = s + "{}: ".format(w)
            for v in self.A[w]:
                s = s +"({},{})#{} ".format(v, w, self.A[w][v])
            s = s + "\n"
        return s
           
    
def ArgMinRemove(Q):
    """ Dummy Queue with a dictionary """
    keys = sorted(Q)
    j, d = keys[0], Q[keys[0]] 
    for v in keys[1:]:
        if Q[v] < d:
            j, d = v, Q[v]
    del Q[j]
    return j
        
        
def ReverseDijkstra(graph, r):
    # Successors
    S = dict([(v, None) for v in graph.N])
    # Labels
    V = dict([(v, Inf) for v in graph.N])
    V[r] = 0
    # Open Nodes
    Fc = dict([(v, V[v]) for v in V])
    # Main Loop
    while len(Fc) > 0:
        print('\nFc:', Fc)
        j = ArgMinRemove(Fc)
        print('argmin{Fc} =', j)
        if j in graph.A:
            for i in graph.A[j]:
                if V[i] > graph.A[j][i] + V[j]:
                    print('update label V[{}] = {}'.format(i, graph.A[j][i]+V[j]))
                    V[i] = graph.A[j][i] + V[j]
                    Fc[i] = graph.A[j][i] + V[j]
                    S[i] = j
                
    return S, V
        
        
# -----------------------------------------------
#   MAIN function
# -----------------------------------------------
if __name__ == "__main__":
    # Static description of arcs
    Ls = [('a', 'b', 5), ('a', 'c', 3), ('a', 'd', 3), ('b', 'c', 2), 
          ('b', 'd', 5), ('c', 'e', 2), ('c', 'd', 3), ('d', 'e', 2), 
          ('d', 'f', 3), ('e', 'g', 3), ('f', 'c', 4), ('g', 'f', 2)]
    
    # Create digraph
    graph = Digraph()
    
    # Add arcs to G
    for v, w, c in Ls:
        graph.add_arc(v, w, c)
        
    print(graph)
    
    S, V = ReverseDijkstra(graph, 'g')
    
    print("Succes:", S)
    print("Labels:", V)
    
    
    
        