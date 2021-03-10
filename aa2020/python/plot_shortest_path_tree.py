# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 11:14:53 2020

@author: Gualandi
"""

import networkx as nx
import matplotlib.pyplot as plt

Ls = [('a', 'b', 5), ('a', 'c', 3), ('a', 'd', 3), ('b', 'c', 2), 
      ('b', 'd', 5), ('c', 'e', 2), ('c', 'd', 3), ('d', 'e', 2), 
      ('d', 'f', 3), ('e', 'g', 3), ('f', 'c', 4), ('g', 'f', 2)]

Cs = dict([((i,j),c) for i,j,c in Ls])
As = [(i,j) for i,j,_ in Ls]

# NetworkX Digraph
G = nx.DiGraph()
G.add_edges_from(As)

val_map = {'g': 0.5714285714285714,
           'a': 0.0}

values = [val_map.get(node, 0.2) for node in G.nodes()]

# Specify the edges you want here
red_edges = [('e', 'g'), ('b', 'c'), ('c', 'e'), ('f', 'c'), ('d', 'e'), ('a', 'c')]
black_edges = [edge for edge in G.edges() if edge not in red_edges]

# Need to create a layout when doing
# separate calls to draw nodes and edges
pos = nx.kamada_kawai_layout(G)

nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('coolwarm'), 
                       node_color = values, node_size = 400)
nx.draw_networkx_labels(G, pos)

nx.draw_networkx_edges(G, pos, edgelist=red_edges, lw=2,
                       edge_color='r', arrows=True)
nx.draw_networkx_edges(G, pos, edgelist=black_edges, arrows=True)
nx.draw_networkx_edge_labels(G, pos, edge_labels=Cs)

plt.savefig("ShortestPathGraph.pdf", bbox_inches='tight')
plt.show()
