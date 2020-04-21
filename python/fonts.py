# -1- coding: utf-8 -1-
"""
Created on Mon Apr  6 10:57:37 2020

@author: Gualandi
"""

B =[[0,0,0,0,0,0,0,0,0],
	[1,1,1,1,1,0,0,0,0],
	[1,0,0,0,0,1,0,0,0],
	[1,0,0,0,0,1,0,0,0],
	[1,0,0,0,0,1,0,0,0],
	[1,0,0,0,0,1,0,0,0],
	[1,1,1,1,1,1,0,0,0],
	[1,0,0,0,0,0,1,0,0],
	[1,0,0,0,0,0,1,0,0],
	[1,0,0,0,0,0,1,0,0],
	[1,0,0,0,0,0,1,0,0],
	[1,1,1,1,1,1,0,0,0],
	[0,0,0,0,0,0,0,0,0],
	[0,0,0,0,0,0,0,0,0],
	[0,0,0,0,0,0,0,0,0],
	[0,0,0,0,0,0,0,0,0],
	[0,0,0,0,0,0,0,0,0]]

U =[[0,0,0,0,0,0,0,0,0],
	[1,0,0,0,0,0,1,0,0],
	[1,0,0,0,0,0,1,0,0],
	[1,0,0,0,0,0,1,0,0],
	[1,0,0,0,0,0,1,0,0],
	[1,0,0,0,0,0,1,0,0],
	[1,0,0,0,0,0,1,0,0],
	[1,0,0,0,0,0,1,0,0],
	[1,0,0,0,0,0,1,0,0],
	[1,0,0,0,0,0,1,0,0],
	[1,0,0,0,0,0,1,0,0],
	[0,1,1,1,1,1,0,0,0],
	[0,0,0,0,0,0,0,0,0],
	[0,0,0,0,0,0,0,0,0],
	[0,0,0,0,0,0,0,0,0],
	[0,0,0,0,0,0,0,0,0]]

O =[[0,0,0,0,0,0,0,0,0],
	[0,1,1,1,1,1,0,0,0],
	[1,0,0,0,0,0,1,0,0],
	[1,0,0,0,0,0,1,0,0],
	[1,0,0,0,0,0,1,0,0],
	[1,0,0,0,0,0,1,0,0],
	[1,0,0,0,0,0,1,0,0],
	[1,0,0,0,0,0,1,0,0],
	[1,0,0,0,0,0,1,0,0],
	[1,0,0,0,0,0,1,0,0],
	[1,0,0,0,0,0,1,0,0],
	[0,1,1,1,1,1,0,0,0],
	[0,0,0,0,0,0,0,0,0],
	[0,0,0,0,0,0,0,0,0],
	[0,0,0,0,0,0,0,0,0],
	[0,0,0,0,0,0,0,0,0]]

N =[[0,0,0,0,0,0,0,0,0],
	[1,0,0,0,0,0,1,0,0],
	[1,0,0,0,0,0,1,0,0],
	[1,1,0,0,0,0,1,0,0],
	[1,0,1,0,0,0,1,0,0],
	[1,0,0,1,0,0,1,0,0],
	[1,0,0,0,1,0,1,0,0],
	[1,0,0,0,0,1,1,0,0],
	[1,0,0,0,0,0,1,0,0],
	[1,0,0,0,0,0,1,0,0],
	[1,0,0,0,0,0,1,0,0],
	[1,0,0,0,0,0,1,0,0],
	[0,0,0,0,0,0,0,0,0],
	[0,0,0,0,0,0,0,0,0],
	[0,0,0,0,0,0,0,0,0],
	[0,0,0,0,0,0,0,0,0]]

A =[[0,0,0,0,0,0,0,0,0],
	[0,0,0,1,0,0,0,0,0],
	[0,0,1,0,1,0,0,0,0],
	[0,1,0,0,0,1,0,0,0],
	[1,0,0,0,0,0,1,0,0],
	[1,0,0,0,0,0,1,0,0],
	[1,0,0,0,0,0,1,0,0],
	[1,1,1,1,1,1,1,0,0],
	[1,0,0,0,0,0,1,0,0],
	[1,0,0,0,0,0,1,0,0],
	[1,0,0,0,0,0,1,0,0],
	[1,0,0,0,0,0,1,0,0],
	[0,0,0,0,0,0,0,0,0],
	[0,0,0,0,0,0,0,0,0],
	[0,0,0,0,0,0,0,0,0],
	[0,0,0,0,0,0,0,0,0]]


P =[[0,0,0,0,0,0,0,0,0],
	[1,1,1,1,1,1,0,0,0],
	[1,0,0,0,0,0,1,0,0],
	[1,0,0,0,0,0,1,0,0],
	[1,0,0,0,0,0,1,0,0],
	[1,0,0,0,0,0,1,0,0],
	[1,1,1,1,1,1,0,0,0],
	[1,0,0,0,0,0,0,0,0],
	[1,0,0,0,0,0,0,0,0],
	[1,0,0,0,0,0,0,0,0],
	[1,0,0,0,0,0,0,0,0],
	[1,0,0,0,0,0,0,0,0],
	[0,0,0,0,0,0,0,0,0],
	[0,0,0,0,0,0,0,0,0],
	[0,0,0,0,0,0,0,0,0],
	[0,0,0,0,0,0,0,0,0]]


S =[[0,0,0,0,0,0,0,0,0],
	[0,1,1,1,1,1,0,0,0],
	[1,0,0,0,0,0,0,0,0],
	[1,0,0,0,0,0,0,0,0],
	[1,0,0,0,0,0,0,0,0],
	[1,0,0,0,0,0,0,0,0],
	[0,1,1,1,1,1,0,0,0],
	[0,0,0,0,0,0,1,0,0],
	[0,0,0,0,0,0,1,0,0],
	[0,0,0,0,0,0,1,0,0],
	[0,0,0,0,0,0,1,0,0],
	[1,1,1,1,1,1,0,0,0],
	[0,0,0,0,0,0,0,0,0],
	[0,0,0,0,0,0,0,0,0],
	[0,0,0,0,0,0,0,0,0],
	[0,0,0,0,0,0,0,0,0],
	[0,0,0,0,0,0,0,0,0]]

Q =[[0,0,0,0,0,0,0,0,0],
	[0,1,1,1,1,1,0,0,0],
	[1,0,0,0,0,0,1,0,0],
	[1,0,0,0,0,0,1,0,0],
	[1,0,0,0,0,0,1,0,0],
	[1,0,0,0,0,0,1,0,0],
	[1,0,0,0,0,0,1,0,0],
	[1,0,0,0,0,0,1,0,0],
	[1,0,0,0,0,0,1,0,0],
	[1,0,0,0,1,0,1,0,0],
	[1,0,0,0,0,1,1,0,0],
	[0,1,1,1,1,1,1,0,0],
	[0,0,0,0,0,0,0,1,0],
	[0,0,0,0,0,0,0,0,0],
	[0,0,0,0,0,0,0,0,0],
	[0,0,0,0,0,0,0,0,0]]

Pi=[[0,0,0,0,0,0,0,0,0],
	[1,0,0,0,0,0,0,0,0],
	[1,0,0,0,0,0,0,0,0],
	[1,0,0,0,0,0,0,0,0],
	[1,0,0,0,0,0,0,0,0],
	[1,0,0,0,0,0,0,0,0],
	[1,0,0,0,0,0,0,0,0],
	[1,0,0,0,0,0,0,0,0],
	[1,0,0,0,0,0,0,0,0],
	[1,0,0,0,0,0,0,0,0],
	[0,0,0,0,0,0,0,0,0],
	[1,0,0,0,0,0,0,0,0],
	[0,0,0,0,0,0,0,0,0],
	[0,0,0,0,0,0,0,0,0],
	[0,0,0,0,0,0,0,0,0],
	[0,0,0,0,0,0,0,0,0]]


from matplotlib import pyplot as mp
import numpy as np

from pyomo.environ import ConcreteModel, Var, Objective, Constraint, SolverFactory
from pyomo.environ import RangeSet, NonNegativeReals



def String2Point(Cs):
    Xs = []
    Ys = []

    v, w = 7,36
    for r in Cs:
        for c in r:
            A = np.matrix(c)
            for i in range(A.shape[0]):
                for j in range(A.shape[1]):
                    if A[i,j] == 1:
                        Xs.append(v+j)
                        Ys.append(w-i)
            v += 10
        w -= 20
        v = 7

    return Xs, Ys


def Write(Cs):
    
    Xs, Ys = String2Point(Cs)
    
    mp.xlim(0,75)
    mp.ylim(0,40)
    mp.scatter(Xs, Ys, s=70, alpha=0.5)

    mp.show()

def OT_2D_Match(Mu, Nu):    
    # Main Pyomo model
    model = ConcreteModel()
    # Parameters
    model.I = RangeSet(n)
    model.J = RangeSet(n)
    # Variables
    model.PI = Var(model.I, model.J, within=NonNegativeReals) 
    # Objective Function
    Cost = lambda x, y: (x[0] - y[0])**2 + (x[1] - y[1])**2
    
    model.obj = Objective(
        expr=sum(model.PI[i,j]*Cost(Mu[i-1], Nu[j-1]) for i,j in model.PI))
    # Constraints on the marginals
    model.Mu = Constraint(model.I, 
                          rule = lambda m, i: sum(m.PI[i,j] for j in m.J) == 1)
    model.Nu = Constraint(model.J, 
                          rule = lambda m, j: sum(m.PI[i,j] for i in m.I) == 1)
    
    # Solve the model
    sol = SolverFactory('gurobi').solve(model, tee=True)

    # Get a JSON representation of the solution
    sol_json = sol.json_repn()
    # Check solution status
    if sol_json['Solver'][0]['Status'] != 'ok':
        return None
    if sol_json['Solver'][0]['Termination condition'] != 'optimal':
        return None

    return model.obj(), dict([((i-1,j-1), model.PI[i,j]()) 
                              for i,j in model.PI if model.PI[i,j]() > 0])
    
def Interpolate(p1, p2, alpha=0.5):
    x1, y1 = p1
    x2, y2 = p2
    x3 = (alpha*x2 + (1-alpha)*x1)
    y3 = (alpha*y2 + (1-alpha)*y1)
    return x3, y3


def Plot(Mu, Nu, plan):
    from math import sqrt
    from matplotlib import pyplot as mp
    from celluloid import Camera
    fig = mp.figure()
    camera = Camera(fig)
    fig.patch.set_visible(False)
    mp.axis('off')
    
    S = 100
    for a in reversed([sqrt(1.0/S*i) for i in range(S+1)]):        
        # Displacement Interpolation
        pi = []
        px = []
        py = []
        for i,j in plan:
            x,y = Interpolate(Mu[i], Nu[j], a)     
            pi.append(plan[i,j])
            px.append(x)
            py.append(y)

        mp.scatter(px, py, color='darkblue', alpha=0.5)
        
        camera.snap()
    # mp.show()
    animation = camera.animate()
    animation.save('auguri.mp4')
    
# -----------------------------------------------
#   MAIN function
# -----------------------------------------------
if __name__ == "__main__":
    np.random.seed(13)
    Cs = [[B, U, O, N, A], [P,A,S,Q,U,A,Pi]]
    
    Xs, Ys = String2Point(Cs)
    n = len(Xs)
    
    Mu = [(x,y) for x, y in zip(Xs, Ys)]
    Nu = [(x,y) for x, y in zip(np.random.normal(35, 1, size=n),
                                np.random.normal(20, 1, size=n))]
    
    z, plan = OT_2D_Match(Mu, Nu)
    
    Plot(Mu, Nu, plan)