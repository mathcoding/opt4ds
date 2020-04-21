# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 10:57:37 2020

@author: Gualandi
"""
from pyomo.environ import ConcreteModel, Var, Objective, Constraint, SolverFactory
from pyomo.environ import RangeSet, NonNegativeReals

import numpy as np

Normalize = lambda x: x/sum(x)
    
def CheckSolution(Mu, Nu, plan, tol=0.000001):
    for i, mu in enumerate(Mu):
        m = 0
        for v, w in plan:
            if v == i:
                m += plan[v, w]
        if abs(m - mu) > tol:
            print("Error: ", abs(m-mu))
            return False

    for j, nu in enumerate(Nu):
        n = 0
        for v, w in plan:
            if w == j:
                n += plan[v, w]
        if abs(nu - n) > tol:
            print("Error: ", abs(n-nu))
            return False

    return True
    
    
def OT_1D(Mu, Nu):
    pi = {} # Trasport Plan
    z = 0   # Optimal Value
    # Residuals
    m, n = len(Mu), len(Nu)
    a = [mu for mu in Mu]
    b = [nu for nu in Nu]
    i, j = 0, 0
    alpha = [n**2 for _ in range(m)]
    beta = [n**2 for _ in range(n)]
    while i < m and j < n:
           
        if a[i] == b[j]:
            pi[i,j] = a[i]
            z += pi[i,j]*(i-j)**2
            i, j = i+1, j+1     
            if i < m:
                alpha[i] = beta[j-1] + (i - (j-1))**2
            if j < n:
                beta[j] = alpha[i-1] - ((i-1) - j)**2
        else: 
            pi[i,j] = min(a[i], b[j])
            z += pi[i,j]*(i-j)**2
            if a[i] > b[j]:
                a[i] = a[i] - b[j]
                b[j] = 0
                j = j + 1
                if j < n:
                    beta[j] = alpha[i] - (i - j)**2
            else:
                b[j] = b[j] - a[i]
                a[i] = 0
                i = i + 1
                if i < m:
                    alpha[i] = beta[j] + (i - j)**2
                
    return z, pi, alpha, beta
          
def OT_LP(Mu, Nu):
    # Main Pyomo model
    model = ConcreteModel()
    # Parameters
    model.I = RangeSet(len(Mu))
    model.J = RangeSet(len(Nu))
    # Variables
    model.PI = Var(model.I, model.J, within=NonNegativeReals) 
    # Objective Function
    model.obj = Objective(
        expr=sum(model.PI[i,j]*(i - j)**2 for i,j in model.PI))
    # Constraints on the marginals
    model.Mu = Constraint(model.I, 
                          rule = lambda m, i: sum(m.PI[i,j] for j in m.J) == Mu[i-1])
    model.Nu = Constraint(model.J, 
                          rule = lambda m, j: sum(m.PI[i,j] for i in m.I) == Nu[j-1])
    
    # Solve the model
    sol = SolverFactory('glpk').solve(model)

    # Get a JSON representation of the solution
    sol_json = sol.json_repn()
    # Check solution status
    if sol_json['Solver'][0]['Status'] != 'ok':
        return None
    if sol_json['Solver'][0]['Termination condition'] != 'optimal':
        return None

    return model.obj(), dict([((i-1,j-1), model.PI[i,j]()) for i,j in model.PI])
              

def OT_RAS(Mu, Nu, l = 10, it=1000):
    import numpy as np
    from math import exp
    
    m, n = len(Mu), len(Nu)
    X = np.zeros((m,n), dtype=np.float64)
    
    for i in range(m):
        for j in range(n):
            X[i,j] = exp(-l*(i-j)**2)
            
    u = np.array(Mu)
    v = np.array(Nu)
    for k in range(it):
        # Row scaling
        rho = u/np.sum(X, 1)
        for i in range(m):
            for j in range(n):
                X[i,j] = rho[i]*X[i,j]
        # Column scaling
        sigma = v/np.sum(X, 0)
        for i in range(m):
            for j in range(n):
                X[i,j] = sigma[j]*X[i,j]    
                
    z = 0
    sol = {}
    for i in range(m):
        for j in range(n):
            z += X[i,j]*(i-j)**2
            if X[i,j] > 0:
                sol[i,j] = X[i,j]
            
    return z, sol
    
    
# Functions for plotting
def Gauss(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def MovingAvg(Ls, w=3):
    """ Moving average of input data, with time window of w days """
    return [sum(Ls[max(0,i-w):min(len(Ls),i+w)])/(2*w) for i in range(len(Ls))]


def PlotGauss(x, mu, nu, plan):
    from matplotlib import pyplot as mp
    from celluloid import Camera
    fig = mp.figure()
    camera = Camera(fig)

    S = 200
    for a in [1.0/S*i for i in range(S+1)]:
        s = r'$\alpha$='+str(round(a,2))
        mp.text(0, 0.0035, s, fontsize=15)
        mp.plot(x, mu, color='blue')
        mp.plot(x, nu, color='darkgreen')
        
        # Displacement Interpolation
        pi = [0 for _ in x]
        for i,j in plan:
            p = (1-a)*x[i] + a*x[j]       
            h = 0
            for hh,px in enumerate(x):
                if px >= p:
                    h = hh-1
                    break
            beta = (p-x[h])/(x[hh]-x[h])
            pi[h] += (1-beta)*plan[i,j]
            pi[h+1] += beta*plan[i,j]
    
        mp.plot(x, MovingAvg(pi, 9), color='red')
        
        camera.snap()
   
    animation = camera.animate()
    animation.save('displacement.mp4')
    #mp.savefig("twoMeasInter.pdf", bbox_inches='tight')
    # mp.show()
            
# -----------------------------------------------
#   MAIN function
# -----------------------------------------------
if __name__ == "__main__":
    Test = 2
    if Test == 1:
        np.random.seed(13)
        Mu = Normalize(np.random.chisquare(1, 4))
        Nu = Normalize(np.random.chisquare(1, 3))

    if Test == 2:
        x = np.linspace(0, 10, 1000)
        Mu = Normalize(Gauss(x, 2, 0.4)+Gauss(x, 5, 1.2))
        Nu = Normalize(Gauss(x, 6, 1))

    
    from time import perf_counter
    t0 = perf_counter()
    z, pi, alpha, beta = OT_1D(Mu, Nu)
    print(z, perf_counter() - t0, CheckSolution(Mu, Nu, pi))

    # t0 = perf_counter()
    # zs, pis = OT_LP(Mu, Nu)
    # print(zs, perf_counter() - t0, CheckSolution(Mu, Nu, pis))


    if Test == 2:
        PlotGauss(x, Mu, Nu, pi)

    
    # for l in [100, 10, 1]:
    # t0 = perf_counter()
    # l = 1
    # z, pi = OT_RAS(Mu, Nu, l)            
    # print(z, perf_counter() - t0, CheckSolution(Mu, Nu, pi), l)
