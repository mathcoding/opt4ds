# Regression on "diabates" dataset

from gurobipy import Model, GRB, quicksum, Env

from sklearn.datasets import load_diabetes

def FittingModelPhi(Xs, Ys, Phi, p=1, alpha=0.01):
    # Map data points to the feature space
    Fs = [Phi(x) for x in Xs]

    m = len(Fs)
    n = len(Fs[0])

    # LP model        
    env = Env(params={'OutputFlag': 0})
    model = Model(env=env)
    
    # Add decision variables:
    # Linear weights:
    w = [model.addVar(lb=-GRB.INFINITY) for _ in range(n)]
    # Regression error: z = |phi(x) - y|
    z = [model.addVar() for _ in range(m)]
    # u = |w|, for regularization term: alpha*|w|
    u = [model.addVar() for _ in range(n)]

    if p == 1:        
        # Linear objective function: |phi_w(x) - y| + alpha*|w|
        # Linearized with variables z and u
        model.setObjective(quicksum(z[i] for i in range(m)) + alpha*quicksum(u[j] for j in range(n)))
    elif p == 2:
        # Quadratic objective function: |phi_w(x) - y|^2 + alpha*|w|
        model.setObjective(quicksum(z[i]*z[i] for i in range(m)) + alpha*quicksum(u[j] for j in range(n)))

    # Add constraints for prediction error
    for i in range(m):
        model.addConstr( quicksum(v*w[j] for j, v in enumerate(Fs[i])) - Ys[i] <= z[i])
        model.addConstr(-quicksum(v*w[j] for j, v in enumerate(Fs[i])) + Ys[i] <= z[i])

    # Add constraints for regularization term
    for j in range(n):
        model.addConstr(u[j] >= w[j])
        model.addConstr(u[j] >= -w[j])

    model.optimize()
    
    if model.Status != GRB.OPTIMAL:
        return None
    
    # Take optimal weights
    wbar = [v.X for v in w]
    # Return a function for fitting a single point
    return lambda x: sum(v*wbar[j] for j, v in enumerate(Phi(x)))

from math import exp, pi
def MakeGaussian(a, b, n):
    D = np.linspace(a, b, n)
    def G(Xs):
        return [exp(-((x)-u)**2/2) for u in D for x in Xs]
    return G

def MakePolynomial(q):
    def P(Xs):
        return [x**j for j in range(q+1) for x in Xs]
    return P

import sys
seed = 13
if len(sys.argv) > 1:
    seed = int(sys.argv[1])
print('seed:', seed, sys.argv)

# Main
data = load_diabetes()
Xs = data.data
Ys = data.target

import numpy as np
a, b = np.min(Xs, 0), np.max(Xs, 0)

# Split train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(Xs, Ys, test_size=0.2, random_state=13)

# Train linear regressor
def RMS(Xs, Ys, F):
    return np.sqrt(sum((F(x) - y)**2 for x, y in zip(Xs, Ys)) / len(Xs))

def R2(Xs, Ys, F):
    # https://en.wikipedia.org/wiki/Coefficient_of_determination
    Sres = sum((F(x) - y)**2 for x, y in zip(Xs, Ys))
    
    mu = np.mean(Ys)
    Stot = sum((y - mu)**2 for y in Ys)

    return 1.0 - Sres/Stot

F = FittingModelPhi(X_train, y_train, MakePolynomial(1), p=1, alpha=0.1)
#F = FittingModelPhi(X_train, y_train, MakeGaussian(-0.2, 0.2, 1), p=1, alpha=0.01)

X_train, X_test, y_train, y_test = train_test_split(Xs, Ys, test_size=0.1, random_state=17)
print('Current model,   train RMS:', round(RMS(X_train, y_train, F), 4))
print('Current model, testing RMS:', round(RMS(X_test, y_test, F), 4))

print('Current model,   train RMS:', round(R2(X_train, y_train, F), 4))
print('Current model, testing RMS:', round(R2(X_test, y_test, F), 4))

from sklearn.metrics import mean_absolute_error, r2_score
print('Sklearn metrics:', mean_absolute_error(y_test, [F(x) for x in X_test]), r2_score(y_test, [F(x) for x in X_test]))

