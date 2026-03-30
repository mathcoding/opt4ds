from gurobipy import Model, GRB, quicksum, Env

import numpy as np
from math import pi
from numpy.random import uniform, normal, exponential
from matplotlib import pyplot as plt

# Generate a set of sample points with white noise
# N=Number of sample points
def GenerateSample(n=25, stdev=0.1, seed=13):
    np.random.seed(seed)
    Xs = uniform(0, 2*pi, n)   
    # Samples with white noise
    Ys = np.sin(Xs) + normal(0.0, stdev, n)
    return Xs, Ys

def GenerateRareSample(n=25, stdev=0.1, seed=13):
    np.random.seed(seed)
    Xs = uniform(0, 2*pi, n)   
    # Samples with white noise
    Ys = np.sin(Xs) + normal(0.0, stdev, n) + exponential(size=n)
    return Xs, Ys

def PlotSinSamples(Xs, Ys):
    # Plot True sin function
    D = np.linspace(0, 2*pi, 1000)
    plt.plot(D, np.sin(D), color='blue', alpha=0.5)
    # Plot sample points
    plt.plot(Xs, Ys, 'o', color='red', alpha=0.5)
    plt.show()

def PlotPrediction(Xs, Ys, F):
    # Plot True sin function
    D = np.linspace(0, 2*pi, 1000)
    plt.plot(D, np.sin(D), color='blue', alpha=0.5)
    # Plot sample points
    plt.plot(D, [F(x) for x in D], color='green', alpha=0.3)
    # Plot sample points
    plt.plot(Xs, Ys, 'o', color='red', alpha=0.5)
    plt.plot(Xs, [F(x) for x in Xs], 'o', color='green', alpha=0.3)
    # Fix axis
    plt.axis([0, 2*pi, -1.5, +1.5])
    # Show plot
    plt.show()

def RMS(Xs, Ys, F):
    return np.sqrt(sum((F(x) - y)**2 for x, y in zip(Xs, Ys)) / len(Xs))

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

# Basis functions
def MakePolynomial(q):
    def P(x):
        return [x**j for j in range(q+1)]
    return P

from math import exp, pi
def MakeGaussian(a, b, n):
    D = np.linspace(a, b, n)
    def G(x):
        return [exp(-(x-u)**2/2) for u in D]
    return G

# Trainining data
N = 100
Xs, Ys = GenerateRareSample(n=N, seed=13)
Xt, Yt = GenerateSample(n=N, seed=17, stdev=0.1)


F_LP = FittingModelPhi(Xs, Ys, MakePolynomial(3))
print('Polynomial basis, trainig RMS:', round(RMS(Xs, Ys, F_LP), 4))
print('Polynomial basis, testing RMS:', round(RMS(Xt, Yt, F_LP), 4))


F_LP = FittingModelPhi(Xs, Ys, MakeGaussian(0, 2*pi, 5))
print('Gaussian basis, trainig RMS:', round(RMS(Xs, Ys, F_LP), 4))
print('Gaussian basis, testing RMS:', round(RMS(Xt, Yt, F_LP), 4))

# PlotPrediction(Xs, Ys, F_LP)
# PlotPrediction(Xt, Yt, F_LP)

# -------------------------------------
# Comparison with ScikitLearn
# -------------------------------------
from sklearn import linear_model

# First: trasform data into the feature space
# Documentation: https://scikit-learn.org/stable/modules/preprocessing.html#polynomial-features
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(3) # Polynomial of degree q=3
Xs = poly.fit_transform(Xs.reshape(-1,1))
Xt = poly.fit_transform(Xt.reshape(-1,1))

# Second: training by standard L2 regression (first derivative equal to zero)
# Documentation: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
LR = linear_model.LinearRegression()
LR.fit(Xs, Ys)

# Arrange dataset as numpy ndarrays
def Fr():
    F = LR.predict
    return lambda x: F(np.array([x]))[0]
F = Fr()

# Measure errors:
print('Scikit, trainig RMS:', round(RMS(Xs, Ys, F), 4))
print('Scikit, testing RMS:', round(RMS(Xt, Yt, F), 4))

