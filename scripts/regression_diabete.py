# Regression on "diabates" dataset

from gurobipy import Model, GRB, quicksum, Env

from sklearn.datasets import load_diabetes

def FittingPolynomialLP(As, Bs, q):
    # Add variables: fitting a polynomial of degree q
    # Create the dataset for fitting the polynomial
    A = [[a**j for j in range(q+1)] for a in As]

    # LP model        
    env = Env(params={'OutputFlag': 0})
    model = Model(env=env)
    
    # Add variables
    x = [model.addVar(lb=-GRB.INFINITY) for _ in range(q+1)]
    u = [model.addVar(obj=0.01) for _ in range(q+1)]
    z = [model.addVar(obj=1.0) for _ in Bs]

    # Add constraints
    for i, b in enumerate(Bs):
        model.addConstr( quicksum(a*x[j] for j, a in enumerate(A[i])) - b <= z[i])
        model.addConstr(-quicksum(a*x[j] for j, a in enumerate(A[i])) + b <= z[i])

    for i, v in enumerate(x):
        model.addConstr( v <= u[i])
        model.addConstr( -v <= u[i])
        
    # Solve the model
    model.optimize()

    if model.Status != GRB.OPTIMAL:
        return None
    
    # Return the prediction function
    return MakeFunction([v.X for v in x])

def FittingByLP(Xs, Ys):
    def G(x):
        return [x[0], x[0]*x[0], x[1], x[1]*x[1], x[1]*x[1], x[2], x[2]*x[2], x[3], x[3]*x[3], x[4], x[4]*x[4], 
                x[5], x[5]*x[5], x[6], x[6]*x[6], x[7], x[7]*x[7], x[8], x[8]*x[8], x[9], x[9]*x[9]]
    Zs = Xs#list(map(G, Xs))

    print(Xs[0])
    print(Zs[0])

    n = len(Zs[0])
    m = len(Zs)

    # LP model        
    env = Env(params={'OutputFlag': 0})
    model = Model(env=env)

    # Add variables
    w = [model.addVar(lb=-100000, ub=100000) for _ in range(n+1)]
    u = [model.addVar(obj=0.0) for _ in range(n+1)]
    z = [model.addVar(obj=0.0) for _ in Ys]

    # Add constraints
    for i, y in enumerate(Ys):
        model.addConstr( quicksum(x*w[j] for j, x in enumerate(Zs[i])) + w[n] - y <= z[i])
        model.addConstr(-quicksum(x*w[j] for j, x in enumerate(Zs[i])) - w[n] + y <= z[i])

    for i, v in enumerate(w):
        model.addConstr(  v <= u[i])
        model.addConstr( -v <= u[i])
    
    model.setObjective(quicksum(zi * zi for zi in z) + quicksum(0.001*ui for ui in u))

    # Solve the model
    model.optimize()

#    if model.Status != GRB.OPTIMAL:
#        return None
    
    # Return the polynomial coefficients
    wbar = [v.X for v in w]

    # Return a function for fitting a single point
    def Predict(A):
        Ab = A#G(A)
        return wbar[-1] + sum(a*x for a, x in zip(Ab, [v for v in wbar[:-1]]))
    return Predict

# Main
data = load_diabetes()

# Take dataset
Xs = data.data
Ys = data.target

import sys
seed = 13
if len(sys.argv) > 1:
    seed = int(sys.argv[1])
print('seed:', seed, sys.argv)
# Split train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(Xs, Ys, test_size=0.2, random_state=seed)

# Train linear regressor
F = FittingByLP(X_train, y_train)

import numpy as np
print('Train error:', round(np.sqrt(np.abs(y_train - [F(x) for x in X_train]).mean()), 3))
print(' Test error:', round(np.sqrt(np.abs(y_test - [F(x) for x in X_test]).mean()), 3))

from sklearn import linear_model

lin_regr = linear_model.LinearRegression()
lin_regr.fit(X_train, y_train)

def print_accuracy(f):
    print(
        f"Sckit error: {round(np.sqrt(np.mean(abs(f(X_test) - y_test))), 3)}"
    )
print_accuracy(lin_regr.predict)

from sklearn import linear_model
clf = linear_model.Lasso(alpha=0.1)
clf.fit(X_train, y_train)
print_accuracy(clf.predict)
