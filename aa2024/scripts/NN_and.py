import numpy as np
import matplotlib.pyplot as plt

from gurobipy import Model, GRB, quicksum

def AddBias(X):
    return [x for x in X] + [1]

def Sign(x, eps=1e-04):
    return 1 if x > eps else -1

def MIPand(Xs, Ys, Phi=AddBias):
    # Map input to feature space
    Xs = [Phi(x) for x in Xs]

    # Pyomo Model object
    model = Model()

    # Number of samples
    n = len(Xs)
    # Input dimension
    m = len(Xs[0])
 
    # Decision variables: weights
    w = [model.addVar(vtype=GRB.BINARY) for i in range(m)]

    # Decision variables: to measure accuracy
    alpha = [model.addVar(obj=1.0, vtype=GRB.CONTINUOUS) for i in range(n)]
    S = [model.addVar(vtype=GRB.BINARY) for i in range(n)]

    # Activation function: yhat_i = sign(x_i^T * w)
    M = 1000
    for k in range(n):
        model.addConstr( quicksum((2*w[i] - 1)*Xs[k][i] for i in range(m)) >= 0.01 - M*(1 - S[k]) )
        model.addConstr( quicksum((2*w[i] - 1)*Xs[k][i] for i in range(m)) <= 0.00 + M*S[k] )

    # To check accuracy: alpha[k] = |Y[k] - (1 - 2*S[k])|
    for k in range(n):
        model.addConstr(  Ys[k] - (2*S[k] - 1) <= 2*alpha[k] )
        model.addConstr( -Ys[k] + (2*S[k] - 1) <= 2*alpha[k] )
    
    model.optimize()

    if model.status != GRB.OPTIMAL:
        return None
    
    # Return the final solution
    wbar = [w[i].X for i in range(m)]
    return lambda X: Sign(sum((2*wbar[i] - 1)*x for i,x in enumerate(Phi(X))))

from numpy.random import normal
def AddNoise(X, mu=0.1):
    return list(map(lambda x: x+normal(0, mu), X))

#-----------------------------------------------
# MAIN function
#-----------------------------------------------
if __name__ == "__main__":
    from numpy.random import seed
    seed(13)

    Xand = [(-1, -1), (-1, +1), (+1, -1), (+1, +1)]
    Yand = [-1, -1, -1, +1]

    if False:
        # Model returns a prediction function
        F = MIPand(Xand, Yand)

        # To verify the model 
        for x, y in zip(Xand, Yand):
            yhat = F(x)
            print(y, yhat)
    else:
        Xtrain = [AddNoise(x) for x in Xand]
        # Model returns a prediction function
        F = MIPand(Xtrain, Yand)

        # To verify the model 
        Xtest = [AddNoise(x) for x in Xand]
        for x, y in zip(Xtest, Yand):
            yhat = F(x)
            print(x, y, yhat)
