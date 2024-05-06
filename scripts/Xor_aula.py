from gurobipy import Model, GRB, quicksum

def AddBias(X):
    return [x for x in X] + [1]

def Sign(x):
    return 1 if x >= 0.01 else -1

def MIPxor(Xs, Ys, Phi=lambda x: x, NH=2):
    Xs = [Phi(x) for x in Xs]

    m = len(Xs[0])
    n = len(Xs)

    nh = NH

    # Create model ILP
    model = Model()
    model.setParam(GRB.Param.TimeLimit, 5) # In seconds
    model.setParam(GRB.Param.OutputFlag, 1) # 0: silent, 1: normal, 2: verbose

    wp, wn = {}, {}
    for i in range(m):
        for h in range(nh):
            wp[i, h] = model.addVar(obj=0.001, vtype=GRB.BINARY) 
            wn[i, h] = model.addVar(obj=0.001, vtype=GRB.BINARY) 

    up = [model.addVar(obj=0.001, vtype=GRB.BINARY) for h in range(nh)]
    un = [model.addVar(obj=0.001, vtype=GRB.BINARY) for h in range(nh)]
    
    # Bias on output
    ubp = model.addVar(obj=0.001, vtype=GRB.BINARY)
    ubn = model.addVar(obj=0.001, vtype=GRB.BINARY) 

    z = {}
    vp, vn = {}, {}
    for k in range(n):
        for h in range(nh):
            z[k, h] = model.addVar(vtype=GRB.BINARY)
            # Variable to linearize the product of binary variables
            vp[k, h] = model.addVar(vtype=GRB.BINARY)
            vn[k, h] = model.addVar(vtype=GRB.BINARY)

    y_hat = [model.addVar(vtype=GRB.BINARY) for k in range(n)]
    alpha = [model.addVar(lb=0, obj=1) for k in range(n)]

    # First layer constraints
    M = 1000
    for k in range(n):
        for h in range(nh):
            model.addConstr( quicksum(Xs[k][i]*(wp[i,h]-wn[i,h]) for i in range(m)) >= 0.01 - M*(1 - z[k,h]) )
            model.addConstr( quicksum(Xs[k][i]*(wp[i,h]-wn[i,h]) for i in range(m)) <= 0.00 + M*z[k,h] )

    # Second layer constraints
    for k in range(n):
        # (2z-1)*(up - un) = 2z*up - 2z*un - up + un
        model.addConstr( (ubp - ubn) + quicksum((2*vp[k,h] - 2*vn[k,h] - up[h] + un[h]) for h in range(nh)) >= 0.01 - M*(1 - y_hat[k]))
        model.addConstr( (ubp - ubn) + quicksum((2*vp[k,h] - 2*vn[k,h] - up[h] + un[h]) for h in range(nh)) <= 0.00 + M*y_hat[k])

    for k in range(n):
        for h in range(nh):
            model.addConstr( vp[k,h] >= z[k,h] + up[h] - 1 )
            model.addConstr( vp[k,h] <= z[k,h] )
            model.addConstr( vp[k,h] <= up[h] )

            model.addConstr( vn[k,h] >= z[k,h] + un[h] - 1 )
            model.addConstr( vn[k,h] <= z[k,h] )
            model.addConstr( vn[k,h] <= un[h] )

    for k in range(n):
        model.addConstr( (2*y_hat[k]-1) - Ys[k] <= alpha[k])
        model.addConstr( Ys[k] - (2*y_hat[k]-1) <= alpha[k])

    model.optimize()

    if model.status != GRB.OPTIMAL and model.status != GRB.TIME_LIMIT:
        return None
    
    wbar = {(i, h): wp[i, h].X - wn[i,h].X for i, h in wp}
    ubar = [up[h].X - un[h].X for h in range(nh)]
    ubias = ubp.X - ubn.X
    print('wbar nonzero', sum(wbar[i,h] != 0 for i, h in wbar))
    print('ubar nonzero', sum(u != 0 for u in ubar) + (ubias != 0))

    def F(x):
        z = Phi(x)
        return Sign(sum(ubar[h]*Sign(sum(z[i]*wbar[i,h] for i in range(m))) for h in range(nh)))

    return F

def MIPxor_noncnvex(Xs, Ys, Phi=lambda x: x, NH=2):
    Xs = [Phi(x) for x in Xs]

    m = len(Xs[0])
    n = len(Xs)

    nh = NH

    # Create model ILP
    model = Model()
    model.setParam(GRB.Param.TimeLimit, 5) # In seconds
    model.setParam(GRB.Param.OutputFlag, 1) # 0: silent, 1: normal, 2: verbose

    # For automatic linearization techniques
    model.setParam(GRB.Param.NonConvex, 2)

    wp, wn = {}, {}
    for i in range(m):
        for h in range(nh):
            wp[i, h] = model.addVar(obj=0.001, vtype=GRB.BINARY) 
            wn[i, h] = model.addVar(obj=0.001, vtype=GRB.BINARY) 

    up = [model.addVar(obj=0.001, vtype=GRB.BINARY) for h in range(nh)]
    un = [model.addVar(obj=0.001, vtype=GRB.BINARY) for h in range(nh)]
    
    z = {}
    for k in range(n):
        for h in range(nh):
            z[k, h] = model.addVar(vtype=GRB.BINARY)

    y_hat = [model.addVar(vtype=GRB.BINARY) for k in range(n)]
    alpha = [model.addVar(lb=0, obj=1) for k in range(n)]

    # First layer constraints
    M = 1000
    for k in range(n):
        for h in range(nh):
            model.addConstr( quicksum(Xs[k][i]*(wp[i,h]-wn[i,h]) for i in range(m)) >= 0.01 - M*(1 - z[k,h]) )
            model.addConstr( quicksum(Xs[k][i]*(wp[i,h]-wn[i,h]) for i in range(m)) <= 0.00 + M*z[k,h] )

    # Second layer constraints
    for k in range(n):
        model.addConstr( quicksum(z[k,h]*(up[h] - un[h]) for h in range(nh)) >= 0.01 - M*(1 - y_hat[k]))
        model.addConstr( quicksum(z[k,h]*(up[h] - un[h]) for h in range(nh)) <= 0.00 + M*y_hat[k])

    for k in range(n):
        model.addConstr( (2*y_hat[k]-1) - Ys[k] <= alpha[k])
        model.addConstr( Ys[k] - (2*y_hat[k]-1) <= alpha[k])

    model.optimize()

    if model.status != GRB.OPTIMAL and model.status != GRB.TIME_LIMIT:
        return None
    
    wbar = {(i, h): wp[i, h].X - wn[i,h].X for i, h in wp}
    ubar = [up[h].X - un[h].X for h in range(nh)]

    print('wbar nonzero', sum(wbar[i,h] != 0 for i, h in wbar))
    print('ubar nonzero', sum(u != 0 for u in ubar))

    def F(x):
        z = Phi(x)
        return Sign(sum(ubar[h]*Sign(sum(z[i]*wbar[i,h] for i in range(m))) for h in range(nh)))

    return F


from numpy.random import normal, seed
seed(13)
def AddNoise(X, mu=0.1):
    return list(map(lambda x: x + normal(0, mu), X))

Xor = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
Yor = [-1, 1, 1, -1]

# Sample points
Xtrain, Ytrain = [], []
for _ in range(250):
    Xtrain.extend([AddNoise(x) for x in Xor])
    Ytrain.extend(Yor)

Xtest, Ytest = [], []
for _ in range(250):
    Xtest.extend([AddNoise(x) for x in Xor])
    Ytest.extend(Yor)

nh=2
F1 = MIPxor(Xtrain, Ytrain, Phi=AddBias, NH=nh)
F2 = MIPxor(Xtrain, Ytrain, Phi=AddBias, NH=nh)

acc1, acc2 = 0, 0
for x, y in zip(Xtest, Ytest):
    acc1 = acc1 + (F1(x) == y)
    acc2 = acc2 + (F2(x) == y)
#    print(x, F(x), y)

print('Accuracy:', acc1/len(Xtest), len(Xtest), len(Xtrain), 'NH',nh)
print('Accuracy Non convex:', acc2ù/len(Xtest), len(Xtest), len(Xtrain), 'NH',nh)
    







