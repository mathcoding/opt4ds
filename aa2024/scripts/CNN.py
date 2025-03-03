from gurobipy import Model, GRB, quicksum
import numpy as np
from math import sqrt
from scipy import signal

def Parse(filename):
    fh = open(filename, 'r')

    fh.readline()

    Xs, Ys = [], []
    for row in fh:
        line = row.replace('\n','').split(';')
        v = int(line[0])
        Ys.append(-1 if v == 4 else 1)
        Xs.append(list(map(int, line[1:])))

    return np.matrix(Xs), np.array(Ys)

import matplotlib.pyplot as plt
import cv2

def Sign(x):
    return 1 if x >= 0.01 else -1

def DrawDigit(A):
    plt.imshow(A)
    plt.show()

def ApplyFilterOld(A, F):
    k, _ = F.shape
    A = A.reshape((28,28))
    n, m = A.shape
    R = np.zeros((n//2, m//2))
    for i in range(0, n-2, 2):
        for j in range(0, m-2, 2):
            R[i//2,j//2] = Sign(np.sum(np.multiply(A[i:i+k,j:j+k], F)))
    return R

def ApplyFilter(A, F, bias):
    A = A.reshape((28,28)).astype(np.float32)
    C = cv2.filter2D(A, ddepth=-1, kernel=F)
    return np.sign(bias + C[1:-1:2, 1:-1:2]-0.01)

class BatchLearner(object):
    def __init__(self, Xtrain, Ytrain, F=1, K=3) -> None:
        self.F = F # Number of filters
        self.K = K # Size of filters
        self.mu = 0
        self.Xs = Xtrain
        self.Ys = Ytrain
        self.accuracy = 0.0
        self.Results = []

    def evalTrain(self, F):
        Xs, Ys = self.Xs, self.Ys
        acc = 0
        n = len(Ys)
        for i in range(n):
            if F(Xs[i]) == Ys[i]:
                acc += 1
        return (acc/n*100)

    def solve(self, Xs, Ys, timelimit=10):
        self.model = Model()
        #self.model.setParam(GRB.Param.NonConvex, 2)
        self.model.setParam(GRB.Param.OutputFlag, 0)

        model = self.model
        model.setParam(GRB.Param.TimeLimit, timelimit)
        model.setParam(GRB.Param.BestObjStop, 1.00)
        model.setParam(GRB.Param.MIPFocus, 1)
        F = self.F
        K = self.K

        # Number of samples
        n, m = Xs.shape
        N = int(sqrt(m))

        # Variables: w_ih weights of fisrt layer
        wp, wn = {}, {}
        #wbp, wbn = {}, {}
        for f in range(F):
            #wbp[f] = model.addVar(obj=0.001, vtype=GRB.BINARY)
            #wbn[f] = model.addVar(obj=0.001, vtype=GRB.BINARY)
            for f1 in range(K):
                for f2 in range(K):
                    wp[f, f1, f2] = model.addVar(obj=0.001, vtype=GRB.BINARY)
                    wn[f, f1, f2] = model.addVar(obj=0.001, vtype=GRB.BINARY)

        # Variables: u_h weights of second layer
        up, un = {}, {}
        for f in range(F):
            for i in range(0, N-K, 2):
                for j in range(0, N-K, 2):
                    up[f,i,j] = model.addVar(obj=0.001, vtype=GRB.BINARY)
                    un[f,i,j] = model.addVar(obj=0.001, vtype=GRB.BINARY)

        # output bias
        #ubp = model.addVar(obj=0.001, vtype=GRB.BINARY)
        #ubn = model.addVar(obj=0.001, vtype=GRB.BINARY)

        # First layer activation
        z = {}
        vp, vn = {}, {}
        for k in range(n):
            for f in range(F):
                for i in range(0, N-K, 2):
                    for j in range(0, N-K, 2):
                        z[k,f,i,j] = model.addVar(vtype=GRB.BINARY)
                        vp[k,f,i,j] = model.addVar(vtype=GRB.BINARY)
                        vn[k,f,i,j] = model.addVar(vtype=GRB.BINARY)

        # Accuracy
        y_hat = [model.addVar(vtype=GRB.BINARY) for _ in range(n)]
        alpha = [model.addVar(lb=0, obj=1) for _ in range(n)] 

        # First layer: input x_ki to hidden units z_kh
        M = 1 + max(sum(Xs[k, i] for i in range(m)) for K in range(n))
        for k in range(n):
            for f in range(F):
                for i in range(0, N-K, 2):
                    for j in range(0, N-K, 2):
                        model.addConstr( quicksum(Xs[k, (i+f1)*N+j+f2]*(wp[f, f1, f2] - wn[f,f1,f2]) for f1 in range(K) for f2 in range(K) if Xs[k, (i+f1)*N+j+f2] > 0) >= 0.01 - M*(1 - z[k,f,i,j]) )
                        model.addConstr( quicksum(Xs[k, (i+f1)*N+j+f2]*(wp[f, f1, f2] - wn[f,f1,f2]) for f1 in range(K) for f2 in range(K) if Xs[k, (i+f1)*N+j+f2] > 0) <= 0.00 + M*z[k,f,i,j] )
                        #model.addConstr( (wbp[f] - wbn[f]) + quicksum(Xs[k, (i+f1)*N+j+f2]*(wp[f, f1, f2] - wn[f,f1,f2]) for f1 in range(K) for f2 in range(K) if Xs[k, (i+f1)*N+j+f2] > 0) >= 0.01 - M*(1 - z[k,f,i,j]) )
                        #model.addConstr( (wbp[f] - wbn[f]) + quicksum(Xs[k, (i+f1)*N+j+f2]*(wp[f, f1, f2] - wn[f,f1,f2]) for f1 in range(K) for f2 in range(K) if Xs[k, (i+f1)*N+j+f2] > 0) <= 0.00 + M*z[k,f,i,j] )

        # linearization constraints
        for k in range(n):
            model.addConstr( quicksum((2*vp[k,f,i,j] - 2*vn[k,f,i,j] - up[f,i,j] + un[f,i,j]) for f,i,j in up) >= 0.01 - M*(1-y_hat[k]))
            model.addConstr( quicksum((2*vp[k,f,i,j] - 2*vn[k,f,i,j] - up[f,i,j] + un[f,i,j]) for f,i,j in up) <= 0.00 + M*y_hat[k])
            #model.addConstr( quicksum((2*z[k,f,i,j] - 1)*(up[f,i,j] - un[f,i,j]) for f,i,j in up) >= 0.01 - M*(1-y_hat[k]))
            #model.addConstr( quicksum((2*z[k,f,i,j] - 1)*(up[f,i,j] - un[f,i,j]) for f,i,j in up) <= 0.00 + M*y_hat[k])
            #model.addConstr( (ubp - ubn) + quicksum((2*z[k,f,i,j] - 1)*(up[f,i,j] - un[f,i,j]) for f,i,j in up) >= 0.01 - M*(1-y_hat[k]))
            #model.addConstr( (ubp - ubn) + quicksum((2*z[k,f,i,j] - 1)*(up[f,i,j] - un[f,i,j]) for f,i,j in up) <= 0.00 + M*y_hat[k])

        if True:
            for k,f,i,j in vp:
                model.addConstr( vp[k,f,i,j] >= z[k,f,i,j] + up[f,i,j] - 1 )
                model.addConstr( vp[k,f,i,j] <= z[k,f,i,j] )
                model.addConstr( vp[k,f,i,j] <= up[f,i,j] )

                model.addConstr( vn[k,f,i,j] >= z[k,f,i,j] + un[f,i,j] - 1 )
                model.addConstr( vn[k,f,i,j] <= z[k,f,i,j] )
                model.addConstr( vn[k,f,i,j] <= un[f,i,j] )

        for k in range(n):
            model.addConstr( (2*y_hat[k]-1) - Ys[k] <= 0.5*alpha[k])
            model.addConstr( Ys[k] - (2*y_hat[k]-1) <= 0.5*alpha[k])

        # Fix idnetity kernel
        #for f1 in range(K):
        #    for f2 in range(K):
        #        model.addConstr( wn[f,f1,f2] == 0)
        #        if (f1, f2) == (K//2, K//2):
        #            model.addConstr( wp[f,f1,f2] == 1)
        #        else:
        #            model.addConstr( wp[f,f1,f2] == 0)
#                wp[f, f1, f2] = model.addVar(obj=0.001, vtype=GRB.BINARY)
#                wn[f, f1, f2] = model.addVar(obj=0.001, vtype=GRB.BINARY)

        if hasattr(self, 'wp_bar'):
            # load previous solution
            for f,f1,f2 in wp:
                wp[f,f1,f2].VarHintVal = (self.wp_bar[f,f1,f2]/self.mu)
                wn[f,f1,f2].VarHintVal = (self.wn_bar[f,f1,f2]/self.mu)
            for f,i,j in up:
                up[f,i,j].VarHintVal = (self.up_bar[f,i,j]/self.mu)
                un[f,i,j].VarHintVal = (self.un_bar[f,i,j]/self.mu)

            # Load previous solutions
        if hasattr(self, 'wp_start'): 
            for f,f1,f2 in self.wp_start:
                wp[f,f1,f2].Start = self.wp_start[f, f1, f2]
                wn[f,f1,f2].Start = self.wn_start[f, f1, f2]

            for f,i,j in self.up_start:
                up[f,i,j].Start = self.up_start[f, i, j]
                un[f,i,j].Start = self.un_start[f, i, j]

        model.optimize()
        
        if model.status != GRB.OPTIMAL and model.status != GRB.TIME_LIMIT and model.status != GRB.USER_OBJ_LIMIT:
            return None

        print('Objective:', model.objVal, 'runtime:', model.Runtime)

        # Store previous solution
        if True:
            if model.objVal <= 25:
                if hasattr(self, 'wp_bar'):
                    self.mu += 1
                    self.wp_bar = {(f,f1,f2): (wp[f,f1,f2].x+self.wp_bar[f,f1,f2]) for f,f1,f2 in wp}
                    self.wn_bar = {(f,f1,f2): (wn[f,f1,f2].x+self.wn_bar[f,f1,f2])  for f,f1,f2 in wp}
                    self.up_bar = {(f,i,j): (up[f,i,j].x+self.up_bar[f,i,j]) for f,i,j in up}
                    self.un_bar = {(f,i,j): (un[f,i,j].x+self.un_bar[f,i,j]) for f,i,j in up}
                else:
                    self.mu = 1
                    self.wp_bar = {(f,f1,f2): wp[f,f1,f2].x for f,f1,f2 in wp}
                    self.wn_bar = {(f,f1,f2): wn[f,f1,f2].x for f,f1,f2 in wp}
                    self.up_bar = {(f,i,j): up[f,i,j].x for f,i,j in up}
                    self.un_bar = {(f,i,j): un[f,i,j].x for f,i,j in up}

        # Build predict function
        KK = {}
        for f in range(F):
            KK[f] = np.matrix([[(wp[f, f1, f2].x - wn[f, f1, f2].x) for f1 in range(K) for f2 in range(K)]])
            print('K', KK[f])

        ubar = {(f,i,j): (up[f,i,j].x - un[f,i,j].x) for f,i,j in up}
        wbar = {(f,f1,f2): (wp[f,f1,f2].x - wn[f,f1,f2].x) for f,f1,f2 in wp}
        #wb_bar = {(f): (wbp[f].x - wbn[f].x) for f in wbp}
        #ub_bar = (ubp.x - ubn.x)
        print([ubar[f,i,j] for f,i,j in up if ubar[f,i,j] != 0.00])
        def Predict(X):
            tot = 0
            A = X.reshape((28,28))
            for f in range(F):
                Kernel = np.array([wbar[f,f1,f2] for f1 in range(K) for f2 in range(K)]).reshape([K,K])
                        
                H = ApplyFilter(A, Kernel, 0)#wb_bar[f])
                kk,tt = H.shape
                for i in range(0, N-K, 2):
                    for j in range(0, N-K, 2):
                        tot += ubar[f,i,j]*H[i//2, j//2]
                    
            return Sign(tot)
     
        acc = self.evalTrain(Predict)
        if acc > self.accuracy:
            self.accuracy = acc
            self.wp_start = {(f,f1,f2): wp[f,f1,f2].x for f,f1,f2 in wp} 
            self.wn_start = {(f,f1,f2): wp[f,f1,f2].x for f,f1,f2 in wp} 
            self.up_start = {(f,i,j): up[f,i,j].x for f,i,j in up} 
            self.un_start = {(f,i,j): un[f,i,j].x for f,i,j in up} 

        self.Results.append( (acc, model.objVal) )
#        print('accuracy:', acc, '- model:', model.objVal)
        return Predict


if __name__ == "__main__":
    #F1 = np.matrix([[0,0,0], [0,0,1], [0,0,0]])
    #F2 = np.zeros((3,3)) #np.matrix([[-1,-1,-1], [0,0,0], [1,1,1]])
    #DrawDigit(ApplyFilter(Xs[1], F1))
    #DrawDigit(ApplyFilter(Xs[1], F2))

    if True:
        Xs, Ys = Parse('../data/all_nine_four.csv')
        #Xs, Ys = Parse('../data/all_three_four.csv')
        n, m = Xs.shape

        cnn = BatchLearner(Xs, Ys, F=1, K=3)

        from time import perf_counter
        time_start = perf_counter()
        step = 64
        F = cnn.solve(Xs[:step], Ys[:step], timelimit=120)
        Ps = [F]
        for i in range(step, 1000, step):
            print('batch:', i)
            F = cnn.solve(Xs[i:i+step], Ys[i:i+step], timelimit=10)
            Ps.append(F)

        # record end time
        time_end = perf_counter()
        # calculate the duration
        tain_duration = time_end - time_start


        mu_acc, min_acc, max_acc = 0, 100, 0
        for acc, obj in cnn.Results:
            mu_acc += acc
            min_acc = min(min_acc, acc)
            max_acc = max(max_acc, acc)

        # record end time
        time_end = perf_counter()
        # calculate the duration
        all_duration = time_end - time_start

        print('END mu accuracy:', round(mu_acc/len(Ps),2), round(min_acc, 2), round(max_acc, 2), ' over', len(Ps), 'time:', round(all_duration,3), 'train time:', round(tain_duration, 3))


    # Redo Lenet-1 training with MILP (exercise)
    # https://github.com/grvk/lenet-1/blob/master/LeNet-1.ipynb
    # https://proceedings.neurips.cc/paper/1989/file/53c3bce66e43be4f209556518c2fcb54-Paper.pdf


    # END mu accuracy: 73.18 54.59 87.0  over 63 time: 163.612 train time: 96.549
    # END mu accuracy: 72.61 54.59 87.0  over 63 time: 172.764 train time: 103.598
    # END mu accuracy: 77.33 57.89 87.0  over 63 time: 137.646 train time: 68.678

    # Adding bias to the filters:
    # END mu accuracy: 78.49 50.7 89.1  over 63 time: 471.191 train time: 404.592
    # Linearizzato: END mu accuracy: 73.22 50.7 86.83  over 63 time: 323.174 train time: 254.318
    # END mu accuracy: 80.33 64.2 88.81  over 32 time: 67.16 train time: 33.033 (batch di 16)


    # END mu accuracy: 65.63 47.3 86.0  over 63 time: 476.338 train time: 406.786
    # END mu accuracy: 65.24 48.2 88.33  over 63 time: 477.512 train time: 408.928
