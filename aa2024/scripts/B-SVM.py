from gurobipy import Model, GRB, quicksum
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt

import logging
logging.basicConfig(filename='test-all.log',
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)

def Parse(filename):
    fh = open(filename, 'r')
    fh.readline()
    Xs, Ys = [], []
    for row in fh:
        line = row.replace('\n','').split(';')
        v = int(line[0])
        Ys.append(-1 if v == 4 else 1)
        Xs.append(list(map(int, line[1:])))
    n = len(Xs)
    Xs = np.matrix(Xs)/255
    return Xs, np.array(Ys)


class BatchLearner(object):
    def __init__(self, Xtrain, Ytrain) -> None:
        self.Xtrain = Xtrain
        self.Ytrain = Ytrain

        self.mu = 0
        self.wp_hint = np.zeros(784)
        self.wn_hint = np.zeros(784)
        self.wbp_hint = 0
        self.wbn_hint = 0

        self.accuracy = 0.0
        self.Results = []

        self.Sol = []
  
    def evalTrain(self, F):
        Xs, Ys = self.Xtrain, self.Ytrain
        acc = 0
        n = len(Ys)
        for i in range(n):
            if F(Xs[i]) == Ys[i]:
                acc += 1
        return (acc/n*100)
    
    def solve(self, Xs, Ys, timelimit=10):
        model = Model()
        model.setParam(GRB.Param.OutputFlag, 0)
        model.setParam(GRB.Param.TimeLimit, timelimit)
        model.setParam(GRB.Param.BestObjStop, 1.00)
        #model.setParam(GRB.Param.Method, 2) # Barrier method
        #model.setParam(GRB.Param.Crossover, 0) # No crossover
        model.setParam(GRB.Param.Method, 1) # Dual Simplex

        # Number of samples
        n, m = Xs.shape
        N = int(sqrt(m))
       
        # Variables: w_ih weights 
        wp, wn = {}, {}
        for i in range(m):
            wp[i] = model.addVar(obj=0.1, vtype=GRB.CONTINUOUS)
            wn[i] = model.addVar(obj=0.1, vtype=GRB.CONTINUOUS)

        # Bias variable
        wbp, wbn = {}, {}
        wbp = model.addVar(obj=0.1, vtype=GRB.CONTINUOUS)
        wbn = model.addVar(obj=0.1, vtype=GRB.CONTINUOUS)

        # Variable for margin
        z = [model.addVar(obj=1, lb=0.0, vtype=GRB.CONTINUOUS) for k in range(n)]

        # Constraints
        for k in range(n):
            model.addConstr(quicksum([Ys[k]*Xs[k,i]*(wp[i] - wn[i]) for i in range(m)]) + Ys[k]*(wbp - wbn) >= 1-z[k])

        model.update()

        # Set hint for variable
        if self.mu > 0:
            muWbar, muBias = self.meanWeights()
            if muBias >= 0.0001:
                wbp.VarHintVal = muBias
                wbn.VarHintVal = 0.0
            elif muBias <= -0.0001:
                wbp.VarHintVal = 0.0
                wbn.VarHintVal = -muBias
            else:
                wbp.VarHintVal = 0.0
                wbn.VarHintVal = 0.0

            for i in wp:
                if muWbar[i] >= 0.0001:
                    wp[i].VarHintVal = muWbar[i]
                    wn[i].VarHintVal = 0.0
                elif muWbar[i] <= -0.0001:
                    wp[i].VarHintVal = 0.0
                    wn[i].VarHintVal = -muWbar[i]
                else:
                    wp[i].VarHintVal = 0.0
                    wn[i].VarHintVal = 0.0

            if muBias >= 0.0001:
                wbp.PStart = muBias
                wbn.PStart = 0.0
            elif muBias <= -0.0001:
                wbp.PStart = 0.0
                wbn.PStart = -muBias
            else:
                wbp.PStart = 0.0
                wbn.PStart = 0.0
            for i in wp:
                if muWbar[i] >= 0.0001:
                    wp[i].PStart = muWbar[i]
                    wn[i].PStart = 0.0
                elif muWbar[i] <= -0.0001:
                    wp[i].PStart = 0.0
                    wn[i].PStart = -muWbar[i]
                else:
                    wp[i].PStart = 0.0
                    wn[i].PStart = 0.0

        model.optimize()

        if model.status != GRB.Status.OPTIMAL and model.status != GRB.TIME_LIMIT and model.status != GRB.USER_OBJ_LIMIT:
            return None

        if model.SolCount == 0:
            return None
                
        if model.status == GRB.USER_OBJ_LIMIT:
            model.setParam(GRB.Param.BestObjStop, 0.0)
            model.setParam(GRB.Param.TimeLimit, 5)
            model.optimize()

        # Build predictor function
        wbar = np.array([wp[i].x - wn[i].x for i in wp])
        wbias = wbp.x - wbn.x

        # Build predictor function
        def Predict(x):
            return 1 if (wbias + np.dot(x, wbar)) >= 0 else -1
        
        acc = self.evalTrain(Predict)

        # Update internal values
        print('accuracy:', round(acc, 3), 'obj:', round(model.objVal, 3), 'runtime', round(model.runtime, 2), 'status:', model.status)
        # Keep the best start solution
        if acc > self.accuracy:
            self.accuracy = acc
            self.Predict = Predict

        self.Results.append( (acc, model.objVal) )
        self.Sol.append( (wbar, wbias) )

        # Return prediction function
        return Predict
    
    def meanWeights(self):
        muWbar = np.zeros(784)
        muBias = 0.0
        for wbar, wbias in self.Sol:
            muWbar += wbar
            muBias += wbias
        muWbar = muWbar/len(self.Sol)
        muBias = muBias/len(self.Sol)
        return muWbar, muBias
    
    def meanPredict(self):
        muWbar, muBias = self.meanWeights()
        return lambda x: 1 if (muBias + np.dot(x, muWbar)) >= 0 else -1
    
    def showWeights(self):
        A = np.array([self.wp_start[i] - self.wn_start[i] for i in self.wp_start]).reshape(28,28)
        A[np.abs(A) < 1e-09] = np.nan
        plt.imshow(A, cmap='rainbow')
        plt.colorbar()
        plt.show()

if __name__ == "__main__":
    import keras

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.astype(float)/255
    x_train = x_train.reshape(x_train.shape[0],784)

    x_test = x_test.astype(float)/255
    x_test = x_test.reshape(x_test.shape[0],784)

    np.random.seed(13)
    batch_size = 512
    epochs = 30
    for a in range(10):
        for b in range(a+1, 10):
            I = np.where((y_train == a) | (y_train == b))
            Xs = x_train[I]
            Ys = y_train[I]
            DigitOne = Ys[0]
            Ys = np.where(Ys == DigitOne, 1, -1)
            n, m = Xs.shape

            bsvm = BatchLearner(Xs, Ys)

            from time import perf_counter
            time_start = perf_counter()
            for i in range(epochs):
                Sample = np.random.choice(n, n, replace=False)
                bsvm.solve(Xs[Sample], Ys[Sample], timelimit=3600)

            # record end time
            time_end = perf_counter()
            # calculate the duration
            tain_duration = time_end - time_start

            mu_acc, min_acc, max_acc = 0, 100, 0
            for acc, obj in bsvm.Results:
                mu_acc += acc
                min_acc = min(min_acc, acc)
                max_acc = max(max_acc, acc)
            if len(bsvm.Results) > 0:
                mu_acc = round(mu_acc/len(bsvm.Results), 3)

            # record end time
            time_end = perf_counter()
            # calculate the duration
            all_duration = time_end - time_start

            # bsvm.showWeights()

            print('LOG', a, b, 'END mu accuracy:', mu_acc, round(min_acc, 2), round(max_acc, 2), ' over', len(bsvm.Results), 'time:', round(all_duration,3), 'train time:', round(tain_duration, 3))

            I = np.where((y_test == a) | (y_test == b))
            Xs = x_test[I] 
            Ys = y_test[I]
            Ys = np.where(Ys == DigitOne, 1, -1)
        
            acc = 0
            n = len(Ys)
            for i in range(n):
                if bsvm.Predict(Xs[i]) == Ys[i]:
                    acc += 1
            acc = (acc/n*100)

            print('LOG', a, b, 'Best Test accuracy:', round(acc, 2))

            acc = 0
            n = len(Ys)
            F = bsvm.meanPredict()
            for i in range(n):
                if F(Xs[i]) == Ys[i]:
                    acc += 1
            acc = (acc/n*100)

            print('LOG', a, b, 'Mean Test accuracy:', round(acc, 2))
            logging.info('LOG %d %d %d TestAccuracy %f MuAcc %f time %f t_train %f', a, b, batch_size, acc, mu_acc, all_duration, tain_duration)