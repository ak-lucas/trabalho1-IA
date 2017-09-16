# -*- coding: utf-8 -*-
from regressao_logistica_regularizado import RegularizedLogisticRegression	#colocar regressao_logistica_regularizado.py na mesma pasta
import csv
import numpy as np
from sklearn.metrics import accuracy_score
import time

#PREPARAÇÃO DO DATASET
X = []
Y = []
with open('iris_mod.csv', 'r') as f:
	reader = csv.reader(f)
	for r in reader:
		x = r[:-1]
		X.append([float(a) for a in x])
		Y.append(int(r[-1]))

X = np.array(X)
Y = np.array(Y)

indices = np.arange(X.shape[0])

np.random.shuffle(indices)

X = X[indices]
Y = Y[indices]

TRAIN_SIZE = int(.8 * X.shape[0])

X_train = X[:TRAIN_SIZE]
Y_train = Y[:TRAIN_SIZE]

X_test = X[TRAIN_SIZE:]
Y_test = Y[TRAIN_SIZE:]

#REGRESSÃO LOGÍSTICA
LR = RegularizedLogisticRegression()

LR.fit(X_train,Y_train, epochs=30, learning_rate=0.08, Lambda=0.1, print_results=False)
print "train accuracy: " + str(LR.accuracy_score(X_train,Y_train)*100.0) + "%"

Y_predict = LR.predict(X_test)

print 'test accuracy: ' + str(LR.accuracy_score(X_test,Y_test)*100.0) + "%"
#print accuracy_score(Y_test,Y_predict)	#sklearn accuracy

print 'final loss: ' + str(LR.loss[-1])
