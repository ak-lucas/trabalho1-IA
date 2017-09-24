# -*- coding: utf-8 -*-

import csv
import numpy as np
from dataset import Dataset
import matplotlib.pyplot as plt
from regressao_linear_regularizado import RegularizedLinearRegression

# CARREGA DATASET
dataset = Dataset("datasets/combined_cycle_power_plant_dataset.csv", 1)
TRAIN_SIZE = int(.8 * dataset.m)
epocas = 10

originalX = dataset.X
originalY = dataset.Y

loss = []
error = []
lossNormalizado = []
errorNormalizado = []

for i in xrange(10):
	dataset.X = originalX
	dataset.Y = originalY

	indices = np.arange(dataset.m)
	np.random.shuffle(indices)

	X = dataset.X[indices]
	Y = dataset.Y[indices]
	
	X_train = dataset.X[:TRAIN_SIZE]
	Y_train = dataset.Y[:TRAIN_SIZE]

	X_test = dataset.X[TRAIN_SIZE:]
	Y_test = dataset.Y[TRAIN_SIZE:]

	lr = RegularizedLinearRegression()
	lr.fit(X_train, Y_train, epochs=epocas, learning_rate=0.000001, Lambda=0.00)
	loss.append(lr.loss)

	dataset.dataset_scaling()

	X_train = dataset.X[:TRAIN_SIZE]
	Y_train = dataset.Y[:TRAIN_SIZE]

	X_test = dataset.X[TRAIN_SIZE:]
	Y_test = dataset.Y[TRAIN_SIZE:]

	lrScaling = RegularizedLinearRegression()
	lrScaling.fit(X_train, Y_train, epochs=epocas, learning_rate=0.5, Lambda=0.00)
	lossNormalizado.append(lrScaling.loss)

loss = np.array(loss)
lossNormalizado = np.array(lossNormalizado)

mediaLoss = loss.mean(axis=0)/10000
mediaLossNormalizado = lossNormalizado.mean(axis=0)/10000

x = range(0, epocas + 1)
fig, ax = plt.subplots()

ax.plot(x , mediaLoss,'g-', label=u"Puro")
ax.plot(x , mediaLossNormalizado,'r-', label=u"Normalizado")

box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1,
         box.width, box.height * 0.9])

# Put a legend below current axis
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
	  fancybox=True, shadow=True, ncol=5)

ax.get_xaxis().get_major_formatter().set_scientific(True)

plt.title(u"Comparação: Erro no Treino dos dados puros e normalizados")
plt.xlabel(u"Iterações")
plt.ylabel(u"Função Custo($x10^4$)")
plt.grid(True)

fig.savefig("comparacao.png") 
plt.close(fig)
