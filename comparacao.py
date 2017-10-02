# -*- coding: utf-8 -*-

import csv
import numpy as np
import sys
from dataset import Dataset
import matplotlib.pyplot as plt
from regressao_linear_regularizado import RegularizedLinearRegression
from experimentos import ModelSelection

# CARREGA DATASET
datasetPuro = Dataset("datasets/combined_cycle_power_plant_dataset.csv", 1)
lr1 = float(sys.argv[1])

datasetNormalizado = Dataset("datasets/combined_cycle_power_plant_dataset.csv", 1)
datasetNormalizado.dataset_scaling()
lr2 = float(sys.argv[2])

RL = RegularizedLinearRegression()
MS = ModelSelection()

epocas = 3

loss = []
lossNormalizado = []
fold = 0
LAMBDA = 0

for train,val in MS.k_fold(datasetPuro.X, k=5, shuffle=True):
	# inicialização para cada fold

	RL.fit(datasetPuro.X[train], datasetPuro.Y[train], epochs=epocas, learning_rate=lr1, Lambda=LAMBDA)
	loss.append(RL.loss)

	RL.fit(datasetNormalizado.X[train], datasetNormalizado.Y[train], epochs=epocas, learning_rate=lr2, Lambda=LAMBDA)
	lossNormalizado.append(RL.loss)

	fold += 1

meanPuro = np.asarray(loss).mean(axis=0)/10000
stdeviationPuro = np.asarray(loss).std(axis=0)/10000
meanNormalizado = np.asarray(lossNormalizado).mean(axis=0)/10000
stdeviationNormalizado = np.asarray(lossNormalizado).std(axis=0)/10000

x = range(0, epocas + 1)
fig, ax = plt.subplots()

ax.plot(x, meanPuro, color='red', label=u"$Custo_{treino_{puro}}$", linewidth=1, marker='*')
ax.fill_between(x, meanPuro-stdeviationPuro, meanPuro+stdeviationPuro, color='red', alpha=0.2)
ax.plot(x, meanNormalizado, color='blue', label=u"$Custo_{treino_{normalizado}}$", linewidth=1, marker='*')
ax.fill_between(x, meanNormalizado-stdeviationNormalizado, meanNormalizado+stdeviationNormalizado, color='blue', alpha=0.2)

#plt.ylim([0.0, 350])
plt.xlabel(u'$Época$')
plt.ylabel(u'Custo($x10^4 $)')

box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1,
 box.width, box.height * 0.9])
lgd = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=5)
plt.grid(True)

fig.savefig('imagens/ccpp_comparacao.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.close(fig)