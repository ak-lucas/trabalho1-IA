# -*- coding: utf-8 -*-

import csv
import sys
import numpy as np
import matplotlib.pyplot as plt

from dataset import Dataset
from regressao_linear_regularizado import RegularizedLinearRegression
from experimentos import ModelSelection

# PARAMETROS
epocas = int(sys.argv[1])
alphas = [int(a) for a in (sys.argv[2]).split(',')]

# INICIALIZAÇÃO
RL = RegularizedLinearRegression()
MS = ModelSelection()
colors = ['red', 'blue', 'black', 'pink', 'gray', 'yellow', 'green']
LAMBDA = 0
fold = 1
loss = {}

# CARREGA DATASET
dataset = Dataset("datasets/combined_cycle_power_plant_dataset.csv", 1)
dataset.dataset_scaling()

# 
for train,val in MS.k_fold(dataset.X, k=5, shuffle=True):
	# inicialização para cada fold
	fig, ax = plt.subplots()
	loss[str(fold)] = []

	for a in alphas:
		# ajusta o modelo
		RL.fit(dataset.X[train], dataset.Y[train], epochs=epocas, learning_rate=a, Lambda=LAMBDA)

		# guarda o custo em cada iteração
		loss[str(fold)].append(RL.loss)

	fold += 1

    
loss_ = []
for k in ['1', '2', '3', '4','5']:
	loss_.append(loss[k])
   
mean = np.asarray(loss_).mean(axis=0)
stdeviation = np.asarray(loss_).std(axis=0)

fig, ax = plt.subplots()

#     stdeviation = np.asarray(loss_).std(axis=0)[i]
#     ax.plot(range(epochs[0]+1), mean, color=colors[i], linewidth=1)
#     ax.fill_between(range(epochs[0]+1), mean-stdeviation, mean+stdeviation , color=colors[i], linewidth=1, alpha=0.2)
    
# plt.ylim([0.1, 1])
# plt.xlabel(u'Épocas')
# plt.ylabel(u'Custo')

# box = ax.get_position()
# ax.set_position([box.x0, box.y0 + box.height * 0.1,
# box.width, box.height * 0.9])
# lgd = ax.legend(handles=legends[:], loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=5)
# plt.title(title)
# plt.grid(True)

# fig.savefig('media_std.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
# plt.show()