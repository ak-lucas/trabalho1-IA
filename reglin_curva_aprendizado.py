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
alpha = float(sys.argv[2])
degree = int(sys.argv[3])
lambdaa = float(sys.argv[4])
imagename = sys.argv[5]

# INICIALIZAÇÃO
RL = RegularizedLinearRegression()
MS = ModelSelection()

colors = ['red', 'blue', 'black', 'magenta', 'gray', 'yellow', 'green',  'cyan', 'orange', 'pink']
i = 0	
erroTreino = []
erroValidacao = []
loss = []

# CARREGA DATASET
dataset = Dataset("datasets/combined_cycle_power_plant_dataset.csv", 1)
dataset.init_polynomial()
for d in xrange(1,degree + 1):
	dataset.generate_polynomial_attributes(d)

size = dataset.X_polinomial.shape[0]
M = [int(0.2*size), int(0.4*size), int(0.6*size), int(0.8*size), int(size)]

# 
indices = range(size)
for m in M:
	np.random.shuffle(indices)
	X = np.array(dataset.X_polinomial[0:m])
	Y = np.array(dataset.Y[0:m])

	erroTreino.append([])
	erroValidacao.append([])
	loss.append([])

	for train,val in MS.k_fold(X, k=5, shuffle=True):
		# inicialização para cada fold
		
		RL.fit(X[train], Y[train], epochs=epocas, learning_rate=alpha, Lambda=lambdaa)
		loss[i].append(RL.loss[-1])

		Y_pred = RL.predict(X[train])
		erro = RL.mean_absolute_error(Y[train], Y_pred)
		erroTreino[i].append(erro)

		Y_pred = RL.predict(X[val])
		erro = RL.mean_absolute_error(Y[val], Y_pred)
		erroValidacao[i].append(erro)
	
	i += 1

meanTreino = np.asarray(erroTreino).mean(axis=1)
stdeviationTreino = np.asarray(erroTreino).std(axis=1)
meanValidacao = np.asarray(erroValidacao).mean(axis=1)
stdeviationValidacao = np.asarray(erroValidacao).std(axis=1)
meanLoss = np.asarray(loss).mean(axis=1)

fig, ax = plt.subplots()

ax.plot(M, meanTreino, color='red', label=u"$Erro_{treino}$", linewidth=1, marker='*')
ax.fill_between(M, meanTreino-stdeviationTreino, meanTreino+stdeviationTreino, color='red', alpha=0.2)
ax.plot(M, meanValidacao, color='blue', label=u"$Erro_{validação}$", linewidth=1, marker='*')
ax.fill_between(M, meanValidacao-stdeviationValidacao, meanValidacao+stdeviationValidacao, color='blue', alpha=0.2)

#plt.ylim([0.0, 350])
plt.xlabel(u'M')
plt.ylabel(u'Erro')

box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1,
 box.width, box.height * 0.9])
lgd = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=5)
plt.grid(True)

fig.savefig('imagens/ccpp_%s.png' % imagename, bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.close(fig)

