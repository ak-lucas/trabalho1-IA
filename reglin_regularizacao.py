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
lambdas = [float(a) for a in (sys.argv[4]).split(',')]
imagename = sys.argv[5]

# INICIALIZAÇÃO
RL = RegularizedLinearRegression()
MS = ModelSelection()
colors = ['red', 'blue', 'black', 'magenta', 'gray', 'yellow', 'green',  'cyan', 'orange', 'pink']
fold = 0	
erroTreino = []
erroValidacao = []

# CARREGA DATASET
dataset = Dataset("datasets/combined_cycle_power_plant_dataset.csv", 1)
dataset.dataset_scaling()
dataset.init_polynomial()
for d in xrange(1,degree + 1):
	dataset.generate_polynomial_attributes(d)

# 
for train,val in MS.k_fold(dataset.X_polinomial, k=5, shuffle=True):
	# inicialização para cada fold
	erroTreino.append([])
	erroValidacao.append([])

	for l in lambdas:
		# ajusta o modelo
		RL.fit(dataset.X_polinomial[train], dataset.Y[train], epochs=epocas, learning_rate=alpha, Lambda=l)

		# calcula e guarda o erro no treino
		Y_pred = RL.predict(dataset.X_polinomial[train])
		erro = RL.mean_absolute_error(dataset.Y[train], Y_pred)
		erroTreino[fold].append(erro)

		# calcula e guarda o erro na validação
		Y_pred = RL.predict(dataset.X_polinomial[val])
		erro = RL.mean_absolute_error(dataset.Y[val], Y_pred)
		erroValidacao[fold].append(erro)

	fold += 1

meanTreino = np.asarray(erroTreino).mean(axis=0)
stdeviationTreino = np.asarray(erroTreino).std(axis=0)
meanValidacao = np.asarray(erroValidacao).mean(axis=0)
stdeviationValidacao = np.asarray(erroValidacao).std(axis=0)

fig, ax = plt.subplots()
x = range(len(lambdas))

ax.plot(x, meanTreino, color='red', label=u"$Erro_{treino}$", linewidth=1, marker='*')
ax.fill_between(x, meanTreino-stdeviationTreino, meanTreino+stdeviationTreino, color='red', alpha=0.2)
ax.plot(x, meanValidacao, color='blue', label=u"$Erro_{validação}$", linewidth=1, marker='*')
ax.fill_between(x, meanValidacao-stdeviationValidacao, meanValidacao+stdeviationValidacao, color='blue', alpha=0.2)

#plt.ylim([0.0, 350])
plt.xlabel(u'$\lambda$')
plt.ylabel(u'Erro')

box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1,
 box.width, box.height * 0.9])
lgd = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=5)
plt.grid(True)

fig.savefig('imagens/regularizacao/ccpp_%s.png' % imagename, bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.close(fig)

