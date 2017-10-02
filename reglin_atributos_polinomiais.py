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
graus = int(sys.argv[3])

# INICIALIZAÇÃO
RL = RegularizedLinearRegression()
MS = ModelSelection()
LAMBDA = 0
fold = 0	
erroTreino = []
erroValidacao = []
loss = []

# CARREGA DATASET
dataset = Dataset("datasets/combined_cycle_power_plant_dataset.csv", 1)

# 
for train,val in MS.k_fold(dataset.X, k=5, shuffle=True):
	# inicialização para cada fold
	dataset.init_polynomial()
	erroTreino.append([])
	erroValidacao.append([])
	loss.append([])

	for g in xrange(1,graus + 1):
		# ajusta o modelo
		dataset.generate_polynomial_attributes(g)
		RL.fit(dataset.X_polinomial[train], dataset.Y[train], epochs=epocas, learning_rate=alpha, Lambda=LAMBDA)
		loss[fold].append(RL.loss[-1])

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
meanLoss = np.asarray(loss).mean(axis=0)

print "Função de Custo médio por grau:"
for l in meanLoss:
	print "\t %.5f" % l
fig, ax = plt.subplots()
x = range(1, graus + 1)

ax.plot(x, meanTreino, color='red', label=u"$Erro_{treino}$", linewidth=1, marker='*')
ax.fill_between(x, meanTreino-stdeviationTreino, meanTreino+stdeviationTreino, color='red', alpha=0.2)
ax.plot(x, meanValidacao, color='blue', label=u"$Erro_{validação}$", linewidth=1, marker='*')
ax.fill_between(x, meanValidacao-stdeviationValidacao, meanValidacao+stdeviationValidacao, color='blue', alpha=0.2)
ax.axhline(y=meanValidacao[np.argmin(meanValidacao, axis=0)], color='black', linestyle='dashed')

#plt.ylim([0.0, 350])
plt.xlabel(u'$\eta$')
plt.ylabel(u'Erro')

box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1,
 box.width, box.height * 0.9])
lgd = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=5)
plt.grid(True)

fig.savefig('imagens/atributos_polinomiais/ccpp_%d_%.3f_%d.png' % (graus,alpha,epocas), bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.close(fig)

