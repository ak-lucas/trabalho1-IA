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
alphas = [float(a) for a in (sys.argv[2]).split(',')]
imagename = sys.argv[3]

# INICIALIZAÇÃO
RL = RegularizedLinearRegression()
MS = ModelSelection()
colors = ['red', 'blue', 'black', 'magenta', 'gray', 'yellow', 'green',  'cyan', 'orange', 'pink']
LAMBDA = 0
fold = 1	
loss = {}

# CARREGA DATASET
dataset = Dataset("datasets/combined_cycle_power_plant_dataset.csv", 1)
dataset.dataset_scaling()

# 
for train,val in MS.k_fold(dataset.X, k=5, shuffle=True):
	# inicialização para cada fold
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
x = range(epocas+1)

print "Erro médio na última época - Diferença entre a última e a penúltima época:"
for i in xrange(len(alphas)):
	ax.plot(x, mean[i], color=colors[i], label=u"$alpha =$ %.3f" % alphas[i], linewidth=1)
	ax.fill_between(x, mean[i]-stdeviation[i], mean[i]+stdeviation[i], color=colors[i], alpha=0.2)
	print "\t alpha %.3f: %f \t %f" % (alphas[i], mean[i][-1], mean[i][-1] - mean[i][-2]) 
plt.xlabel(u'Épocas')
plt.ylabel(u'Custo')

box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1,
box.width, box.height * 0.9])
lgd = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=5)
plt.ylim([0.0, 125000])
plt.grid(True)

fig.savefig('imagens/calibracao_alpha/ccpp_%s.png' % imagename, bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.close(fig)

