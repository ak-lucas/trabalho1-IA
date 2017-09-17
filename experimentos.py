# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

class Plot:
	def __init__(self):
		pass

	# X contém o conjuntos de pontos do eixo x e Y contém o conjunto de pontos do eixo y
	def plot_graphic(self, X, Y, xlabel='', ylabel='', color='blue', linewidth=2, title='', grid=False):
		plt.plot(X, Y, color=color, linewidth=linewidth)
		plt.xlabel(xlabel)
		plt.ylabel(ylabel)
		plt.title(title)
		plt.grid(grid)

		plt.show()


class Dataset:
	def __init__(self):
		pass		

	# recebe um array numpy e retorna os valores máximos, mínimos, médios e o desvio padrão de cada coluna
	def dataset_statistics(self, X):
		Max = X.max(axis=0)
		Min = X.min(axis=0)
		Mean = X.mean(axis=0)
		Std = X.std(axis=0)

		# axis=0 coluna / axis=1 linha
		# print "valor máximo de cada atributo: ", Max
		# print "valor mínimo de cada atributo: ", Min
		# print "valor médio de cada atributo: ", Mean
		# print "desvio padrão de cada atributo: ", Std

		ds = {
		'max':Max,
		'min':Min,
		'mean':Mean,
		'std':Std
		}

		return ds

	# normalização min-max
	def dataset_scaling(self, X):
		# print Max.shape, X.shape
		numerator = np.subtract(X, Min)
		denominator = np.subtract(Max, Min)

		# print numerator.shape
		# print denominator.shape

		return np.divide(numerator,denominator)

	def generate_polynomial_attributes(self, X):
		# TO DO
		pass

class ModelSelection:
	def __init__(self):
		pass

	def k_fold(self, X, k, shuffle=False):
		indices = np.arange(X.shape[0])

		if shuffle:
			# embaralha índices
			np.random.shuffle(indices)

		# divide array de índices em k conjuntos de tamanhos iguais
		indices_ = np.asarray(np.split(indices, k))
		

		test = []
		train = []
		
		nbgroups = range(len(indices_))
		# cria duas listas com os índices dos conjuntos de treino e teste
		for i in nbgroups:
			test.append(indices_[i])
			train.append(indices_[np.delete(nbgroups, i)].flatten())

		#print "train", train[0].shape
		#print "teste", test[0].shape

		return zip(train, test)

X = np.random.randn(100,4)
Y = np.random.randn(100)

cv = ModelSelection()

for train,test in cv.k_fold(X, k=5, shuffle=False):
	print train, "\n\n", test, "\n"
	print "train: ", X[train].shape, Y[train].shape
	print "test: ",X[test].shape, Y[test].shape, "\n\n\n"
	



