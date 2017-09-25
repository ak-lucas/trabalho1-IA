# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import itertools

class Plot:
	def __init__(self):
		pass

	# X contém o conjuntos de pontos do eixo x e Y contém o conjunto de pontos do eixo y
	def plot_graphic(self, X, Y, xlabel='', ylabel='', color='blue', linewidth=2, title='', grid=False):
		plt.plot(X, Y, color=color, linewidth=linewidth)
		plt.ylim([0, 1.2])
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
		ds = self.dataset_statistics(X)
		
		numerator = np.subtract(X, ds['min'])
		denominator = np.subtract(ds['max'], ds['min'])

		return np.divide(numerator,denominator)

	def generate_polynomial_attributes(self, X, degree):
		X_ = X
		#print X_
		combinations = {}
		for d in range(2,degree+1):
			combinations[d] = []
			for i in X:
				# gera todas combinações, com repetição, de grau d, dos atributos de cada exemplo e guarda em um dicionário
				combinations[d].append(np.prod(list(itertools.combinations_with_replacement(i, d)), axis=1))

			# concatena novos atributos ao dataset
			X_ = np.concatenate((X_, np.asarray(combinations[d])), axis=1)
			#print np.asarray(combinations[d])

		#print X_.shape
		#print X_

		# retorna o novo dataset com atributos polinomiais de grau 1 a d
		return X_

class ModelSelection:
	def __init__(self):
		pass

	def k_fold(self, X, k, shuffle=False):
		indices = np.arange(X.shape[0])

		if shuffle:
			# embaralha índices
			np.random.shuffle(indices)

		# divide array de índices em k conjuntos de tamanhos iguais
		indices_ = np.asarray(np.array_split(indices, k))

		test = []
		train = []

		nbgroups = range(len(indices_))

		# cria duas listas com os índices dos conjuntos de treino e teste
		for i in nbgroups:
			test.append(indices_[i])
			train.append(np.concatenate(np.asarray(indices_[np.delete(nbgroups, i)])))

		return zip(train, test)

X = np.array([[2,3],[4,5], [6,7], [8,9]])
Y = np.random.randn(4)

dt = Dataset()
dt.generate_polynomial_attributes(X,3)
#print dt.dataset_statistics(X)
#print dt.dataset_scaling(X)

#cv = ModelSelection()

#for train,test in cv.k_fold(X, k=5, shuffle=False):
#	print train, "\n\n", test, "\n"
#	print "train: ", X[train].shape, Y[train].shape
#	print "test: ",X[test].shape, Y[test].shape, "\n\n\n"
	



