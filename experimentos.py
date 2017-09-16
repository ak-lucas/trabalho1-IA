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


class DatasetStatistics:
	def __init__(self):
		pass		

	# recebe um array numpy e retorna os valores máximos, mínimos, médios e o desvio padrão de cada coluna
	def data_statistics(self, X):
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

class DatasetScaling:
	def __init__(self):
		pass

	# normalização min-max
	def min_max(self, X):
		# print Max.shape, X.shape
		numerator = np.subtract(X, Min)
		denominator = np.subtract(Max, Min)

		# print numerator.shape
		# print denominator.shape

		return np.divide(numerator,denominator)

class CrossValidation:
	def __init__(self):
		pass

	def k_fold(self, X, Y, k):
		# TO DO
		pass