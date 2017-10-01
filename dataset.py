# -*- coding: utf-8 -*-
import sys
import numpy as np
import csv
import matplotlib.pyplot as plt
import itertools

class Dataset:

	def __init__(self, filename, haveHeader=0):
		self.X = []
		self.Y = []
		self.X_polinomial = []
		self.m = 0
		self.attributeNames = []
		self.statistics = {}
		self.filename = filename[:filename.find('.csv')]

		X = []
		Y = []
		with open(filename, 'r') as f:
			reader = csv.reader(f)
			
			if haveHeader == 1:
				self.attributeNames = next(reader)[:-1]

			for r in reader:
				x = r[:-1]
				X.append([float(a) for a in x])
				Y.append(to_number(r[-1]))

		self.X = np.array(X)
		self.Y = np.array(Y)
		self.m = self.X.shape[0]

		if len(self.attributeNames) == 0:
			self.attributeNames = range(self.X.shape[1])

	# retorna os valores máximos, mínimos, médios e o desvio padrão de cada coluna do treino
	def dataset_statistics(self, plot=False):
		Max = self.X.max(axis=0)
		Min = self.X.min(axis=0)
		Mean = self.X.mean(axis=0)
		Std = self.X.std(axis=0)

		self.statistics = {
			'max':Max,
			'min':Min,
			'mean':Mean,
			'std':Std
		}

		if plot == True:
			self.plot_statistics()
		
	# normalização min-max
	def dataset_scaling(self):
		self.dataset_statistics()
		
		numerator = np.subtract(self.X, self.statistics['min'])
		denominator = np.subtract(self.statistics['max'], self.statistics['min'])

		self.X = np.divide(numerator,denominator)

	def plot_statistics(self, printOut=True):

		x = range(1, len(self.attributeNames) + 1)
		fig, ax = plt.subplots()
		ax.plot(x , self.statistics['mean'],'_r', label=u"Média")
		#ax.plot(x , self.statistics['max'],'r1', label=u"Máximo")
		#ax.plot(x , self.statistics['min'],'b1', label=u"Mínimo")
		ax.errorbar(x, self.statistics['mean'], [self.statistics['mean'] - self.statistics['min'], self.statistics['max'] - self.statistics['mean']],
             fmt='_r', ecolor='gray', lw=1)

		ax.set_xticks([i for i in x])
		ax.set_xticklabels(self.attributeNames)

		box = ax.get_position()
		ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])

		# Put a legend below current axis
		ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
        	  fancybox=True, shadow=True, ncol=5)
		
		plt.title(u"Dataset: Estatísticas")
		plt.xlabel(u"Atributos")
		plt.ylabel(u"Valores")
		
		fig.savefig("%s.png" % (self.filename)) 
		plt.close(fig)

		if printOut == True:
			print self.statistics

	def init_polynomial(self, datascaling=True):
		self.X_polinomial = self.X

		if datascaling == True:
			Max = self.X_polinomial.max(axis=0)
			Min = self.X_polinomial.min(axis=0)

			numerator = np.subtract(self.X_polinomial, Min)
			denominator = np.subtract(Max, Min)

			self.X_polinomial = np.divide(numerator, denominator)

	def generate_polynomial_attributes(self, degree, datascaling=True):
		combinations = []
		
		if degree != 1:
			for i in self.X:
				combinations.append(np.prod(list(itertools.combinations_with_replacement(i, degree)), axis=1))
			combinations = np.array(combinations)
			if datascaling == True:
				Max = combinations.max(axis=0)
				Min = combinations.min(axis=0)

				numerator = np.subtract(combinations, Min)
				denominator = np.subtract(Max, Min)

				combinations = np.divide(numerator, denominator)
			# concatena novos atributos ao dataset
			self.X_polinomial = np.concatenate((self.X_polinomial, np.asarray(combinations)), axis=1)

def to_number(c):
	if c.count('.') == 0:
		return int(c)
	else:
		return float(c)

def USAGE():
	return ''.join(["USAGE: python dataset.py FILENAME BOOL\n", "\t WHERE FILENAME is dataset filename and BOOL is 0 if data has no header or 1 otherwise."])

def error():
	print "Something wrong happened. Please, check if you selected right file and configuration."
	print USAGE()

def main(filename, haveHeader):

	dataset = Dataset(filename, haveHeader)
	dataset.dataset_statistics(True)

if __name__ == '__main__':

	try:
		if len(sys.argv) != 3:
			raise Exception

		filename = sys.argv[1]
		haveHeader = int(sys.argv[2])

		if haveHeader != 0 and haveHeader != 1 :
			raise Exception
		
		main(filename, haveHeader)

	except Exception as e:
		error()
