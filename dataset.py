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

		X = np.array(X)
		Y = np.array(Y)

		indices = np.arange(X.shape[0])

		np.random.shuffle(indices)

		self.X = X[indices]
		self.Y = Y[indices]

		if len(self.attributeNames) == 0:
			self.attributeNames = range(X.shape[1])

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
		ds = self.dataset_statistics()
		
		numerator = np.subtract(self.X, ds['min'])
		denominator = np.subtract(ds['max'], ds['min'])

		self.X = np.divide(numerator,denominator)

	def plot_statistics(self):

		x = range(1, len(self.attributeNames) + 1)
		fig, ax = plt.subplots()
		ax.plot(x , self.statistics['mean'],'g*', label=u"Média")
		ax.plot(x , self.statistics['max'],'r1', label=u"Máximo")
		ax.plot(x , self.statistics['min'],'b1', label=u"Mínimo")
		ax.set_xticks([i for i in x])
		ax.set_xticklabels(self.attributeNames)

		box = ax.get_position()
		ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])

		# Put a legend below current axis
		ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
        	  fancybox=True, shadow=True, ncol=5)
		
		plt.title(u"Dataset: Estatísticas")
		fig.savefig("%s.png" % (self.filename)) 
		plt.close(fig)
		

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


