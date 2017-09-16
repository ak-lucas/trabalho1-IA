# -*- coding: utf-8 -*-
import numpy as np

# Hyperparameters: 
#	-Lambda: fator de regularização
#	-learning_rate: taxa de aprendizado
#	-epochs: número de iterações

class RegularizedLinearRegression():
	def __init__(self):
		self.theta_n = []
		self.theta_0 = 0.
		self.loss = []

	# inicializa os pesos aleatoriamente com amostras da distribuição normal
	def init_weights(self, dim):
		return np.random.randn(dim).reshape(dim,1)
		#return np.ones(dim).reshape(dim,1)

	# função de custo
	def loss_function(self, Y, gH, Lambda, m):
		loss = np.sum(np.power(gH,2))/(2*m) + np.multiply(Lambda,np.sum(np.power(self.theta_n,2)))/(2*m)

		return loss

	def prints(self, epoch):
		print "--epoca %s: " % epoch
		print "loss: ", self.loss[epoch]
		print "theta: ", self.theta_0.reshape(theta[0].shape[0]), self.theta_n.reshape(theta[1].shape[0])

	def gradient_descent(self, epochs, X, Y, learning_rate, Lambda, m, print_results):
		for i in xrange(epochs):
			# calcula H
			H = np.dot(self.theta_n.T, X) + self.theta_0

			# calcula gradientes
			gH = H - Y
			
			gTheta_n = np.dot(X, gH.T)/m
			gTheta_0 = np.sum(gH)/m

			# calcula função de custo
			loss = self.loss_function(Y, gH, Lambda, m)
			self.loss.append(loss)

			# atualiza pesos
			self.theta_0 -= learning_rate*gTheta_0
			self.theta_n = self.theta_n*(1-(learning_rate*Lambda/m)) - learning_rate*gTheta_n

			if print_results:
				self.prints(i)

		# calcula função de custo final
		# calcula H
		H = np.dot(self.theta_n.T, X) + self.theta_0

		# calcula gradientes
		gH = H - Y
		loss = self.loss_function(Y, gH, Lambda, m)
		self.loss.append(loss)

	def fit(self, X, Y, epochs=3, learning_rate=0.01, Lambda=0.001, print_results=False):
		# dimensão dos dados
		m = X.shape[0]
		n = X.shape[1]

		# inicializa os pesos aleatoriamente
		self.theta_n = self.init_weights(n)
		self.theta_0 = self.init_weights(1)

		X = X.T
		Y = Y.reshape(1,m)

		# otimizador
		self.gradient_descent(epochs, X, Y, learning_rate, Lambda, m, print_results)

		return self

	def mean_squared_error(self, Y_true, Y_pred):
		return np.power((Y_true - Y_pred),2).mean()
    
	def mean_absolute_error(self, Y_true, Y_pred):
		return np.absolute(Y_true - Y_pred).mean()
    
	def predict(self, X):
		X = X.T

		# previsão da saída
		Y_predict = np.dot(self.theta_n.T, X) + self.theta_0

		return Y_predict
