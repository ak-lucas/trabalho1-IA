# -*- coding: utf-8 -*-
import numpy as np

#Hyperparameters: 
#	-Lambda: fator de regularização
#	-learning_rate: taxa de aprendizado
#	-epochs: número de iterações

class RegularizedLogisticRegression():
	def __init__(self):
		self.theta_n = []
		self.theta_0 = 0.
		self.loss = []

	def sigmoid(self, x):
		return (1/(1+np.exp(-x)))

	#inicializa os pesos aleatoriamente com amostras da distribuição normal
	def init_weights(self, dim):
		return np.random.randn(dim).reshape(dim,1)
		#return np.ones(dim).reshape(dim,1)

	#função de custo: cross-entropy
	def loss_function(self, Y, sigmoid_z, Lambda, m):
        # VERIFICAR SE A PARTE DA REGULARIZAÇÃO ESTÁ CERTA
		loss = -np.sum(np.multiply(Y,np.log(sigmoid_z)) + np.multiply(1-Y,np.log(1-sigmoid_z)))/m + np.multiply(np.sum(np.power(self.theta_n,2)), Lambda)/m

		return loss

	def prints(self, epoch):
		print "--epoca %s: " % epoch
		print "loss: ", self.loss[epoch]
		print "theta: ", self.theta_0.reshape(theta[0].shape[0]), self.theta_n.reshape(theta[1].shape[0])

	def gradient_descent(self, epochs, X, Y, Lambda, learning_rate, m, print_results):
		for i in xrange(epochs):
			#calcula Z
			Z = np.dot(self.theta_n.T, X) + self.theta_0

			#calcula gradientes
			sigmoid_z = self.sigmoid(Z)	#função de ativação

			gZ = sigmoid_z - Y
			
			gTheta_n = np.dot(X, gZ.T)/m
			gTheta_0 = np.sum(gZ)/m

			#calcula função de custo
			loss = self.loss_function(Y, sigmoid_z, Lambda, m)
			self.loss.append(loss)

			#atualiza pesos
			self.theta_0 -= learning_rate*gTheta_0
			self.theta_n = self.theta_n*(1-(learning_rate*Lambda/m)) - learning_rate*gTheta_n

			if print_results:
				self.prints(i)

		#calcula função de custo final
		Z = np.dot(self.theta_n.T, X) + self.theta_0
		sigmoid_z = self.sigmoid(Z)	#função de ativação
		loss = self.loss_function(Y, sigmoid_z, Lambda, m)

		self.loss.append(loss)

	def fit(self, X, Y, epochs=3, learning_rate=0.01, Lambda=0.001, print_results=False):
		#dimensão dos dados
		m = X.shape[0]
		n = X.shape[1]

		#inicializa os pesos aleatoriamente
		self.theta_n = self.init_weights(n)
		self.theta_0 = self.init_weights(1)
		
		X = X.T
		Y = Y.reshape(1,m)

		#verifica as dimensões
		#assert(self.theta_n.shape[0] == X.shape[0])
		
		self.gradient_descent(epochs, X, Y, Lambda, learning_rate, m, print_results)

		return self

	def accuracy_score(self, X, Y):
		m = X.shape[0]
		Y_pred = self.predict(X)
		#número de exemplos menos o número de erros dividido pelo número de exemplos
		accuracy =  float(m - np.sum(np.logical_xor(Y_pred, Y)))/m

		return accuracy
    
	def binary_error(self, X, Y):
		return 1 - self.accuracy_score(X,Y)

	def predict(self, X):
		X = X.T

		#verifica as dimensões antes de fazer o produto interno
		#assert(self.theta_n.shape[0] == X.shape[0])

		Z = np.dot(self.theta_n.T, X) + self.theta_0
		sigmoid_z = self.sigmoid(Z)	#função de ativação

		#Z.shape == (1,m)
		#sigmoid_z.shape = (1,m) -> todas as predições estão neste array

		#verifica se cada predição é maior ou igual a 0.5 e atribui classe 0 ou 1
		Y_predict = np.greater_equal(sigmoid_z, 0.5)

		return Y_predict.astype(int).flatten()


