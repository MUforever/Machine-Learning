from cvxopt import matrix, solvers
import numpy as np 

class Primal_SVM_Slack:
	def __init__(self, C):
		self.C = C

	def train(self, X_train, Y_train):
		N, D = X_train.shape
		Y_train[Y_train==0] = -1
		Y_train = Y_train.reshape((N, 1))

		P = np.eye(N+D+1)
		temp = np.ones((N+D+1, 1))
		temp[D:] = 0
		P = np.multiply(P, temp)

		q = np.zeros((N+D+1, 1))
		q[D+1:] = self.C

		G = -np.multiply(Y_train, X_train)
		G = np.hstack((G, -Y_train))
		tempG = -np.eye((N))
		G = np.hstack((G, tempG))
		tempG1 = np.zeros((N, D+1))
		tempG2 = -np.eye((N))
		tempG = np.hstack((tempG1, tempG2))

		G = np.vstack((G, tempG))

		h = -np.ones((N, 1))
		temph = np.zeros((N, 1))
		h = np.vstack((h, temph))

		P = matrix(P)
		q = matrix(q)
		G = matrix(G)
		h = matrix(h)

		solvers.options['show_progress'] = False
		sol = solvers.qp(P, q, G, h)
		self.w = np.array(sol['x'][:D])
		self.b = np.array(sol['x'][D])

	def predict(self, X):
		prediction = np.dot(X, self.w) + self.b

		prediction = np.sign(prediction)
		return prediction

	def get_accuracy(self, X, Y):
		prediction = self.predict(X)
		Y = Y.reshape((Y.shape[0], 1))

		accuracy = float(np.sum(prediction==Y)) / float(X.shape[0])
		return accuracy*100

