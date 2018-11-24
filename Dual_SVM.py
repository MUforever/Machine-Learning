from cvxopt import matrix, solvers
import numpy as np  

class Dual_SVM:
	def __init__(self, sigma, C=None):
		self.kernel = self.gaussian_kernel
		self.C = C
		self.sigma = sigma

	def gaussian_kernel(self, x, y):
		return np.exp(-np.linalg.norm(x-y)**2/(2*self.sigma**2))

	def train(self, X, Y):
		N, D = X.shape
		Y = Y.reshape((N, 1))
		K = np.zeros((N, N))
		for i in xrange(N):
			for j in xrange(N):
				K[i, j] = self.kernel(X[i], X[j])


		P = matrix(np.outer(Y, Y)*K)
		q = matrix(-np.ones((N, 1)))

		G = np.eye((N))
		G = matrix(np.vstack((G, -np.eye((N)))))

		h = self.C*np.ones((N, 1))
		h = matrix(np.vstack((h, np.zeros((N, 1)))))
		A = matrix(Y.reshape((1, N)))
		b = matrix(0.0)

		solvers.options['show_progress'] = False
		sol = solvers.qp(P, q, G, h, A, b)

		alpha = np.ravel(sol['x'])
		sv = []
		for i, a in enumerate(alpha):
			if self.C >= a > 1e-5:
				sv.append(i)

		sv = np.array(sv)
		self.alpha = alpha[sv]

		ind = np.arange(len(alpha))[sv]

		self.X_sv = X[sv]
		self.Y_sv = Y[sv]

		self.b = 0
		for i in xrange(len(self.alpha)):
			self.b += Y[sv[i]]
			for j in xrange(len(self.alpha)):
				self.b -= self.alpha[j]*self.Y_sv[j]*self.kernel(self.X_sv[j], X[sv[i]])
		self.b /= float(len(self.alpha))
		

	def predict(self, X):
		prediction = np.zeros(len(X))

		for i in xrange(X.shape[0]):
			s = 0
			for a, y_sv, sv in zip(self.alpha, self.Y_sv, self.X_sv):
				s += a*y_sv*self.kernel(X[i], sv)
			prediction[i] = s

		prediction += self.b 
		prediction = np.sign(prediction)

		return prediction

	def get_accuracy(self, X, Y):
		N, _ = X.shape
		prediction = self.predict(X).reshape((N, 1))
		Y = Y.reshape((N, 1))

		accuracy = float(np.sum(prediction==Y)) / float(X.shape[0])
		return accuracy*100


