import numpy as np  
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

class Adaboost_as_CD:
	def __init__(self):
		self.weak_clfs = []
		self.alphas = []
		self.losses = []

	def train(self, X, Y):
		N, D = X.shape
		for t in xrange(D):
			choosed_attribute = X[:, t].reshape((N,1))
			clf = DecisionTreeClassifier(criterion="entropy", max_depth = 1)
			clf.fit(choosed_attribute, Y)
			self.weak_clfs.append(clf)

		self.alphas = [0 for i in xrange(D)]
		Y = Y.reshape((N, 1))

		pre_exponential_loss = 0
		cur_exponential_loss = 0
		while True:
			changed = False
			for t_prime in xrange(D):
				old_alpha_t_prime = self.alphas[t_prime]
				h_t_prime = self.weak_clfs[t_prime].predict(X[:, t_prime].reshape((N, 1))).reshape((N, 1))

				exponential_term = 0
				for t in xrange(D):
					if t != t_prime:
						exponential_term += self.alphas[t]*self.weak_clfs[t].predict(X[:, t].reshape((N, 1))).reshape((N, 1))

				exponential_term = -Y*exponential_term
				accuracy = (h_t_prime==Y)
				accuracy = np.sum(accuracy*np.exp(exponential_term))

				error = (h_t_prime!=Y)
				error = np.sum(error*np.exp(exponential_term))


				new_alpha_t_prime = 0.5*np.log(accuracy/error)
				self.alphas[t_prime] = new_alpha_t_prime

				if new_alpha_t_prime - old_alpha_t_prime > 1e-10:
					changed = True

			cur_exponential_loss = self.exponential_loss(X, Y)
			self.losses.append(cur_exponential_loss)

			if not changed:
				break
			pre_exponential_loss = cur_exponential_loss

			

	def exponential_loss(self, X, Y):
		N, D = X.shape
		prediction = np.zeros((N, 1))
		
		for i in xrange(D):
			prediction += self.weak_clfs[i].predict(X[:, i].reshape((N, 1))).reshape((N, 1))*self.alphas[i]

		Y = Y.reshape((N, 1))
		exponential_term = Y*prediction
		loss = np.sum(np.exp(-exponential_term))
		return loss

	def predict(self, X):
		N, D = X.shape
		prediction = np.zeros((N, 1))
		
		for i in xrange(D):
			prediction += self.weak_clfs[i].predict(X[:, i].reshape((N,1))).reshape((N, 1))*self.alphas[i]
		# print prediction
		return np.sign(prediction)

	def get_accuracy(self, X, Y):
		prediction = self.predict(X)
		N, _ = X.shape
		Y = Y.reshape((N, 1))
		assert prediction.shape == Y.shape

		accuracy = np.sum(prediction==Y)/float(N)*100
		return accuracy



























