import numpy as np  
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from itertools import combinations
import matplotlib.pyplot as plt

class Adaboost:
	def __init__(self, T):
		self.T = T
		self.alphas = []
		self.errors = []
		self.weak_clfs = []
		self.choosed_attributes_each_round = []

	def choose_best_weak_clf(self, X, Y, weights):
		best_error = float('inf')
		best_clf = None
		best_ht = None
		best_comb = None
		N, D = X.shape
		l = range(D)
		comb = combinations(l, 3)
		Y = Y.reshape((N, 1))

		for t in l:
			choosed_attributes = X[:, t].reshape((N, 1))
			clf = DecisionTreeClassifier(criterion="entropy", max_depth=1)
			weights = weights.reshape((N))
			clf.fit(choosed_attributes, Y, sample_weight=weights)

			ht = clf.predict(choosed_attributes).reshape((N, 1))
			error = (ht!=Y)
			error = np.dot(weights.T, error)
			if error < best_error:
				best_error = error
				best_clf = clf
				best_ht = ht
				best_comb = t

		self.choosed_attributes_each_round.append(best_comb)
		return best_clf, best_error, best_ht


	def train(self, X, Y):
		N, D = X.shape
		weights = np.ones((N, 1)) / float(N)
		Y = Y.reshape((N, 1))
		
		for _ in xrange(self.T):
			best_weak_clf, error, h_t = self.choose_best_weak_clf(X, Y, weights)
			assert error != 0.5
			self.weak_clfs.append(best_weak_clf)
			alpha_t = 0.5*(np.log(1 - error) - np.log(error))
			self.alphas.append(alpha_t)
			self.errors.append(error)

			weights = weights*np.exp(-Y*h_t*alpha_t) / (2.0*np.sqrt(error*(1-error)))
			assert np.sum(weights)-1.0 < 1e-10

	def exponential_loss(self, X, Y):
		N, _ = X.shape
		prediction = np.zeros((N, 1))
		
		for i in xrange(len(self.alphas)):
			t = self.choosed_attributes_each_round[i]
			prediction += self.weak_clfs[i].predict(X[:, t].reshape((N,1))).reshape((N, 1))*self.alphas[i]

		Y = Y.reshape((N, 1))
		exponential_term = Y*prediction
		loss = np.sum(np.exp(-exponential_term))
		return loss

	def predict(self, X):
		N, _ = X.shape
		prediction = np.zeros((N, 1))
		assert len(self.alphas) == self.T
		assert len(self.choosed_attributes_each_round) == len(self.alphas) == len(self.weak_clfs)
		
		for i in xrange(len(self.alphas)):
			t = self.choosed_attributes_each_round[i]
			prediction += self.weak_clfs[i].predict(X[:, t].reshape((N,1))).reshape((N, 1))*self.alphas[i]
		return np.sign(prediction)

	def get_accuracy(self, X, Y):
		prediction = self.predict(X)
		N, _ = X.shape
		Y = Y.reshape((N, 1))
		assert prediction.shape == Y.shape
		accuracy = np.sum(prediction==Y)/float(N)*100

		return accuracy