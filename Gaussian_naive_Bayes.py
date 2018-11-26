import numpy as np  
import csv

class Gaussian_Naive_Bayes:
	def __init__(self):
		self.class_labels = []
		self.mus = {}
		self.stds = {}
		self._priors = {}

	def get_class_labels(self, Y):
		self.class_labels = np.unique(Y)
	def separate_data_by_label(self, X, Y, label):
		return X[np.where(Y==label)]
	def prior(self, Y):
		for label in self.class_labels:
			self._priors[label] = np.sum(Y[Y==label]) / float(len(Y))

	def mean(self, X):
		mu = np.mean(X, axis = 0)
		return mu
	def std(self, X):
		std = np.std(X, axis = 0)
		return std

	def gaussian_probability(self, mu, std, X):
		gaussian = np.exp((-(X-mu)**2)/(2.0*std**2)) / (std*np.sqrt(2.0*np.pi))
		return gaussian

	def get_parameters_for_each_label(self, X, Y):
		for label in self.class_labels:
			X_separated = self.separate_data_by_label(X, Y, label)
			self.mus[label] = self.mean(X_separated)
			self.stds[label] = self.std(X_separated)

	def train(self, X, Y):
		self.get_class_labels(Y)
		self.prior(Y)
		self.get_parameters_for_each_label(X, Y)

	def predict(self, X):
		N, _ = X.shape
		num_labels = len(self.class_labels)
		# probability: a NxL matrix, where N is the number of data points and L is the types of labels
		# so for every data point, every label will give its P(x|y) when doing prediction
		probability = np.zeros((N, num_labels))
		for i, label in enumerate(self.class_labels):
			mu = self.mus[label]
			std = self.stds[label]
			gaussian = self.gaussian_probability(mu, std, X)
			prod = np.prod(gaussian, axis = 1)
			probability[:, i] = prod*self._priors[label]

		# Choose the class label that can give the new data point the highest probability
		# as its predicted label
		prediction = np.argmax(probability, axis = 1)

		for i in xrange(len(self.class_labels)-1, -1, -1):
			prediction[prediction==i] = self.class_labels[i]

		return prediction

	def get_accuracy(self, X, Y):
		prediction = self.predict(X)
		N, _ = X.shape
		Y = Y.reshape((N, 1))
		prediction = prediction.reshape((N, 1))
		assert prediction.shape == Y.shape
		accuracy = np.sum(prediction==Y)/float(N)*100

		return accuracy





