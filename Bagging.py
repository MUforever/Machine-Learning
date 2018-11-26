import numpy as np  
from sklearn.tree import DecisionTreeClassifier

class Bagging:
	def __init__(self, boostrap_sample):
		self.boostrap_sample = boostrap_sample
		self.weak_clfs = []

	def get_sample(self, X, Y):
		sampleX = []
		sampleY = []
		N, D = X.shape
		Y = Y.reshape((N, 1))
		for _ in xrange(N):
			rand = np.random.randint(N)
			sampleX.append(X[rand, :])
			sampleY.append(Y[rand, :])
		sampleX = np.array(sampleX)
		sampleY = np.array(sampleY)
		return sampleX, sampleY

	def train(self, X, Y):
		N, D = X.shape
		for t in xrange(self.boostrap_sample):
			sampleX, sampleY = self.get_sample(X, Y)
			clf = DecisionTreeClassifier(criterion="entropy", max_depth = 1)
			clf.fit(sampleX, sampleY)
			self.weak_clfs.append(clf)
			
	def predict(self, X):
		N, D = X.shape
		prediction = np.zeros((N, 1))
		for i in xrange(self.boostrap_sample):
			prediction += self.weak_clfs[i].predict(X).reshape((N, 1))

		prediction = prediction / float(D)
		return np.sign(prediction)

	def get_accuracy(self, X, Y):
		prediction = self.predict(X)
		N, _ = X.shape
		Y = Y.reshape((N, 1))
		assert prediction.shape == Y.shape

		accuracy = np.sum(prediction==Y)/float(N)*100
		return accuracy






	