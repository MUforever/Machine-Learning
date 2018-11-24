import numpy as np

class KNN:
	def __init__(self, k, data):
		self.k = k
		self.data = data
		self.X_train = self.data[:, 1:]

		self.Y_train = self.data[:, 0]

	def euclideanDistance(self, d1, d2):

		dist = np.linalg.norm(d1-d2)
		return dist

	def getNeighbors(self, d):
		n = self.data.shape[0]
		distances = []
		for i in xrange(n):
			dist = self.euclideanDistance(d, self.X_train[i])
			distances.append([dist, self.data[i]])


		distances = sorted(distances, key = lambda x:x[0])
		neighbors = []
		for i in xrange(self.k):
			neighbors.append(distances[i][1])
		return neighbors

	def prediction(self, d):
		neighbors = self.getNeighbors(d)

		pos = 0
		neg = 0
		for i in xrange(len(neighbors)):
			if neighbors[i][0] == 0:
				neg += 1
			elif neighbors[i][0] == 1:
				pos += 1
		# print pos, neg
		return 1 if pos >= neg else 0

	def getAccuracy(self, testSet):
		N =testSet.shape[0]
		labels = np.zeros((N, 1))
		n = testSet.shape[0]
		for i, d in enumerate(testSet[:, 1:]):
			label = self.prediction(d)
			labels[i] = label

		labels = labels.reshape((N, 1))
		true_label = testSet[:, 0].reshape((N, 1))

		
		# misClass = np.sum(np.abs(labels-true_label))

		# accuracy = 1.0-misClass/n 
		accuracy = np.sum(labels==true_label)/float(n)
		return accuracy*100.0



