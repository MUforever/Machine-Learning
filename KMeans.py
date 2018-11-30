import numpy as np  

class KMeans:
	def __init__(self, k):
		self.k = k

	def fit(self, X):
		N, D = X.shape
		centroids = []
		clusters = []

		# Choose initial centroids randomly
		# for i in xrange(self.k):
		# 	c = np.random.uniform(-3, 3, D)
		# 	centroids.append(c)

		# Using kmean++ to select initial centroids
		clusters = self.kmean_plusplus_init(X)
		for c in clusters:
			centroids.append(X[c])

		while True:
			clusters = {}
			for i in xrange(self.k):
				clusters[i] = []
			for d in X:
				distances = [np.linalg.norm(d-c) for c in centroids]
				clusters[np.argmin(distances)].append(d)

			new_centroids = []
			diff = 0

			for i in xrange(self.k):
				c = centroids[i]
				if clusters[i]:
					new_c = np.mean(clusters[i], axis=0)
				else:
					new_c = np.zeros((c.shape))
				diff += np.sum(np.abs(new_c-c))
				new_centroids.append(new_c)

			if diff < 1e-10:
				break
			centroids=new_centroids

		self.centroids = centroids
		self.clusters = clusters

	def get_clusters(self):
		return self.clusters
	def get_centrois(self):
		return self.centroids

	def kmean_plusplus_init(self, X):
		N, D = X.shape
		clusters = []
		first_c = np.random.randint(0, N+1, size=1)[0]
		clusters.append(first_c)

		while len(clusters) < self.k:
			weights = []
			for n in xrange(N):
				min_d = float('inf')
				for c in clusters:
					min_d = min(min_d, np.linalg.norm(X[n]-X[c]))
				weights.append(min_d)

			prob = weights / np.sum(weights)

			new_c = np.random.choice(range(N), p=prob)
			if new_c not in clusters:
				clusters.append(new_c)
		return clusters

	def objective(self):
		res = 0
		for i in xrange(self.k):
			for d in self.clusters[i]:
				res += np.linalg.norm(d-self.centroids[i])**2
		return res

