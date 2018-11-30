import numpy as np  

data = []
with open("leaf.data") as f:
	for line in f:
		d = [float(x) for x in line.split(",")]
		data.append(d)

data = np.array(data)

X = data[:, 1:]
Y = data[:, 0]
Y = Y.astype(np.int)

def standrize(X):
	mu = np.mean(X, axis = 0)
	std = np.std(X, axis = 0)

	X_normalize = (X-mu)/std
	return X_normalize

X = standrize(X)

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


ks = [12, 18, 24, 36, 42]

for k in ks:
	res = []
	for _ in xrange(10):
		kmeans = KMeans(k)
		kmeans.fit(X)
		res.append(kmeans.objective())
	mu = sum(res) / 20.0
	var = np.var(res)

	print "Mean = " + str(mu) + " Variance = " + str(var) + " with k = " + str(k)

# kmeans = KMeans(36)
# kmeans.fit(X)
# error = []
# for i in xrange(1, 37):
# 	x = X[Y==i]

# 	if len(x)!=0:
# 		true_mu = np.mean(x, axis = 0)
# 		min_dist = float('inf')

# 		for m in kmeans.centroids:
# 			dist = np.sum(np.linalg.norm(m-true_mu))
# 			min_dist = min(min_dist, dist)
# 		error.append(min_dist)
# print np.sum(error)
