import numpy as np 
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

class SpectralClustering:
	def __init__(self, sigma=0.2, n_clusters=2):
		self.k = n_clusters
		self.sigma = sigma

	def Similarity_Matrix(self, X):
		N, _ = X.shape
		A = np.zeros((N, N))
		for i in xrange(N):
			for j in xrange(N):
				exp_term = -np.linalg.norm(X[i, :]-X[j, :])/(2*self.sigma**2)
				A[i, j] = np.exp(exp_term)
		return A
	def Lapacian_Matrix(self, X):
		N, _ = X.shape
		A = self.Similarity_Matrix(X)
		D = np.zeros((N, N))
		for i in xrange(N):
			D[i, i] = np.sum(A[i, :])

		L = D-A
		return L
	def get_V(self, X):
		L = self.Lapacian_Matrix(X)
		eig_val, eig_vectors = np.linalg.eigh(L)
		V = eig_vectors.real
		return V[:, :self.k]

	def fit_predict(self, X):
		V = self.get_V(X)
		cluster_labels = KMeans(n_clusters=self.k, random_state=0).fit_predict(V)
		return cluster_labels
