import numpy as np  
from scipy.stats import multivariate_normal
import math

class GMM:
	def __init__(self, k):
		self.K = k

	def convert_invalid_to_zero(self, Obj):
		Obj[np.isnan(Obj)] = 0.0
		Obj[np.isinf(Obj)] = 0.0
		return Obj

	def fit(self, X):
		N, D = X.shape
		mu = []
		sigma = []
		for _ in xrange(self.K):
			mu.append(np.random.uniform(-3, 3, D))
			sigma.append(np.identity(D))
			
		mu = np.array(mu)
		pi = np.ones(self.K)/self.K
		Q = np.zeros((N, self.K))
		pre_mu = []
		reg_cov = 1e-2*np.identity(D)

		pre_ll = None
		while True:
			# E step
			for n in xrange(N):
				for k in xrange(self.K):
					nprob = multivariate_normal.pdf(X[n], mean=mu[k], cov=sigma[k])
					Q[n, k] = pi[k]*nprob
				Q[n] /= np.sum(Q[n, :])
			# M step
			sum_n_qnk = np.sum(Q, axis = 0)
			mu = np.dot(X.T, Q) / sum_n_qnk
			mu = self.convert_invalid_to_zero(mu)
			mu = mu.T
			for k in xrange(self.K):
				for n in xrange(N):
					xmu = X[n, :]-mu[k]
					sigma[k] += Q[n, k]*np.outer(xmu, xmu)
				sigma[k] /= sum_n_qnk[k]
				sigma[k] = self.convert_invalid_to_zero(sigma[k])
				sigma[k] += reg_cov
			pi = sum_n_qnk / N

			# Stop creterion: if log-likelihood doesn't change much, stop
			if not pre_ll:
				pre_ll = self.cal_ll(X, mu, sigma, Q)
			else:
				ll = self.cal_ll(X, mu, sigma, Q)
				if np.abs(ll-pre_ll) < 0.5:
					self.Q = Q
					self.mu = mu
					self.sigma = sigma
					return ll
				else:
					pre_ll = ll

	def cal_ll(self, X, mu, sigma, Q):
		N, K = Q.shape
		ll = 0
		reg_cov = 1e-3*np.identity(X.shape[1])
		for n in xrange(N):
			for k in xrange(K):
				sigma[k] = self.convert_invalid_to_zero(sigma[k])
				log_n_prob = np.log(multivariate_normal.pdf(X[n], mean=mu[k], cov=sigma[k]))
				log_q = np.log(Q[n, k])
				if np.isnan(log_n_prob) or np.isinf(log_n_prob): 
					log_n_prob = 0.0
				if np.isnan(log_q) or np.isinf(log_q):
					log_q = 0.0
				ll += Q[n, k]*(log_n_prob-log_q)
		return ll




		
