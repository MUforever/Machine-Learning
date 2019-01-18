import numpy as np  

class NeuralNetwork:
	def __init__(self, layer_dims, learning_rate, iteration=1000):
		self.init_parameters(layer_dims)
		self.iteration = iteration
		self.learning_rate = learning_rate

	def init_parameters(self, layer_dims):
		self.parameters = {}
		self.L = len(layer_dims)
		for l in range(1, self.L):
			self.parameters['W'+str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1])*0.01
			self.parameters['b'+str(l)] = np.zeros((layer_dims[l]))

	def linear_forward(self, A, W, b):
		Z = np.dot(W, A) + b
		cache = (A, W, b)
		return Z, cache

	def sigmoid(self, Z):
		A = 1.0 / (1.0 + np.exp(-Z))
		return A, (Z) 
	def relu(self, Z):
		A = np.max(0, Z)
		return A, (Z)
	def linear_activation_forward(self, A_prev, W, b, activation):
		if activation == 'sigmoid':
			Z, linear_cache = self.linear_forward(A_prev, W, b)
			A, activation_cache = self.sigmoid(Z)
		elif activation == 'relu':
			Z, linear_cache = self.linear_cache(A_prev, W, b)
			A, activation_cache = self.relu(Z)

		cache = (linear_cache, activation_cache)
		return A, cache

	def L_model_forward(self, X):
		caches = []
		A = X

		for l in xrange(1, self.L):
			A, cache = self.linear_activation_forward(A_prev,
				self.parameters['W'+str(l)],
				self.parameters['b'+str(l)],
				activation='relu')
			caches.append(cache)

		AL, cache = self.linear_activation_forward(A, 
			self.parameters['W'+str(self.L)],
			self.parameters['b'+str(self.L)],
			activation='sigmoid')

		caches.append(cache)
		return AL, caches


	def compute_cost(self, AL, Y):
		m = Y.shape[1]

		cost = (-1/m) * np.sum(np.multiply(Y, np.log(AL)) + np.multiply(1-Y, np.log(1-AL)))
		return cost

	def linear_backward(self, dZ, cache):
		A_prev, W, b = cache
		m = A_prev.shape[1]

		dW = np.dot(dZ, cache[0].T)/m
		db = np.sum(dZ, axis = 1, keepdims=True)/m
		dA_prev = np.dot(cache[1].T, dZ)

		return dA_prev, dW, db

	def sigmoid_backward(self, dA, activation_cache):
		Z = activation_cache[0]
		dZ = dA*self.sigmoid(Z)*(1-self.sigmoid(Z))
		return dZ
	def relu_backward(self, dA, activation_cache):
		Z = activation_cache[0]
		dZ = dA*(1.0 if Z > 0 else 0.0)
		return dZ

	def linear_activation_backward(self, dA, cache, activation):
		linear_cache, activation_cache = cache

		if activation=='sigmoid':
			dZ = self.sigmoid_backward(dA, activation_cache)
		elif activation=='relu':
			dZ = self.relu_backward(dA, activation_cache)

		dA_prev, dW, db = self.linear_backward(dZ, linear_cache)
		return dA_prev, dW, db
	def L_model_backward(self, AL, Y, caches):
		self.grads = {}
		m = AL.shape[1]
		Y = Y.reshape(AL.shape)

		dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
		current_cache = caches[self.L-1]
		self.grads['dA'+str(self.L-1)], self.grads['dW'+str(self.L)], self.grads['db'+str(self.L)] = self.linear_activation_backward(dAL, current_cache, activation='sigmoid')

		for l in reversed(range(self.L-1)):
			current_cache = caches[l]
			dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(self.grads['dA'+str(l+1)], current_cache, activation='relu')

			self.grads['dA'+str(l)] = dA_prev_temp
			self.grads['dW'+str(l+1)] = dW_temp
			self.grads['db'+str(l+1)] = db_temp

	def update_parameters(self):
		for l in xrange(self.L):
			self.parameters['W'+str(l)] -= self.learning_rate*self.grads['dW'+str(l)]
			self.parameters['b'+str(l)] -= self.learning_rate*self.grads['db'+str(l)]


	def fit(self, X, Y):
		for _ in range(self.iteration):
			AL, caches = self.L_model_forward(X)
			self.L_model_backward(AL, Y, caches)
			self.update_parameters(self.learning_rate)

	def predict(self, X):
		AL, _ = self.L_model_forward(X)
		return AL
























			)
