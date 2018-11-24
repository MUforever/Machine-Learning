import numpy as np  
from collections import OrderedDict
import pydot

class Node:
	def __init__(self):
		self.value = None
		self.children = []
		self.next = None

class DecisionTree:
	def __init__(self, X, Y, attributes):
		self.X = X
		self.Y = Y
		self.attributes = attributes
		self.root = None
		self.n_nodes = 0
		self.graph = pydot.Dot(graph_type = 'graph')
		
	def probability(self, attribute):
		_, freqs = np.unique(attribute, return_counts=True)
		probs = freqs/float(attribute.shape[0])

		return probs
	def entropy(self, p):
			return np.sum(-p*np.log2(p))

	def IG(self, p_y, cond_entropy):
		H_y = self.entropy(p_y) 
		IG_y = H_y - cond_entropy
		return IG_y
	def conditional_entropy(self, p_x, p_y_x):
		p_y_x_hat = 1.0-p_y_x
		cond_entropy = -p_x*(p_y_x*np.log2(p_y_x)+p_y_x_hat*np.log2(p_y_x_hat))
		cond_entropy[np.isnan(cond_entropy)] = 0.0
		return np.sum(cond_entropy)
	def choose_best_attribute(self, X, Y, attributes):
		# attributes = X.shape[1]
		best_gain = float('-inf')
		best_attribute = None

		for attribute in attributes:
			attribute -= 1
			attribute_data = X[:, attribute]
			p_y = self.probability(Y)
			p_x = self.probability(attribute_data)

			unique = np.unique(attribute_data)
			p_y_x = np.zeros(len(p_x))
			for i, u in enumerate(unique):
				index = np.argwhere(attribute_data==u)
				Y = Y.astype(np.float)
				p_y_x[i] = np.sum(Y[index])/float(len(Y[index]))

			cond_entropy = self.conditional_entropy(p_x, p_y_x)
			infor_gain = self.IG(p_y, cond_entropy)


			if best_gain <= infor_gain:
				best_gain = infor_gain
				best_attribute = attribute
		return best_attribute

	def get_most_common_label(self, sample):
		Y = sample[:, 0]
		print "label: " + str(np.sum(Y)/float(len(Y)))
		return 1.0 if np.sum(Y)/float(len(Y)) > 0.5 else 0.0

	def fit(self, X, Y):
		self.ID3(X, Y)

	def ID3(self, X, Y):
		attributes = [i for i in xrange(1, X.shape[1]+1)]
		self.root = self.ID3_recur(X, Y, attributes, self.root)

	def ID3_recur(self, X, Y, attributes, root):
		root = Node()
		self.n_nodes += 1
		Y = Y.astype(np.float)
		if np.sum(Y) == len(Y) or np.sum(Y) == 0.0:
			root.value = Y[0]
			return root

		if len(attributes) == 0:
			
			root.value = 1 if np.sum(Y)/float(len(Y)) > 0.5 else 0
			return root 
		best_attribute = self.choose_best_attribute(X, Y, attributes)
		root.value = best_attribute+1
		values = np.unique(X[:, best_attribute])
		Y = Y.reshape((Y.shape[0], 1))
		for vi in xrange(len(values)):
			child = Node()
			child.value = values[vi]
			root.children.append(child)

			sample = np.hstack((Y, X))

			branch_samples = sample[sample[:, best_attribute+1]==values[vi]]

			if branch_samples.shape[0] == 0:
				self.n_nodes += 1

				child.next = self.get_most_common_label(sample)
			else:
				if len(attributes) > 0 and best_attribute+1 in attributes:
					remove_index = attributes.index(best_attribute+1)
					attributes.pop(remove_index)
				child.next = self.ID3_recur(branch_samples[:, 1:], branch_samples[:, 0], attributes, child.next)
		return root

	def get_n_nodes(self):
		return self.n_nodes

	def get_label(self, x, root):
		if not root.children:
			return root.value

		split_attribute = root.value-1
		value_in_x = x[split_attribute]
		child = None
		for c in root.children:
			if c.value == value_in_x:
				child = c
				break
		return self.get_label(x, child.next)


	def predict(self, X):
		prediction = np.zeros((X.shape[0], 1))
		for i, x in enumerate(X):
			label = self.get_label(x, self.root)
			prediction[i] = label

		return prediction

	def get_accuracy(self, X, Y):
		prediction = self.predict(X).reshape((X.shape[0], 1))
		Y = Y.reshape((Y.shape[0], 1))
		
		return np.sum(prediction==Y) / float(Y.shape[0])*100

	

















