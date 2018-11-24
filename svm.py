from cvxopt import matrix, solvers
import numpy as np

X = []
Y = []
with open("mystery.data") as f:
	for line in f:
		data = [float(x) for x in line.split(",")]
		X.append(data[:4])
		Y.append(data[4])

X = np.array(X)
Y = np.array(Y)
N, D = X.shape
Y = Y.reshape((N, 1))

X_feature = np.hstack((X, X[:, 0].reshape(N, 1)*X[:, 1].reshape(N, 1), X[:, 0].reshape(N,1)*X[:, 2].reshape(N, 1), X[:, 1].reshape(N,1)*X[:, 2].reshape(N, 1), X**2))
N, D = X_feature.shape

P = np.eye(D+1, D+1)
temp = np.ones((D+1, 1))
temp[D] = 0
P = np.multiply(P, temp)
q = np.zeros((D+1, 1))
G = -np.multiply(Y, X_feature)
G = np.hstack((G, -Y))

h = -np.ones((N, 1))

P = matrix(P)
q = matrix(q)
G = matrix(G)
h = matrix(h)

sol = solvers.qp(P, q, G, h)

print sol['status']
# print sol['x']
# print sol['primal objective']

opt_l = len(sol['x'])
opt_w, opt_b = sol['x'][:opt_l-1:], sol['x'][opt_l-1]
opt_w = np.array(opt_w)

support_vectors = []
original_sv = []

epsilon = 1e-9
for i in xrange(N):
	if Y[i]*(np.dot(X_feature[i], opt_w) + opt_b) - 1 <= epsilon:
		support_vectors.append(X_feature[i])
		original_sv.append(X_feature[i][:4])
		

optimal_margin = 2.0/np.linalg.norm(opt_w)
print "\n\n"
print "------------Learned Parameters--------------"
print "             Learned Weights: \n"
for i in xrange(11):
	print "              " + str(sol['x'][i])
print "\n             Learned Bias: \n"
print "              " + str(sol['x'][11]) + "\n"
print "------------Optimal Margin-----------------"
print "              " + str(optimal_margin) + "\n"
print "\n------------Support Vectors-----------------"
for i, sv in enumerate(support_vectors):
	print "               " + str(i+1) + ": " + str(sv) + "\n"

print "\n------------Original Data Point in SV----------------"
for i, osv in enumerate(original_sv):
	print "               " + str(i+1) + ": " + str(osv) + "\n"


