import numpy as np
import pylab as plt
import scipy.spatial.distance as scidist
np.random.seed(1234)
import seaborn as sns 
sns.set_palette('bright')

def gen_data(N1=100, N2=100):
	X1 = np.random.normal(0, 1, size=(2, N1)) + np.array([[1], [1]])
	X2 = np.random.normal(0, 1, size=(2, N2)) + np.array([[-1], [-1]])
	Y1 = np.zeros((N1, ))
	Y2 = np.ones((N2, ))
	X = np.hstack((X1, X2))
	Y = np.hstack((Y1, Y2))
	print('X contains %d Examples of Class 1 and %d Examples of Class 2 in %d dimensions.'% (X1.shape[1], X2.shape[1], X.shape[0]))
	print('Shape of X: %s' % (X.shape, ))
	print('Shape of Y: %s' % (Y.shape, ))
	return X, Y

print('Generate train data...')
Xtrain, Ytrain = gen_data()

print('Generate test data...')
Xtest, Ytest = gen_data()

def plot_data(X, Y, prefix=''):
	for i in range(int(Y.max())+1):
		plt.scatter(X[0, Y==i], X[1, Y==i], label='%s Class %d' % (prefix, i))
	plt.show()

print('Train data')
plot_data(Xtrain, Ytrain)

print('Test data')
plot_data(Xtest, Ytest)

#compute distance to all training points, all 3 functions do the same thing 

'''	
	parameters: 
	Xa: shape: (D, N)
	Xb: shape: (D, 1)

	returns:
	dist: shape: (N, )
'''

def cdist_single_numpy(Xa, Xb):
	return np.sqrt(((Xa-Xb)**2).sum(0))

def cdist_single_linalg(Xa, Xb):
	return np.linalg.norm(Xa-Xb, axis=0)

cdist = cdist_single_numpy

def knn_predict_single(Xtrain, Ytrain, Xquery, k=3):
	Xquery = Xquery.reshape((Xtrain.shape[0], 1))
	dist = cdist(Xtrain, Xquery)
	knn_indices = np.argsort(dist)[:k]
	yvotes = Ytrain[knn_indices]
	yvotes, ycounts = np.unique(yvotes, return_counts=True)
	max_vote = np.argmax(ycounts)
	pred_y = yvotes[max_vote]

	return pred_y

#now we chose a ssample to query

Xquery = Xtest[:, 0:1]
pred_y = knn_predict_single(Xtrain, Ytrain, Xquery, k=3)
plot_data(Xtrain, Ytrain)
plt.scatter(Xquery[0], Xquery[1], s=100, marker='x', label='Query')
plt.show()
