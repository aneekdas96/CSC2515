import numpy as np


def huber_loss(a, delta):
	# np.where returns second arg if first arg evaluates to true. else returns third arg
	losses = np.where(abs(a) <= delta, 1/2*(a**2), delta*(abs(a) - delta/2))
	return losses

def train_huber(X, targets, delta):
    lr = 0.01 ## learning rate

    '''Input : X, targets [data and outcome], delta value
       Output : w,b 
    '''
    
    epoch = 1
    max_epochs = 70
    # initialize my weights and bias to 0
    weights = np.zeros([4])
    bias = 0.0
    
    for epoch in range(max_epochs):
	    # get predicted value of y
	    y_pred = np.dot(X, weights) + bias
	    a = y_pred-targets # shape (100, )
	    
	    # now since we are performing full batch we add all the losses and perform gradient descent
	    losses = huber_loss(a, delta)
	    total_loss = np.sum(losses)
	    print('---------------------------------------------------------')
	    print('Epoch : ', epoch, ' | Loss : ', total_loss)
	    
	    # calculate gradient of loss w.r.t. weights
	    dLdy = a
	    # since np.where can evaluate one condition evaluating to True or False, we will make of copy of 'a' called 'dLdy'. then apply resulants to 'dLdy' based on each condition applied on 'a'
	    dLdy = np.where(abs(a) <= delta, a, dLdy) # condition 1 : if abs(a) <= delta, then return a
	    dLdy = np.where(a > delta, delta, dLdy) # condition 2 : if a > delta, then return delta
	    dLdy = np.where(a < -delta, -delta, dLdy) # condition 3 : if a < -delta, then return -delta
	    dLdw = dLdy.dot(X)

	    # update the weights
	    weights = weights - (lr * dLdw) 
	    print('weights : ', weights)

	    # calculate gradient of loss w.r.t. bias
	    # as calculated the bias is the same as dLdy as dydb = 1, so,  
	    dLdb = np.sum(dLdy)
	    bias = bias - (lr * dLdb)
	    print('bias : ', bias)

    return weights, bias
  
    

# Ground truth weights and bias used to generate toy data
w_gt = [1, 2, 3, 4] # shape (4, ), these should be my weights after training 
b_gt = 5 # this should be my bias after training

# Generate 100 training examples with 4 features
X = np.random.randn(100, 4) # shape (100, 4)
targets = np.dot(X, w_gt) + b_gt # shape (100, )


# Gradient descent
w, b = train_huber(X, targets, delta=2)
