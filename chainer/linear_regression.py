import numpy as np
import matplotlib.pyplot as plt
from chainer import Chain, Variable,optimizers
import chainer.functions as F
import chainer.links as L
import random

class Regression(Chain):
	def __init__(self):
		super(Regression,self).__init__(
			l1 = L.Linear(1,100),
			l2 = L.Linear(100,100),
			l3 = L.Linear(100,1),
		)

	def __call__(self,x):
		h = F.relu(self.l1(x))
		h = F.relu(self.l2(h))
		h = self.l3(h)
		return h

def generate_data(number_of_data,variance_of_noise):
	x = []
	y = []
	for i in range(number_of_data):
		val = random.uniform(-1.7,1.7)
		x.append([val])
		y.append([random.gauss(val**4-2*val**2+0.5*val+2.0,variance_of_noise)])

	return x,y

if __name__ == "__main__":
	#Parameters
	batchsize = 5
	max_epoch = 60
	number_of_data = 160
	variance_of_noise = 0.12
	train_ratio = 0.8 #how much data will be used in training
	slice_position = int(number_of_data * train_ratio)

	x,y = generate_data(number_of_data,variance_of_noise)

	x_train = x[:slice_position]
	y_train = y[:slice_position]
	x_train = Variable(np.array(x_train,dtype=np.float32))
	y_train = Variable(np.array(y_train,dtype=np.float32))

	x_test = x[slice_position:]
	y_test = y[slice_position:]
	x_test = Variable(np.array(x_test,dtype=np.float32))
	y_test = Variable(np.array(y_test,dtype=np.float32))

	model = Regression()
	optimizer = optimizers.MomentumSGD(lr=0.01,momentum=0.9)
	optimizer.setup(model)

	#Train
	N = len(x_train)
	perm = np.random.permutation(N)
	for epoch in range(max_epoch):
		for i in range(0,N,batchsize):
			x_train_batch = x_train[perm[i:i + batchsize]]
			y_train_batch = y_train[perm[i:i + batchsize]]

			model.cleargrads()
			t = model(x_train_batch)
			loss = F.mean_squared_error(t,y_train_batch)

			loss.backward()
			optimizer.update()
		print("epoch:",epoch,"loss:",loss.data)

	#Test
	model.cleargrads()
	t_test = model(x_test)
	loss = F.mean_squared_error(t_test,y_test)
	print("loss of test data:",loss.data)

	x_grandtruth = np.arange(-1.7,1.7,0.1)
	y_grandtruth = x_grandtruth**4-2*x_grandtruth**2+0.5*x_grandtruth+2.0

	#Plot
	plt.plot(x_grandtruth,y_grandtruth,color="red",label="Grand Truth")
	plt.scatter(x_train.data,y_train.data,label="Training Data")
	plt.scatter(x_test.data,t_test.data,label="Test Data")
	plt.xlabel("x",fontsize=16)
	plt.ylabel("y",fontsize=16)
	plt.xlim(-1.7,1.7)
	plt.ylim(0,4)
	plt.grid(True)
	plt.legend()
	plt.show()