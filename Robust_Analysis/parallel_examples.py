import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy import optimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF
from multiprocessing import Pool
import pdb
import time


class baseclass:
	def __init__(self):
		self.X_sample = np.random.normal(size = (100,2))
		self.Y_sample = np.random.normal(size = (100,1))
		self.B = 1
		self.R = 2
		self.kernel = RBF(1.0)
		self.delta = 1e-6
		self.beta = 0
		self.bounds = np.array([[-5,5],[-5,5]])
		pass

	def calc_musigma(self):
		Kn = self.kernel(self.X_sample)
		t = self.X_sample.shape[0]
		eta = 2/t
		Kinv = np.linalg.inv(Kn+(1+eta)*np.identity(self.X_sample.shape[0]))
		self.Kinv = Kinv
		self.mu = lambda x: np.dot(np.dot(self.kernel(x.reshape(1,-1), self.X_sample).reshape(1,-1),
			Kinv),self.Y_sample)[0,0]
		self.sigma = lambda x: (self.kernel(x.reshape(1,-1)) - np.dot(np.dot(self.kernel(x.reshape(1,-1), self.X_sample).reshape(1,-1), Kinv),self.kernel(x.reshape(1,-1), self.X_sample).reshape(-1,1)))[0,0]

		innersqrt = np.linalg.det((1+eta)*np.identity(self.X_sample.shape[0]) + self.kernel(self.X_sample))
		self.beta = self.B + self.R*math.sqrt(2*math.log(math.sqrt(innersqrt)/self.delta))
		pass


	def propose_location(self):
		x0_array = np.random.normal(size=(200,2))
		parallel_minimize([self.X_sample, self.Y_sample, self.kernel, self.Kinv, self.beta, self.bounds, x0_array])


def minimize(args):
	x0, X_sample, Y_sample, kernel, Kinv, beta, bounds = args
	mu = lambda x: np.dot(np.dot(kernel(x.reshape(1,-1), X_sample).reshape(1,-1),Kinv),Y_sample)[0,0]
	sigma = lambda x: (kernel(x.reshape(1,-1)) - np.dot(np.dot(kernel(x.reshape(1,-1), X_sample).reshape(1,-1), Kinv),kernel(x.reshape(1,-1), X_sample).reshape(-1,1)))[0,0]
	min_obj = lambda x: -mu(x) - beta*sigma(x)
	res = optimize.minimize(min_obj, x0 = x0, bounds = bounds, method='L-BFGS-B')
	return [res.fun, res.x]

def parallel_minimize(args):
	X_sample, Y_sample, kernel, Kinv, beta, bounds, x0_array = args

	opt_args = [(x0_array[i,:].reshape(-1,1), X_sample, Y_sample, kernel, Kinv, beta, bounds) for i in range(x0_array.shape[0])]
	p = Pool(10)
	print(p.map(minimize, opt_args))


if __name__ == "__main__":
	base = baseclass()
	base.calc_musigma()
	start = time.time()
	base.propose_location()
	print('Final calculation time: %.4f'%(time.time()-start))



