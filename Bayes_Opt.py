import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF
from multiprocessing import Pool
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib import rc, font_manager, cm
import pdb
import pickle
import time

matplotlib.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath,amssymb}']
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
plt.rcParams.update({'font.size':24})
plt.rcParams["figure.figsize"] = (12.8,8)

class CVaR_optimizer:
	def __init__(self, objective, bounds, B, R=1, delta=0.05, n_restarts = 0,
		n_init_samples = 5, tolerance = 0.05, length_scale = 1, constraints = None,
		avail_processors = 1, debug = False, granularity = 30, risk_prob = 0.05,
		lb = -5, ub = 5):
		# Objective here encodes the objective function to be optimized
		# Bounds indicates the bounds over which objective is to be optimized.
		# This algorithm will assume that the bounding region is hyper-rectangular
		# as this assumption can be made trivially (if Objective is to be optimized
		# over a compact, non-rectangular space, and if Objective is continuous over
		# said space, then define a continuous, surjective map from a larger hyper-
		# rectangular space to the compact space in question.  The resulting composite
		# function is a continuous objective function which still optimizes the original).

		# Objective should meet the criteria that objective(x) = y, where x is an 1 x N
		# numpy array and y is a float.

		# Bounds should be an Nx2 numpy array, where N is the dimensionality of the
		# constraint space.  The first column contains the lower bounds, the second the
		# upper bound.

		# B is the presumed upper bound on the RHKS norm for the Objective in question

		# R is the presumed upper bound on the variance of the measurement noise (should
		# there not be measurement noise, R will default to 1)

		# 1-delta is the required confidence, i.e. at termination the result will hold
		# with probability 1-delta (should the bounding values be correct)

		# n_restarts is the number of times the GPR hyperparameters will be re-initialized when
		# fitting (larger n_restarts indicates longer time to convergence, default value = 0 implying
		# that the hyperparameters will not be optimized.  In general, the hyperparameters needn't be
		# optimized as the algorithm will use the universal Matern Kernel or the squared exponential kernel)

		# n_init_samples is the number of samples the algorithm will start off with, the default will be 5,
		# and they will be randomly sampled from the feasible, hyper-rectangular space.  For problems
		# of a high dimension, seeding with a larger number of initial samples will likely expedite the
		# solution process, though it will lead to large runtimes for GPR regression in later stages (as
		# the number of samples explodes).

		# tolerance prescribes the required tolerance within which we would like the optimal value
		# to lie

		# length_scale: if the length_scale of the objective function is known apriori, then please input it
		# (for use in an RBF kernel), else it will be initialized to 1 (for which the kernel is universal)

		# constraint will be initialized to None unless otherwise populated.  We presume constraints is 
		# a list of constraint functions for the optimization problem at hand, each of which has the same
		# evaluation scheme, i.e.: constraints[i](x) = some float.  Here, i is the i-th constraint, and
		# x is a 1xN numpy array.  Additionally, we assume wlog that each constraint function is to be kept
		# negative, i.e. the constraints are constraint[i](x) <= 0.

		# avail_processors: the number of available processors for a parallelized computation speedup.
		# not implemented atm.

		# debug: Throws a debug flag in the solver wherever required.  Useful for debugging purposes

		# granularity: The granularity with which samples are chosen (uniformly between CVaR bounds)
		# for samples in the rollout process of the CVaR optimization procedure.

		# risk_prob: the alpha value for the CVaR analysis.

		# lb/ub: the lower and upper bounds of the objective function, respectively (i.e. the objective
		# function is guaranteed to be within this regime.)

		self.objective = objective        # Should output a real value, a sample of a random variable
		self.bounds = bounds
		self.beta = []
		self.B = B
		self.R = R
		self.delta = delta
		self.mu = []                      # Initializing fields to contain the mean function and
		self.sigma = []                   # covariance function respectively.
		self.dimension = bounds.shape[0]  # Dimensionality of the feasible space
		self.X_sample = None              # Instantiating a variable to keep track of all sampled, X values
		self.base_sample = None          # Instantiating a variable to keep track of all un-augmented sampled values
		self.Y_sample = None              # Instantiating a variable to keep track of all sampled, Y values
		self.cmax =  -1e6                 # Instantiating a variable to keep track of the current best value
		self.best_sample = None           # Instantiating a variable to keep track of the current best sample
		self.n_init_samples = n_init_samples
		self.n_restarts = n_restarts
		self.tol = tolerance
		self.max_val = []
		self.term_sigma = 0
		self.UCB_sample_pt = 0
		self.UCB_val = -1e6
		self.constraints = constraints
		self.length_scale = length_scale
		self.avail_processors = avail_processors
		self.debug = debug
		self.xbase = np.linspace(self.bounds[0,0], self.bounds[0,1], 500).tolist()
		self.risk_measure = risk_measure
		self.granularity = granularity
		self.alpha = risk_prob
		self.F = 1e5
		self.iteration = 0
		self.riskbounds = [(lb - (1-self.alpha)*ub)/self.alpha, ub]
		self.totalbounds = np.vstack((self.bounds, np.asarray(self.riskbounds).reshape(1,-1)))
		self.inner_obj = lambda x,s: s + max((x-s,0))/(1-self.alpha)

	def check_constraints(self,x):
		if self.constraints == None:
			return True
		else:
			flag = True
			for constraint in self.constraints:
				flag = flag and (constraint(x) <=0)
				if flag == False: break
			return flag

	def initialize(self, sample = None, observations = None, init_flag = False):
		# Initialize a starting set of samples and their y-values.  Initial sample size is set to 5

		'''
		The purpose of this script is to initialize a set of samples self.X_samples and observations
		self.Y_samples either from scratch or from a set of prior samples and observations
		'''

		if init_flag == False:
			sample_flag = False
			while not sample_flag:
				self.base_sample = np.random.uniform(self.bounds[:,0], self.bounds[:,1],
					size=(self.n_init_samples,self.dimension))
				met_constraints_flags = []
				for i in range(self.n_init_samples):
					met_constraints_flags.append(self.check_constraints(self.base_sample[i,:].reshape(1,-1)))
				sample_flag = sample_flag or all(met_constraints_flags)

			for i in range(self.n_init_samples):
				self.sample_point(self.base_sample[i,:].reshape(1,-1))
		elif sample is not None:
			self.base_sample = sample
			self.sample_point(prior_obs = observations)
		else:
			error('the initialization flag was set to False, but a prior sample set was not provided')

	def sample_point(self, sample_point, prior_obs = None):
		if prior_obs is None:
			'''
			No prior observation was provided
			'''
			if sample_point.shape[0] == self.dimension+1:
				'''
				This sample_point is one which contains the augmented variable 's' to be evaluated.  Code assumes the base sample
				set was augmented prior to this line of code.
				'''
				xval = sample_point[0,:-1].reshape(1,-1)
				s = sample_point[0,-1]
				self.X_sample = np.vstack((self.X_sample, sample_point.reshape(1,-1))) # Assumes self.X_sample was initialized prior
				observation = self.objective(xval)
				self.Y_sample = np.vstack((self.Y_sample, self.inner_obj(observation, s))) 
			else:
				'''
				We are being asked to sample a point without provision of an augmented variable 's' (rollout over a bunch of 's'
				and identify the corresponding expectation approximation)
				'''
				xval = sample_point.reshape(1,-1)
				observation = self.objective(xval)

			s_values = np.linspace(self.riskbounds[0], self.riskbounds[1], self.granularity).tolist()
			for s in s_values:
				Je_val = self.inner_obj(observation, s)
				self.X_sample = np.vstack((self.X_sample, np.hstack((xval,s)))) if self.X_sample else np.hstack((xval,s))
				self.Y_sample = np.vstack((self.Y_sample, Je_val)) if self.Y_sample else np.array([[Je_val]])
		else:
			'''
			A prior observation was provided that is a set of observations without augmented 's' values
			We will assume observations is a numpy array of size N_obs X 1
			'''
			np_obs = prior_obs.shape[0]
			s_values = np.linspace(self.riskbounds[0], self.riskbounds[1], self.granularity).tolist()
			for i in range(np_obs):
				xval = self.base_sample[i,:].reshape(-1,1)
				observation = prior_obs[i,0]
				for s in s_values:
					Je_val = self.inner_obj(observation, s)
					self.X_sample = np.vstack((self.X_sample, np.hstack((xval,s)))) if self.X_sample else np.hstack((xval,s))
					self.Y_sample = np.vstack((self.Y_sample, Je_val)) if self.Y_sample else np.array([[Je_val]])
		pass

	def propose_location(self, N_samples = 1e5):
		'''
		This function should propose a new sampling location and augment the self.base_sample list with this new sample
		The way it should go about this:
			1) Randomly sample (uniformly over the entire augmented space) N_samples samples
			2) Pick the sample which maximizes the lower confidence bound - should be defined within this function
			3) Use that sample as the starting point for a gradient descent minimizer.
			4) Takes the output of that minimizer as the sample point (should produce a sample point (x^*,s^*))
		'''

		# Calculate the lower confidence bound
		def LCB(x):
			return self.mu(x) - self.beta*self.sigma(x)

		# Helper function for the inner minimization problem
		def min_obj(xval,s):
			spt = np.hstack((xval,s))
			return LCB(spt)

		# The function that should be maximized in the gradient descent portion of this location proposal subroutine
		def inf_Je(x, min_pt_flag = False):
			s_values = np.linspace(self.riskbounds[0], self.riskbounds[1], 300).tolist()
			evaluations = [min_obj(x.reshape(1,-1), s) for s in s_values]
			best_s = s_values[evaluations.index(min(evaluations))]

			residual_fn = lambda s: min_obj(x.reshape(1,-1), s)

			min_inf = minimize(residual_fn, x0 = best_s, bounds = np.asarray(self.riskbounds).reshape(1,-1), method = 'L-BFGS-B')

			if min_pt_flag:
				return min_inf.fun, min_inf.x
			else:
				return min_inf.fun

		# Step 1: Randomly sample, over the entire augmented space, N_sampels samples
		init_sample = np.random.uniform(self.totalbounds[:,0], self.totalbounds[:,1], size = (N_samples,self.dimension+1))

		# Step 2: Find the sample that maximizes the LCB
		evaluations = [LCB(init_sample[i,:].reshape(1,-1)) for i in range(N_samples)]
		best_sample_index = evaluations.index(max(evaluations))
		best_sample = init_sample[best_sample_index,:].reshape(1,-1) 

		# Step 3: Use this sample as the seed for a gradient descent maximizer, the objective function
		# for which is inf_Je (i.e. we want to maximize Je, so we're going to minimize the negation)
		res = minimize(lambda x: -1*inf_Je(x), x0 = best_sample[0,:-1].reshape(1,-1), bounds = self.bounds, method = 'L-BFGS-B')

		# Step 4: Identify the optimal augmented state (x,s) by running through the inf_Je process again and outputting the minimizing s
		state_outputs = inf_Je(res.x, min_pt_flag = True)
		opt_state = np.hstack((res.x.reshape(1,-1), state_outputs[1].reshape(1,1)))

		# Step 5: Update F to reflect this new sample point and return the state to be sampled and the maximum, minimized lower bound 
		self.F = 2*self.beta*self.sigma(opt_state)
		return opt_state, -1*state_outputs[0]

	def UCB(self, x):
		# Calculate the Upper Confidence Bound for a value, x, based on the data-set, (x_sample, y_sample).
		# gpr is the Regressed Gaussian Process defining the current mean and standard deviation.

		# Returning the UCB based on the choice of beta.
		if self.check_constraints(x.reshape(1,-1)):
			return self.mu(x) + self.beta*self.sigma(x)
		else:
			return -100

	def propose_location(self, opt_restarts = 20, granularity = 50):
		pass
		
	def calc_musigma(self):
		self.Kn = self.kernel(self.X_sample)
		t = self.X_sample.shape[0]
		eta = 2/t
		self.KI = self.Kn + (1+eta)*np.identity(self.Kn.shape[0])
		self.Kinv = np.linalg.inv(self.KI)
		self.knx = lambda x: self.kernel(self.X_sample,x.reshape(1,-1))
		self.mu = lambda x: (self.knx(x).transpose() @ self.Kinv @ self.Y_sample)[0,0]
		self.sigma = lambda x: (1 - self.knx(x).transpose() @ self.Kinv @ self.knx(x))[0,0]

		innersqrt = np.linalg.det(self.KI)
		self.beta = self.B + self.R*math.sqrt(2*math.log(math.sqrt(innersqrt)/self.delta))
		pass

	def optimizer(self):
		'''
		This optimization subroutine should only be run  in the event that the user wants to optimize for the conditional value at risk
		of a set of distributions.  Steps are as follows:
			1) Every time the code takes a sample of the robustness meausre, it should roll out granularity evenly spaced evaluations of
			s + max(sample - s, 0)/(1-alpha).  Here, alpha (self.alpha) should be the confidence interval for the risk analysis
				a) The composite decision space should be the normal decision space + s (i.e. the optimization problem even over a one 
			dimensional objective function should be a 2-dimenaionsal optimization problem)
			2) After sampling and taking rollouts of the objective function, the code should fit a Gaussian Process to the augmented dataset
			of sample/measurement pairs 
			3) Now, call the mean function for the fitted Gaussian Process mu(x,s) and standard deviation sigma(x,s).  x is the decision variable
			and s is the sliding bound used in the CVaR analysis.  The system should proposed the next sample point x as the sample that maximizes
			the minimum value of the inner objective function over s.  I.e., whereas for normal UCB optimization the acquisition function is
			to maximize the UCB of the fitted Gaussian Process, in this case, the acquisition function aims to maximize the CVaR of the lower
			bound of the fitted Gaussian Process.
			4) After determining the new sample point, rinse and repeat until termination.
			5) The termination condition here is still the same termination condition, i.e. when F = 2*beta*sigma at the determined sample point x
		'''
		
		self.calc_musigma()
		'''
		Determine the next point to sample
		'''
		new_augmented_state, maxminval = self.propose_location()

		while self.F > self.tol:
			'''
			When sampling the new point, not only should the function sample the objective function, but it should also perform the rollouts
			and update the dataset accordingly as well.
			First call will sample the optimal, augmented state (x,s)
			Second call will sample the state (x) with rollouts over the possible values of (s)
			As we are going to sample this point (x) we need to add it to the list of base samples 
			'''
			self.base_sample = np.vstack((self.base_sample, new_augmented_state[0,:-1].reshape(1,-1)))
			self.sample_point(new_augmented_state)
			self.sample_point(new_augmented_state[0,:-1].reshape(1,-1))

			'''
			After sampling the new point, re-calculate the Gaussian Process and determine a new sample point
			'''
			self.calc_musigma()
			self.iteration+=1
			new_augmented_state, maxminval = self.propose_location()

			print('Finished iteration: %d'%self.iteration)
			print('Termination value at next sample point: %.4f'%self.F)
			print('Min-Max Value at next sample point: %.3f'%maxminval)

		pass

		

	def plot_approximation(self):
		'''
		Useful for debugging purposes only.  Will plot the 1-d objective function over the feasible space and plot the GPR approximation
		'''
		xbase = np.linspace(self.bounds[0,0], self.bounds[0,1], 500)
		true_val = [self.debug(np.array([[x]])) for x in xbase]
		ub = [self.mu(np.array([[x]])) + self.beta*self.sigma(np.array([[x]])) for x in xbase]
		lb = [self.mu(np.array([[x]])) - self.beta*self.sigma(np.array([[x]])) for x in xbase]
		maxval = max(ub)
		maxloc = xbase[ub.index(maxval)]
		fig, ax = plt.subplots()
		ax.plot(xbase, true_val, lw = 5, color = colors['black'])
		ax.fill_between(xbase, lb, ub, alpha = 0.5, color = colors['gray'])
		ax.hlines(maxval, xmin = xbase[0], xmax = xbase[-1], lw = 5, color = colors['red'], ls = '--')
		ax.vlines(maxloc, ymin = ax.get_ylim()[0], ymax = ax.get_ylim()[1], lw = 5, color = colors['red'], ls = '--')
		ax.set_xlabel(r'$x$')
		ax.set_ylabel(r'$y$', rotation = 0)
		sample_flag = input('Show samples this time? (y/n): ')
		if sample_flag == 'y':
			sample_flag = None
			ax.scatter(self.X_sample, self.Y_sample, marker = 'x', s = 30, color = colors['green'])
		plt.show()
		command = input('Enter debugger mode? (y/n): ')
		if command == 'y':
			command = None
			pdb.set_trace()
		pass

	

