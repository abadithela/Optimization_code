import numpy as np
import scipy
import sys
import math
from Bayes_Opt import CVaR_optimizer
from UCB_Bayes import UCBOptimizer
import matplotlib
matplotlib.use('TKAgg', force=True)
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib import rc, font_manager, cm
from matplotlib.patches import Circle
import pdb
import pickle

matplotlib.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath,amssymb}']
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
plt.rcParams.update({'font.size':24})
plt.rcParams["figure.figsize"] = (12.8,8)

def mean_func(x, mean = 0):
	return mean

def sigma_func(x, sigma = 0.5):
	return sigma

def example_obj(x):
	return mean_func(x) + np.random.uniform(low = -2*sigma_func(x), high = 2*sigma_func(x))

def CVaR_objective(s, x = np.random.uniform(low = -1, high = 1), alpha = 0.95):
	return (1-alpha)*s[0,0] + max((x - s[0,0], 0))

def plot_example():
	fig, ax = plt.subplots()
	xbase = np.linspace(-3,3,50).tolist()
	mean = [mean_func(x) for x in xbase]
	ub = [mean_func(x) + 2*sigma_func(x) for x in xbase]
	lb = [mean_func(x) - 2*sigma_func(x) for x in xbase]
	ax.plot(xbase, mean, lw = 4, color = colors['black'], label = r'$\mathrm{mean}$')
	ax.fill_between(xbase, lb, ub, lw = 2, alpha = 0.75, color = colors['gray'], label = r'$\pm 2\sigma$')
	ax.grid(lw = 3, alpha = 0.5)
	ax.legend(loc = 'best')
	plt.show()
	pass

def calc_CVaR():
	optimizer = CVaR_optimizer(objective = example_obj, B = 1, R = 1, delta = 1e-6, tolerance = 0.02, risk_prob = 0.95,
		bounds = np.array([[-1,1]]), granularity = 50, n_init_samples = 20, debug = True, verbose = True)
	optimizer.initialize()
	optimizer.optimize()
	pass

def CVaR_uniform(alpha, lx, ux):
	return 1/(1-alpha)*(alpha*lx + (1 - (1-alpha)**2)/2*(ux-lx))

def true_uniform_check(x, lb = -1, ub = 1, alpha = 0.95):
	if x >= ub:
		return (1-alpha)*x
	elif lb <= x <= ub:
		return (1-alpha)*x + 1/((ub-lb))*((ub**2 - x**2)/2 - x*(ub-x))
	else:
		return (1-alpha)*x + 1/((ub-lb))*((ub**2 - lb**2)/2 - x*(ub-lb))

def check_CVaR_calc(alpha = 0.95):
	bounds = [-1,1]
	riskbounds = [(bounds[0] - (1-alpha)*bounds[1])/alpha, bounds[1]]
	optimizer = UCBOptimizer(objective = lambda x: -1*CVaR_objective(x, alpha = alpha), B = 0.5, R = 0.5, delta = 1e-6, 
		tolerance = 0.02, bounds = np.asarray(riskbounds).reshape(1,-1), n_init_samples = 20, debug = lambda x: -1*true_uniform_check(x))
	optimizer.initialize()
	optimizer.optimize()

def maxgate(x):
	return max((x,0))

def bound(x, lx, ux):
	return max((min((x,ux)), lx))

def concentration_ineq(n = 30, alpha = 0.05, delta = 1e-6):
	samples = [np.random.uniform(-1,1) for i in range(n)] + [-1,1]
	samples.sort()
	ub_samples = samples[1:]
	lb_samples = samples[:-1]
	upper_bnd = ub_samples[-1] - 1/alpha*sum([(ub_samples[i+1] - ub_samples[i])*maxgate((i+1)/n - np.sqrt(np.log(1/delta)/(2*n)) - (1-alpha))
		for i in range(n)])
	print(upper_bnd)
	print([maxgate((i+1)/n - np.sqrt(np.log(1/delta)/(2*n)) - (1-alpha)) for i in range(n)])
	pdb.set_trace()
	pass

def plot_CVaR_function(spacing = 200, N = 5000, alpha = 0.95):
	bounds = [0,20]
	riskbounds = [(bounds[0] - (1-alpha)*bounds[1])/alpha, bounds[1]]
	samples = [bound(np.random.geometric(0.2), bounds[0], bounds[1]) for i in range(N)]
	fig, ax = plt.subplots()
	sbase = np.linspace(bounds[0],bounds[1], spacing).tolist()
	output = [None for i in range(spacing)]
	for idx,s in enumerate(sbase):
		output[idx] = sum([CVaR_objective(s = np.array([[s]]), x = x, alpha = alpha) for x in samples])/(N*(1-alpha))

	ax.plot(sbase, output, color = colors['black'], lw = 3, label = r'$J(s)$')
	ax.grid(alpha = 0.5, lw = 3)
	ax.legend(loc = 'best')
	ax.set_xlabel(r'$s$')
	plt.show()
	
	pass






if __name__ == '__main__':
	# plot_example()
	# calc_CVaR()
	# print(CVaR_uniform(0.05,0,1))
	# check_CVaR_calc()
	# concentration_ineq()
	plot_CVaR_function()