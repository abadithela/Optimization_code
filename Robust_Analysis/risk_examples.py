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

def CVaR_objective(s, x = np.random.uniform(low = -1, high = 1), alpha = 0.95):
	return (1-alpha)*s[0,0] + max((x - s[0,0], 0))

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
	'''
	Assumes x is a numpy array
	'''
	obs = x.reshape(-1,).tolist()
	return [max((min((value,ux)), lx)) for value in obs]

def concentration_ineq(n = 400, alpha = 0.05, delta = 1e-6):
	samples = [np.random.uniform(-1,1) for i in range(n)] + [-1,1]
	samples.sort()
	ub_samples = samples[1:]
	lb_samples = samples[:-1]
	upper_bnd = ub_samples[-1] - 1/alpha*sum([(ub_samples[i+1] - ub_samples[i])*maxgate((i+1)/n - np.sqrt(np.log(1/delta)/(2*n)) - (1-alpha))
		for i in range(n)])
	print(upper_bnd)
	# print([maxgate((i+1)/n - np.sqrt(np.log(1/delta)/(2*n)) - (1-alpha)) for i in range(n)])
	pdb.set_trace()
	pass

def plot_CVaR_function(spacing = 100, N = 1000, alpha = 0.95, fname = 'Uniform', repetition = 1000, opt_analysis = None):
	plot_values = [None for i in range(repetition)]
	min_values = [None for i in range(repetition)]
	bounds = [-5,5]
	riskbounds = [(bounds[0] - (1-alpha)*bounds[1])/alpha, bounds[1]]
	sbase = np.linspace(bounds[0],bounds[1], spacing).tolist()

	def sample_gen(sample_count):
		return bound(np.random.uniform(bounds[0],bounds[1], size = (1,sample_count)), bounds[0], bounds[1])

	for i in range(repetition):
		samples = sample_gen(N)
		output = [None for i in range(spacing)]
		for idx,s in enumerate(sbase):
			output[idx] = sum([CVaR_objective(s = np.array([[s]]), x = x, alpha = alpha) for x in samples])/(N*(1-alpha))
		plot_values[i] = output
		min_values[i] = min(output)

	
	true_samples = sample_gen(5000)
	true_values = [sum([CVaR_objective(s = np.array([[s]]), x = x, alpha = alpha) for x in true_samples])/(5000*(1-alpha)) for s in sbase]
	fig, ax = plt.subplots()
	ax.plot(sbase, plot_values[0], color = colors['red'], lw = 3, label = r'$\hat J(s)$')
	for i in range(repetition-1):
		ax.plot(sbase,plot_values[i+1], color = colors['red'], lw = 3)
	ax.plot(sbase, true_values, color = colors['black'], lw = 3, label = r'$J(s)$')
	ax.grid(alpha = 0.5, lw = 3)
	ax.legend(loc = 'best')
	ax.set_xlabel(r'$s$')
	ax.set_title(r'$\mathrm{' + fname + '~distribution}$')
	plt.show()
	fig, ax = plt.subplots()
	differences = [value - min(true_values) for value in min_values]
	ax.hist(differences, bins = 20)
	plt.show()
	pdb.set_trace()
	pass

def check_sample_scheme(s_samples = 30, alpha = 0.95, N = 30):
	bounds = [-1,1]
	riskbounds = [(bounds[0] - (1-alpha)*bounds[1])/alpha, bounds[1]]
	samples = bound(np.random.normal(size = (1,N)), bounds[0], bounds[1])
	max_sample = max(samples)
	min_sample = min(samples)
	R = (max_sample - min_sample)**2/8
	print(R)
	optimizer = UCBOptimizer(objective = lambda s: -1*CVaR_objective(s = s, x = np.random.uniform(bounds[0],bounds[1]), alpha = alpha),
		B = 1, R = R, delta = 1e-6, tolerance = 0.05, bounds = np.asarray(riskbounds).reshape(1,-1))
	rand_s = np.random.uniform(riskbounds[0], riskbounds[1])
	rand_x = samples[np.random.randint(N)]
	obs = -1*CVaR_objective(s = np.array([[rand_s]]), x = rand_x, alpha = alpha)
	optimizer.X_sample = np.array([[rand_s]])
	optimizer.Y_sample = np.array([[obs]])
	for i in range(s_samples-1):
		rand_s = np.random.uniform(riskbounds[0], riskbounds[1])
		rand_x = samples[np.random.randint(N)]
		obs = -1*CVaR_objective(s = np.array([[rand_s]]), x = rand_x, alpha = alpha)
		optimizer.X_sample = np.vstack((optimizer.X_sample,np.array([[rand_s]])))
		optimizer.Y_sample = np.vstack((optimizer.Y_sample,np.array([[obs]])))

	optimizer.calc_musigma()
	true_sample = bound(np.random.normal(size = (1,5000)), bounds[0],bounds[1])
	optimizer.debug = lambda s: -1*sum([(1-alpha)*s + max((x-s,0)) for x in true_sample])/5000
	optimizer.plot_approximation()
	pass

def check_single_opt(alpha = 0.95):
	bounds = [-1,1]
	def sample_gen(sample_count):
		return bound(np.random.normal(0,0.3, size = (1,sample_count)), bounds[0], bounds[1])

	def objective(s):
		return CVaR_objective(s, x = sample_gen(1), alpha = alpha)


	riskbounds = [(bounds[0] - (1-alpha)*bounds[1])/alpha, bounds[1]]
	optimizer = UCBOptimizer(objective = lambda s: -1*objective(s),
		B = 1, R = 0.5, delta = 1e-6, tolerance = 0.03, bounds = np.asarray(riskbounds).reshape(1,-1), n_init_samples = 1)
	optimizer.initialize()
	optimizer.optimize()

	sbase = np.linspace(riskbounds[0], riskbounds[1], 500).tolist()
	true_samples = sample_gen(5000)
	true_values = [sum([CVaR_objective(s = np.array([[s]]), x = x, alpha = alpha) for x in true_samples])/5000 for s in sbase]

	ub = [-1*(optimizer.mu(np.array([[s]])) - optimizer.beta*optimizer.sigma(np.array([[s]]))) for s in sbase]
	lb = [-1*(optimizer.mu(np.array([[s]])) + optimizer.beta*optimizer.sigma(np.array([[s]]))) for s in sbase]
	fig, ax = plt.subplots()
	ax.plot(sbase, true_values, lw = 5, color = colors['black'], label = r'$\mathrm{CVaR}(x)$')
	ax.fill_between(sbase, lb, ub, alpha = 0.5, color = colors['gray'], label = r'$\mathrm{GP~bounds}$')
	# ax.hlines(maxval, xmin = xbase[0], xmax = xbase[-1], lw = 5, color = colors['red'], ls = '--')
	# ax.vlines(maxloc, ymin = ax.get_ylim()[0], ymax = ax.get_ylim()[1], lw = 5, color = colors['red'], ls = '--')
	ax.set_xlabel(r'$s$')
	ax.set_ylabel(r'$y$', rotation = 0)
	ax.scatter(optimizer.X_sample, -1*optimizer.Y_sample, marker = 'x', s = 50, color = colors['green'], label = r'$\mathrm{samples}$')
	ax.set_title(r'$\mathrm{BayesOpt~CVaR~calculation~for~a~normal~distribution}$')
	ax.grid(lw = 3, alpha = 0.5)
	ax.legend(loc = 'best')
	plt.tight_layout()
	plt.savefig(fname = 'Figures/CVaR_normal_GP.jpg', bbox_inches = 'tight', pad_inches = 0.1, dpi = 200)
	plt.show()
	pass

if __name__ == '__main__':
	# plot_CVaR_function()
	# check_sample_scheme()
	check_single_opt()
	# concentration_ineq()