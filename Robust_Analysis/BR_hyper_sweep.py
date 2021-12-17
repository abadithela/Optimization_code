import numpy as np
import scipy
import sys
import math
from UCB_Bayes import UCBOptimizer
import matplotlib
matplotlib.use('TKAgg', force=True)
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib import rc, font_manager, cm
from matplotlib.patches import Circle
import pdb
import pickle


def objective(x, noise = 0):
	x.reshape(1,-1)
	return 2/(np.linalg.norm(x - np.array([[1,1]]))**2 + 1) + 3/(np.linalg.norm(x + np.array([[1,1]]))**2 + 1) + np.random.normal(loc = 0, scale = noise)

def plot_objective():
	xbase = np.linspace(-2,2,100)
	ybase = np.linspace(-2,2,100)
	X,Y = np.meshgrid(xbase,ybase)
	Z = np.zeros((100,100))
	for i in range(100):
		for j in range(100):
			x = np.array([[X[i,j],Y[i,j]]])
			Z[i,j] = objective(x)
	fig, ax = plt.subplots(subplot_kw = dict(projection = '3d'))
	ax.plot_surface(X,Y,Z)
	print('Maximum function value occurs at [-1,-1] with value: %.3f'%(objective(np.array([[-1,-1]]))))
	ax.set_title(r'$\mathrm{Objective~Function}$')
	plt.show()
	pass

def optimize(B = 1, R = 1, debug = None):
	bounds = np.hstack((-2*np.ones((2,1)),2*np.ones((2,1))))
	optimizer = UCBOptimizer(objective = lambda x: objective(x, noise = 0.2), B = B, R = R, delta = 1e-6, tolerance = 0.3, n_init_samples = 10,
		bounds = bounds, verbose = False, debug = debug)
	optimizer.initialize()
	optimizer.optimize()
	true_max = objective(np.array([[-1,-1]]))
	print('Maximum function value occurs at [-1,-1] with value: %.3f'%(true_max))
	print('Calculated maximum greater than true function maximum: %s'%(optimizer.UCB_val >= true_max))
	print('Calculated maximum within tolerance bounds: %s'%(np.abs(optimizer.UCB_val - true_max) <= 0.3))
	return optimizer

def plot_B_sweep(N_samples = 60, pfile = None):
	if pfile is None:
		Bvals = [0.5, 1, 1.5, 2, 2.5, 3]
		labels = [r'$%s$'%B for B in Bvals]
		final_values = [[] for B in Bvals]
		iteration_count = [[] for B in Bvals]
		for idx,B in enumerate(Bvals):
			for sample in range(N_samples):
				optimizer = optimize(B = B)
				final_values[idx].append(optimizer.UCB_val)
				iteration_count[idx].append(optimizer.final_iteration)
				print(' ')
				print('Finished sample %i/%i for B value %d'%(sample+1,N_samples,B))
				print(' ')
	else:
		information = pickle.load(pfile)
		final_values = information[0]
		iteration_count = information[1]

	pickle.dump([final_values, iteration_count], open('B_sweep_information.txt', 'wb'))
	fig, ax = plt.subplots()
	ax.boxplot(final_values)
	ax.set_xticklabels(labels)
	ax.set_xlabel(r'$B$')
	ax.set_title(r'$\mathrm{Spread~of~Results~vs~Increasing~RKHS~norm~B}$')
	plt.savefig(fname = 'Figures/B_sweep_results.jpg', bbox_inches = 'tight', pad_inches = 0)
	plt.show()

	fig, ax = plt.subplots()
	ax.boxplot(iteration_count)
	ax.set_xticklabels(labels)
	ax.set_xlabel(r'$B$')
	ax.set_title(r'$\mathrm{Final~Iteration~Spread~vs~Increasing~RKHS~norm~B}$')
	plt.savefig(fname = 'Figures/B_sweep_iterations.jpg', bbox_inches = 'tight', pad_inches = 0)
	plt.show()
	pass

if __name__ == '__main__':
	plot_B_sweep(N_samples = 60)
