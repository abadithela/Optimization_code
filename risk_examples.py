import numpy as np
import scipy
import sys
import math
from Bayes_Opt import CVaR_optimizer
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

def sigma_func(x, sigma = 1):
	return sigma

def example_obj(x):
	return np.random.normal(loc = mean_func(x), scale = sigma_func(x))

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
	


if __name__ == '__main__':
	plot_example()