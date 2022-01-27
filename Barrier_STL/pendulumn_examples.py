import numpy as np
import time
import math

import control as ct
import cvxpy as cvx
from basic_systems import QP_CBF

'''
All Matplotlib plotting basics
'''
import matplotlib
matplotlib.use('TKAgg', force=True)
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib import rc, font_manager, cm
matplotlib.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath,amssymb}']
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
plt.rcParams.update({'font.size':28})
plt.rcParams["figure.figsize"] = (12.8,8)

def initialize_system():
	system = linear_sys(init_state = np.array([[math.pi,0]]).transpose())

	Q = np.diag(np.array([10,1]))
	R = np.array([[1]])
	A = system.A
	B = system.B
	P = LA.solve_continuous_are(A,B,Q,R)
	K = B.transpose() @ P

	controller = lambda x: -K @ system.x
	system.controller = types.MethodType(controller, system)
	return system

def portray_system(system = initialize_system(), horizon = 500):
	'''
	This function should portray the state history of a given example pendulumn system
	for a time period over which it's simulated (provided as an input to this function )
	'''
	system.simulate(steps = horizon)
	fig, ax = plt.subplots()
	times = [i*system.dt for i in range(horizon+1)]
	ax.plot(times, system.xhist[0,:], lw = 3, color = colors['blue'], label = r'$\theta$')
	ax.plot(times, system.xhist[1,:], lw = 3, color = colors['gold'], label = r'$\dot \theta$')
	ax.set_xlabel(r'$\mathrm{time}$')
	ax.set_title(r'$\mathrm{Example~Trajectory}$')
	ax.grid(lw = 3, alpha = 0.5)
	ax.legend(loc = 'best')
	plt.show()
	pass

# Constructing Barrier function for requirement of eventually reaching the upright position in 	T < 2 seconds.
L = 1 # Lipschitz constant associated with this requirement

def get_h(L,z):
	pass

def construct_cbf(L, z):
	'''
	Constructing barrier function from Lipschitz constant L and signal z
	'''
	f = lambda x: np.identity(2) @ x
	g = lambda x: np.array([[0,1]]).transpose()
	cbf = get_h(L, z)
	dhdx = lambda x: np.zeros((2,1))
	alpha = lambda x: 2*x
	return f,g,cbf, dhdx, alpha

def get_CBF_controller(f,g,h,dhdx, alpha)
	u = QP_CBF(state = np.zeros((2,1)), udes = 0, f = f, g = g, h = h, dhdx = dhdx, alpha = alpha)
	return u


if __name__ == '__main__':
	portray_system()
