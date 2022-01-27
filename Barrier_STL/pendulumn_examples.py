import numpy as np
import time
import math
from basic_systems import linear_sys, pendulumn
import scipy.linalg as LA
import types
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
	system.P = P

	controller = lambda x: -K @ system.x
	dzdt = lambda x: (A - B @ R.inv() @ B.transpose() @ P) @ system.x
	system.controller = types.MethodType(controller, system)
	system.dzdt = types.MethodType(dzdt, system)
	return system

def robustness(system = initialize_system()):
	'''
	This function calculate sthe robustness of a given input signal saved in the state trajectory
	of a system class, the default value for which is provided.
	'''
	norm_seq = [0.2 - np.abs(system.xhist[0,i] - math.pi/2) for i in range(system.xhist.shape[1])]
	return max(norm_seq)

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
	c = robustness(system = system)
	print('The robustness of this system trajectory is %.4f'%c)
	plt.show()
	pass

# Constructing Barrier function for requirement of eventually reaching the upright position in 	T < 2 seconds.
L = 1 # Lipschitz constant associated with this requirement

def get_h(L,z,x, c):
	h = lambda x: c**2 - L**2*(LA.norm(x - z))**2
	return h

def construct_cbf(L, z, x, c, dzdt):
	'''
	Constructing barrier function from Lipschitz constant L and signal z
	'''
	pendulum = pendulumn(init_state = np.array([[math.pi,0]], dt = 0.01))
	f = pendulum.f()
	g = pendulum.g()
	cbf = get_h(L, z, x, c)
	dhdx = lambda x: 2*L**2 * (x - z) @ (-f + dzdt)
	dhdt = lambda x: 2*L**2 * (x - z) @ (-g)
	alpha = lambda x: 1*x
	return f,g,cbf, dhdx, dhdt, alpha

def get_CBF_controller(f,g,h,dhdx, alpha):
	u = QP_CBF(state = np.zeros((2,1)), udes = 0, f = f, g = g, h = h,
		dhdx = dhdx, alpha = alpha, dhdt = dhdt)
	return u


if __name__ == '__main__':
	portray_system()
