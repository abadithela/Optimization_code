import numpy as np
import time
import math
from basic_systems import linear_sys
import scipy.linalg as LA
import types
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


if __name__ == '__main__':
	portray_system()