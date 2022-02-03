import numpy as np
import time
import math
import scipy.linalg as LA
import types
import control as ct
import sys
import pdb
print(sys.path)
sys.path.append('..')
import cvxpy as cvx
from basic_systems import QP_CBF, linear_sys

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

class pendulumn():
	def __init__(self, expert_system, c, init_state = np.zeros((2,1)), dt = 0.01):
		self.x = init_state
		self.xhist = init_state
		self.uhist = None
		self.dt = dt
		self.pi = math.pi
		self.gravity = -9.81
		self.l = 1
		self.L = 1 # Lipschitz constant
		self.t = 0
		self.alpha = lambda x: 1*x
		self.expert_system = expert_system
		self.c = c # Robustness of expert Trajectory

	def controller(self):
		'''
		To be populated with whatever controller you would like to steer this system with
		'''
		u = QP_CBF(state = np.zeros((2,1)), udes = np.array([[0]]), f = self.f, g = self.g, hval = self.h, dhdx = self.dhdx, alpha = self.alpha, dhdt = self.dhdt)
		return u

	def h(self):
		time_step = int(self.t/self.dt)
		return self.c**2 - self.L**2 * np.linalg.norm(self.x-self.expert_system.xhist[:,time_step].reshape(-1,1))**2

	'''
	Constructing dhdx and dhdt in a similar fashion
	'''
	def dhdx(self):
		time_step = int(self.t/self.dt)
		zt = self.expert_system.xhist[:,time_step].reshape(-1,1)
		dhdx = -2*self.L**2 * (self.x - zt)
		return dhdx

	def dzdt(self):
		time_step = int(self.t/self.dt)
		zt = self.expert_system.xhist[:,time_step].reshape(-1,1)
		# pdb.set_trace()
		dzdt = (self.expert_system.A - self.expert_system.B @ self.expert_system.B.transpose() @ self.expert_system.P)@zt
		return dzdt

	def dhdt(self):
		time_step = int(self.t/self.dt)
		zt = self.expert_system.xhist[:,time_step].reshape(-1,1)
		dzdt = self.dzdt()
		dhdt = 2*self.L**2 * (self.x - zt).transpose() @ (dzdt)
		return dhdt

	def f(self):
		return np.array([[self.x[1,0], -self.gravity/self.l*np.sin(self.x[0,0])]]).transpose()

	def g(self):
		return np.array([[0,1]]).transpose()

	def dynamics(self,ctrl_input):
		'''
		Assumes the controller method for this class is populated by a method that outputs
		a vector of two elements for the control input (the linear and angular velocity,
		the angular velocity should be in radians)
		'''
		xdot = self.f() + self.g() @ ctrl_input
		return xdot

	def reset_angle(self):
		result = self.x[0,0] % (2*self.pi)
		if result <= math.pi:
			self.x[0,0] = result
		else:
			self.x[0,0] = result - 2*math.pi
		pass


	def simulate(self, steps = 20, spacing = 100):
		self.interior_dt = self.dt/spacing
		for tsteps in range(steps):
			# pdb.set_trace()
			self.t = tsteps*self.dt
			ctrl_input = self.controller()
			# pdb.set_trace()
			self.uhist = np.hstack((self.uhist, ctrl_input)) if self.uhist is not None else ctrl_input
			for splices in range(spacing):
				self.x = self.x + self.dynamics(ctrl_input = ctrl_input)*self.interior_dt
			self.reset_angle()
			self.xhist = np.hstack((self.xhist, self.x))

	def portray_values(self):
		fig, ax = plt.subplots()
		times = [i*self.dt for i in range(self.xhist.shape[1])]
		ax.plot(times, self.xhist[0,:], lw = 3, color = colors['blue'], label = r'$\theta$')

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
	# system.simulate(steps = horizon)
	fig, ax = plt.subplots()
	times = [i*system.dt for i in range(system.xhist.shape[1])]
	ax.plot(times, system.xhist[0,:], lw = 3, color = colors['blue'], label = r'$\theta$')
	ax.plot(times, system.xhist[1,:], lw = 3, color = colors['gold'], label = r'$\dot \theta$')
	ax.hlines(math.pi/2, times[0], times[-1], color = colors['black'], label = r'$\mathrm{desired}$')
	ax.set_xlabel(r'$\mathrm{time}$')
	ax.set_title(r'$\mathrm{Example~Trajectory}$')
	ax.grid(lw = 3, alpha = 0.5)
	ax.legend(loc = 'best')
	c = robustness(system = system)
	print('The robustness of this system trajectory is %.4f'%c)
	plt.show()
	robustness(system = system)
	pass

# Constructing Barrier function for requirement of eventually reaching the upright position in 	T < 2 seconds.
'''
Construct the corresponding barrier function.
	Note: Convert the time to a time-step so that the appropriate indexed
	state can be identified in expert_system.xhist
'''
# This returns the value of h at time t:


def get_CBF_controller(f,g,h,dhdx, dhdt, alpha):
	u = QP_CBF(state = np.zeros((2,1)), udes = 0, f = f, g = g, hval = h,
		dhdx = dhdx, alpha = alpha, dhdt = dhdt)
	return u

def true_system_simulation():
	'''
	Steps:
		1) Initialize the pendulumn system
		2) Construct the Pendulumn System's controler
			a) The controller should be a QP-CBF controller that filters against
			a provided control input
			b) This controller should follow the robustness setup from prior.
	'''
	pend_sys = pendulumn(init_state = np.array([[math.pi,0]]).transpose(), dt = 0.01)

	'''
	Calculate the robustness of the linear system expert trajectory
	'''
	expert_system = initialize_system()
	expert_system.simulate(steps = 500)
	c = robustness(system = expert_system)


if __name__ == '__main__':
	# portray_system()
	expert_system = initialize_system() # Linear system
	expert_system.simulate(steps = 200)
	# portray_system(system=expert_system)
	c = robustness()
	L = 1
	z = expert_system.xhist
	pendulum = pendulumn(expert_system, c, init_state = np.array([[np.pi],[0]]), dt = 0.01) # True system

	pendulum.simulate(steps=200)
	portray_system(system=pendulum)
