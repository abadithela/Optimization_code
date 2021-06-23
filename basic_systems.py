import numpy as np
import matplotlib.pyplot as plt
import time
import pdb
import cvxpy as cp

class linear_sys():
	def __init__(self, init_state = np.zeros((2,1)),
		A = np.array([[0,1],[0,0]]), B = np.array([[0,1]]).transpose(), dt = 0.01):
		self.A = A
		self.B = B
		self.x = init_state
		self.xhist = init_state
		self.dt = dt
		self.uhist = []

	def initialize(self, state):
		self.x = state
		self.xhist = state
		pass

	def controller(self):
		'''
		This can be whatever function, it just has to produce an input vector of appropriate size (Mx1)
		the specific function 
		'''
		pass

	def dynamics(self):
		return self.A @ self.x + self.B @ self.controller()

	def simulate(self, steps = 20):
		for i in range(steps):
			self.x = self.x + self.dynamics()*self.dt
			self.xhist = np.hstack((self.xhist, self.x))
			self.uhist.append(self.controller())
		pass

class unicycle():
	def __init__(self, init_state = np.zeros((3,1)), dt = 0.01):
		self.x = init_state
		self.xhist = init_state
		self.uhist = []
		self.dt = dt

	def controller(self):
		pass

	def g(self):
		return np.array([[np.cos(self.x[2,0]), 0],[np.sin(self.x[2,0]),0],[0,1]])

	def dynamics(self,ctrl_input):
		'''
		Assumes the controller method for this class is populated by a method that outputs
		a vector of two elements for the control input (the linear and angular velocity,
		the angular velocity should be in radians)
		'''
		xdot = np.array([[
			ctrl_input[0,0]*np.cos(self.x[2,0]),
			ctrl_input[0,0]*np.sin(self.x[2,0]),
			ctrl_input[1,0]
			]]).transpose()
		return xdot

	def simulate(self, steps = 20, spacing = 100):
		self.interior_dt = self.dt/spacing
		for tsteps in range(steps):
			ctrl_input = self.controller()
			self.uhist.append(ctrl_input)
			for splices in range(spacing):
				self.x = self.x + self.dynamics(ctrl_input = ctrl_input)*self.interior_dt
			self.xhist = np.hstack((self.xhist, self.x))

class nonlinear():
	def __init__(self, init_state = np.zeros((3,1)), dt = 0.01):
		self.x = init_state
		self.xhist = init_state
		self.uhist = []
		self.dt = dt

	def controller(self):
		pass

	def f(self):
		pass

	def g(self):
		pass

	def dynamics(self,ctrl_input):
		'''
		Assumes the controller method for this class is populated by a method that outputs
		a vector of two elements for the control input (the linear and angular velocity,
		the angular velocity should be in radians)
		'''
		xdot = self.f() + self.g() @ ctrl_input
		return xdot

	def simulate(self, steps = 20, spacing = 1000):
		self.interior_dt = self.dt/spacing
		for tsteps in range(steps):
			ctrl_input = self.controller()
			self.uhist.append(ctrl_input)
			for splices in range(spacing):
				self.x = self.x + self.dynamics(ctrl_input = ctrl_input)*self.interior_dt
			self.xhist = np.hstack((self.xhist, self.x))

def QP_CBF(state, udes, f, g, h, dhdx, alpha):
	'''
	Filters the desired control input udes, against the CBF condition specified by the CBF, h,
	and the class-kappa function alpha.  This assumes control affine dynamics with
	xdot = f(x) + g(x)*u.  The current state is reported as state.  The assumptions behind the code
	is as follows:
		a) f(state) is a viable function call and outputs an Nx1 vector
		b) g(state) is a viable function call and outputs an NxM vector
		c) (same for h and dhdx) assumes all vectors are column vectors
		d) alpha(h(state)) is a viable function call
		e) assumes udes is an Mx1 desired control input
	'''
	Bmat = g(state)
	m = Bmat.shape[1]
	u = cp.Variable((m,1))
	cost = cp.Minimize((1/2)*cp.quad_form(u,np.identity(m)) - udes.transpose() @ u)
	constr = [dhdx(state).transpose() @ f(state) + dhdx(state).transpose() @ g(state) @ u >= -alpha(h(state))]
	prob = cp.Problem(cost,constr)
	prob.solve()
	return u.value

if __name__ == '__main__':
	f = lambda x: np.identity(2) @ x
	g = lambda x: np.array([[0,1]]).transpose()
	cbf = lambda x: 5
	dhdx = lambda x: np.zeros((2,1))
	alpha = lambda x: 2*x
	print(QP_CBF(state = np.zeros((2,1)), udes = 0, f = f, g = g, h = cbf, dhdx = dhdx, alpha = alpha))
