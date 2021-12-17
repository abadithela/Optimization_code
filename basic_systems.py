import numpy as np
import matplotlib.pyplot as plt
import math
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
		self.uhist = None
		self.dt = dt
		self.pi = math.pi
		self.P = np.diag([5,5,0.2])

	def controller(self):
		'''
		To be populated with whatever controller you would like to steer this system with
		'''
		pass

	def base_controller(self, goal, c, alpha):
		'''
		Will populate this method with a baseline Lyapunov controller that gets to a goal.  Goal will be
		to produce a control input u that minimizes a baseline Lyapunov function with a decay constant specified by alpha
		'''
		error = self.x - goal
		V = 1/2*error.transpose() @ self.P @ error
		if V <= c:
			return np.zeros((2,1))
		else:	
			LgV = error.transpose() @ self.P @ self.g()
			return np.linalg.pinv(LgV)*(-alpha*V)

	def g(self):
		return np.array([[np.cos(self.x[2,0]), 0],[np.sin(self.x[2,0]),0],[0,1]])

	def dynamics(self,ctrl_input):
		'''
		Assumes the controller method for this class is populated by a method that outputs
		a vector of two elements for the control input (the linear and angular velocity,
		the angular velocity should be in radians)
		'''
		xdot = self.g() @ ctrl_input
		return xdot

	def reset_angle(self):
		result = self.x[2,0] % (2*self.pi)
		if result <= math.pi:
			self.x[2,0] = result
		else:
			self.x[2,0] = result - 2*math.pi
		pass


	def simulate(self, steps = 20, spacing = 100):
		self.interior_dt = self.dt/spacing
		for tsteps in range(steps):
			ctrl_input = self.controller()
			self.uhist = np.hstack((self.uhist, ctrl_input)) if self.uhist is not None else ctrl_input
			for splices in range(spacing):
				self.x = self.x + self.dynamics(ctrl_input = ctrl_input)*self.interior_dt
			self.reset_angle()
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

class cart_pendulum():
	def __init__(self, init_state = np.zeros((4,1)), dt = 0.01, params = [2,1,0.2,9.81]):
		'''
		This system assumes a one dimensional input, and a cart/pendulum on a plane
		Furthermore, the parameter list indicates the parameters for the cart, with the
		following setup:
			a) M = params[0] = mass of the cart
			b) m = params[1] = mass of the pendulum ball
			c) l = params[2] = length of the pendulum arm
			d) g = params[3] = gravity
		Assumes the state vector has the following format: x = [x, dx, theta, dtheta]
		'''
		self.x = init_state
		self.xhist = init_state
		self.uhist = []
		self.dt = dt
		self.M = params[0]
		self.m = params[1]
		self.l = params[2]
		self.g = params[3]
		self.get_linear_A()
		self.get_linear_B()

	def controller(self):
		pass

	def f(self):
		vec = np.zeros((4,1))
		vec[0,0] = self.x[1,0]
		vec[1,0] = (self.m*(-self.x[3,0]**2*self.l + self.g*math.cos(self.x[2,0]))*math.sin(self.x[2,0]))/(
			self.M + self.m*math.sin(self.x[2,0])**2)
		vec[2,0] = self.x[3,0]
		vec[3,0] = (self.g*(self.m + self.M)*math.sin(self.x[2,0]) - 
			self.x[3,0]**2*self.l*self.m*math.sin(self.x[2,0])*math.cos(self.x[2,0]))/(self.l*(self.M + self.m*math.sin(self.x[2,0])**2))
		return vec

	def g_dyn(self):
		vec = np.zeros((4,1))
		vec[0,0] = 0
		vec[1,0] = 1/(self.M + self.m*math.sin(self.x[2,0])**2)
		vec[2,0] = 0
		vec[3,0] = math.cos(self.x[2,0])/(self.l*(self.M + self.m*math.sin(self.x[2,0])**2))
		return vec

	def dynamics(self, ctrl_input, disturbance_bound = None):
		if disturbance_bound is None:
			xdot = self.f() + self.g_dyn() @ ctrl_input
		else:
			base_vec = np.vstack((np.zeros((2,1)),np.random.normal(size = (2,1))))
			disturbance = base_vec/np.linalg.norm(base_vec)*disturbance_bound
			xdot = self.f() + self.g_dyn() @ ctrl_input + disturbance
		return xdot
	
	def get_linear_A(self):
		self.A = np.array([
			[0, 1, 0, 0],
			[0,0,self.m*self.g/self.M,0],
			[0,0,0,1],
			[0,0,(self.M+self.m)*self.g/(self.M*self.l),0]
			])
		return self.A

	def get_linear_B(self):
		self.B = np.array([[
			0, 1/self.M, 0, 1/(self.l*self.M)]]).transpose()
		return self.B

	def linearized_dynamics(self, ctrl_input):
		xdot = self.A @ self.x + self.B @ ctrl_input
		return xdot

	def simulate(self, steps = 50, spacing = 100, end_condition = None, disturbance_bound = None):
		self.interior_dt = self.dt/spacing
		if end_condition is None:
			for tsteps in range(steps):
				ctrl_input = self.controller()
				self.uhist.append(ctrl_input[0,0])
				for splices in range(spacing):
					self.x = self.x + self.dynamics(ctrl_input = ctrl_input, disturbance_bound = disturbance_bound)*self.interior_dt
				self.xhist = np.hstack((self.xhist, self.x))
		else:
			cstep = 1
			while end_condition(self.x) == False and cstep <= steps:
				ctrl_input = self.controller()
				self.uhist.append(ctrl_input[0,0])
				for splices in range(spacing):
					self.x = self.x + self.dynamics(ctrl_input = ctrl_input, disturbance_bound = disturbance_bound)*self.interior_dt
				self.xhist = np.hstack((self.xhist, self.x))
				cstep += 1
		pass


class discrete_grid:
	def __init__(self, graph_size = [5,5], init_state = [2,2]):
		self.state = init_state
		self.xhist = [init_state]
		self.uhist = []
		self.graph_size = graph_size

	def controller(self):
		pass

	def update(self):
		x = self.state[0]
		y = self.state[1]
		c_action = self.controller()
		if c_action == 'up':
			y = min((y+1, self.graph_size[1]-1))
		elif c_action == 'down':
			y = max((y-1, 0))
		elif c_action == 'left':
			x = max((x-1, 0))
		elif c_action == 'right':
			x = min((x+1, self.graph_size[0]-1))
		self.state = [x,y]
		self.xhist.append(self.state)
		pass

	def simulate(self, steps = 50):
		for i in range(steps): self.update()
		pass 



if __name__ == '__main__':
	f = lambda x: np.identity(2) @ x
	g = lambda x: np.array([[0,1]]).transpose()
	cbf = lambda x: 5
	dhdx = lambda x: np.zeros((2,1))
	alpha = lambda x: 2*x
	print(QP_CBF(state = np.zeros((2,1)), udes = 0, f = f, g = g, h = cbf, dhdx = dhdx, alpha = alpha))



