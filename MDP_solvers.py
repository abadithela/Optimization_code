import numpy as np
import cvxpy as cp
import pdb


'''
Keep this script the same as the prior UCB one, initiate a class that contains all optimization code
and then run the solver
'''

class MDP_solver:
	def __init__(self, graph, reward_function, des_prob = None, discount = 0.9):
		'''
		This class assumes the following setup for any provided MDP to be solved.
			1) graph: a list of lists with the following structure
				a) graph[0] = the number of nodes in the graph
				b) graph[1] = a list of Ax(N+1) numpy array containing the A actions available at the given
				node and the transition probabilities to other nodes based on taking that action.  This
				includes the self transition available at every node (should such a self-transition
				exist).  The first element in each row should be an integer corresponding to the
				action for which the following transition probability is valid (i.e. if there
				existed two actions in an MDP, a or b, then assign a = 1, b = 2, and use those
				numbers in the beginning of the array)
			2) reward_function: the reward function for the MDP, which assigns to each
			state, action, state pair a real value (the reward for taking that action
			a given state, and ending up in a specific final state)
			3) des_prob: For the linear programming solution, a probability distribution over
			states is required (see Berkeley EECS notes on Linear Programming solution for MDPs
			via Value Iteration).  If none is provided a uniform distribution will be generated
			4) discount = the discount factor for the expectation calculation
		'''
		self.nodes = graph[0]
		self.num_actions = [None for i in range(self.nodes)]
		self.t_prob = graph[1]
		for idx,value in enumerate(graph[1]):
			self.num_actions[idx] = value.shape[0]
		self.reward = reward_function
		if des_prob is None:
			self.des_prob = 1/self.nodes*np.ones((1,self.nodes))
		self.discount = discount

	def setup(self):
		'''
		Produces the following lists:
			1) self.reward_vec: a list where each entry is an NxA array, with N = the number of
			nodes in the MDP, and A, the number of actions.  The entries of this array are the reward
			values for transitioning to this state with a specific action, from a specific start state
			2) self.next_states: a list where each entry is an NxA array, with N = the number of nodes
			in the MDP, and A, the number of actions.  The entries of this array are the states
			to which a feasible transition exists, from the given state (index in list) and given action
			(column in array).
		'''
		self.reward_vec = [None for i in range(self.nodes)]
		self.next_states = [None for i in range(self.nodes)]
		for cnode in range(self.nodes):
			self.reward_vec[cnode] = np.zeros((self.nodes,self.num_actions[cnode]))
			self.next_states[cnode] = np.zeros((self.nodes,self.num_actions[cnode]))
			for action in range(self.num_actions[cnode]):
				transitions = np.nonzero(self.t_prob[cnode][action,1:])[0]
				state_indeces = [i for i in transitions]
				self.next_states[cnode][state_indeces, action] = 1
				for state in state_indeces:
					self.reward_vec[cnode][state,action] = self.reward(cnode, self.t_prob[cnode][action,0], state)
		pass


	def optimize(self):
		'''
		Applies the Linear Programming solution to solve fro the optimal value function for the MDP
		'''
		self.setup()
		V = cp.Variable(self.nodes)
		cost = cp.Minimize(self.des_prob @ V)
		constr = []
		for node in range(self.nodes):
			node_picker = np.zeros((1,self.nodes))
			node_picker[0,node] = 1
			for action in range(self.num_actions[node]):
				constr += [node_picker @ V >= self.t_prob[node][action,1:] @ (
					self.reward_vec[node][:,action].transpose() + self.discount*V )]
		prob = cp.Problem(cost,constr)
		prob.solve()
		self.optimal_value = V.value
		pass

	def find_policy(self):
		'''
		Calculates the optimal value function for each node, then identifies the best action possible at each
		node
		'''
		self.optimize()
		self.policy = [None for i in range(self.nodes)]
		for node in range(self.nodes):
			action_rewards = [None for i in range(self.num_actions[node])]
			for action in range(self.num_actions[node]):
				action_rewards[action] = self.t_prob[node][action,1:] @ (self.reward_vec[node][:,action].transpose() + 
					self.discount*self.optimal_value)
			best_action_index = action_rewards.index(max(action_rewards))
			self.policy[node] = int(self.t_prob[node][best_action_index,0])

		pass

