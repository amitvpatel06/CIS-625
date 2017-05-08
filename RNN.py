import tensorflow as tf 
from gridworld import *
import numpy as np
import csv
import random



class C_single: 

	def __init__(self, params, memory=True):
		self.name = 'C_single'
		self.params = params
		self.layers = 2
		self.build_model_graph()

	def build_model_graph(self): 
		with tf.variable_scope(self.name) as self.scope:
			m = self.params['m']
			n = self.params['n']
			o = self.params['o']
			h = self.params['h_c']
			h_m = self.params['hidden']

			self.input_place_holder = tf.placeholder(tf.float32, shape=[1, m + n], name='input')
			self.target_values = tf.placeholder(tf.float32, shape=[1,o])
			self.W_1 = tf.get_variable('W_1', shape=[m + n, h])
			self.b_1 = tf.get_variable('b_1', shape=[h])
			self.hidden = tf.tanh(tf.matmul(self.input_place_holder, self.W_1) + self.b_1)
			self.W_2 = tf.get_variable('W_2', shape=[h, o])
			self.b_2 = tf.get_variable('b_2', shape=[o])
			self.predictions = tf.cast((tf.matmul(self.hidden, self.W_2) + self.b_2), 'float32') 
			self.max_score = tf.reduce_max(self.predictions, reduction_indices=[1])
			self.max_action = tf.argmax(tf.squeeze(self.predictions), axis=0, name="arg_min")
			self.batch_losses = tf.reduce_sum(tf.squared_difference(self.predictions, self.target_values), axis=1)
			self.loss = tf.reduce_sum(self.batch_losses, axis=0)
			self.trainer = tf.train.AdamOptimizer(learning_rate=0.001)
			self.gvs, self.variables = zip(*self.trainer.compute_gradients(self.loss))
			self.clipped_gradients, _ = tf.clip_by_global_norm(self.gvs, 5.0)
			self.updateWeights = self.trainer.apply_gradients(zip(self.clipped_gradients, self.variables))



class C_NN: 

	def __init__(self, params, memory=True):
		self.name = 'C'
		self.params = params
		self.layers = 2
		self.build_model_graph()

	def build_model_graph(self): 
		with tf.variable_scope(self.name) as self.scope:
			m = self.params['m']
			n = self.params['n']
			o = self.params['o']
			h = self.params['h_c']
			h_m = self.params['hidden']

			self.input_place_holder = tf.placeholder(tf.float32, shape=[1, 2*m + 2*n + h_m], name='input')
			self.target_values = tf.placeholder(tf.float32, shape=[1,o])
			self.W_1 = tf.get_variable('W_1', shape=[2*m + 2*n + h_m, h])
			self.b_1 = tf.get_variable('b_1', shape=[h])
			self.hidden = tf.tanh(tf.matmul(self.input_place_holder, self.W_1) + self.b_1)
			self.W_2 = tf.get_variable('W_2', shape=[h, o])
			self.b_2 = tf.get_variable('b_2', shape=[o])
			self.predictions = tf.cast((tf.matmul(self.hidden, self.W_2) + self.b_2), 'float32') 
			self.max_score = tf.reduce_max(self.predictions, reduction_indices=[1])
			self.max_action = tf.argmax(tf.squeeze(self.predictions), axis=0, name="arg_min")
			self.batch_losses = tf.reduce_sum(tf.squared_difference(self.predictions, self.target_values), axis=1)
			self.loss = tf.reduce_sum(self.batch_losses, axis=0)
			self.trainer = tf.train.AdamOptimizer(learning_rate=0.001)
			self.gvs, self.variables = zip(*self.trainer.compute_gradients(self.loss))
			self.clipped_gradients, _ = tf.clip_by_global_norm(self.gvs, 5.0)
			self.updateWeights = self.trainer.apply_gradients(zip(self.clipped_gradients, self.variables))

class M_RNN: 
	def __init__(self, params):
		self.name = 'M'
		self.params = params
		self.build_model_graph()
		self.buff = []

		m = self.params['m']
		n = self.params['n']
		o = self.params['o']
		for i in range(self.params['window']): 
			self.buff.append((np.zeros(m), np.zeros(n), np.zeros(1)))

	def add_to_history(self, state, reward, action):
		self.buff.append((state, reward, action))
		if len(self.buff) > self.params['window'] + 1:
			self.buff.pop(0)

	def compile_output_and_input(self):
		inp = []
		target = []
		m = self.params['m']
		n = self.params['n']
		o = self.params['o']
		for i in range(self.params['window']):
			sarsa = self.buff[i]
			next_sarsa = self.buff[i+1]
			inp.append(np.concatenate((sarsa[0], sarsa[1], sarsa[2]), axis=0).reshape(1, m + n + 1))
			target.append(np.concatenate((next_sarsa[0], next_sarsa[1]), axis=0).reshape(1, m + n))
		x = np.concatenate(inp, axis=0).reshape(1, self.params['window'], m + n + 1)
		y = np.concatenate(target, axis=0).reshape(self.params['window'], m + n )
		return (x, y)

	def build_model_graph(self): 
		with tf.variable_scope(self.name) as self.scope:
			m = self.params['m']
			n = self.params['n']
			o = self.params['o']
			h = self.params['hidden']
			window = self.params['window']
			self.input_place_holder = tf.placeholder(tf.float32, shape=(1, window, m + n + 1), name='input')
			self.target_sequence_placeholder = tf.placeholder(tf.float32, shape=(window, m + n), name='tgt')
			self.forward_cell_layers = tf.contrib.rnn.BasicRNNCell(h)
			self.rnn_output, self.final_rnn_state = tf.nn.dynamic_rnn(self.forward_cell_layers, self.input_place_holder, \
								sequence_length=[window]*1, dtype=tf.float32)
			self.outs = tf.squeeze(self.rnn_output)
			self.U = tf.get_variable('U', shape=[h, m + n])
			self.b_2 = tf.get_variable('b2', shape=[m+n])
			self.predictions = tf.cast((tf.matmul(self.outs, self.U) + self.b_2), 'float32')
			self.loss = tf.reduce_mean(0.01* tf.squared_difference(self.predictions, self.target_sequence_placeholder))
			self.trainer = tf.train.AdamOptimizer(learning_rate=0.001)
			self.gvs, self.variables = zip(*self.trainer.compute_gradients(self.loss))
			self.clipped_gradients, _ = tf.clip_by_global_norm(self.gvs, 5.0)
			self.updateWeights = self.trainer.apply_gradients(zip(self.clipped_gradients, self.variables))



def C_sim(params, environment, steps, filename):
	gamma = .99
	e = 1
	C = C_single(params)
	init = tf.global_variables_initializer()
	stats = []
	actions = []
	rewards = []
	losses = []
	mloss = 0
	with tf.Session() as sess:
		# init everything
		sess.run(init)
		curr_state = environment.state()
		curr_reward = environment.reward()
		curr_action = np.array([random.randint(0, params['o'])])	
		rewards = []
		m_losses = []
		losses = []
		for ts in range(1, steps + 1):
			print ts
			e = 100 / np.sqrt(ts)
			# given last state, action, reward, predict next action
			C_in = np.concatenate([curr_state, curr_reward]).reshape(1, params['m']+ params['n'])
			predicted_action, Q_est = sess.run((C.max_action, C.predictions), feed_dict={C.input_place_holder: C_in})
			predicted_action = np.array([predicted_action])
			targ_Q = Q_est
			#print 'Q'
			#print Q_est
			# randomly sample another part of the policy
			if np.random.rand(1) < e:
				predicted_action = np.array([random.randint(0, params['o'] - 1)])	
			#print 'action'
			#print predicted_action
			# move and update C, add to M's history
			environment.make_move(predicted_action[0])
			# get next state, reward, action 
			curr_state = environment.state()
			curr_reward = environment.reward()
			curr_action = predicted_action
			actions.append(curr_action[0])
			rewards.append(curr_reward[0])
			# predict arg max value of next state
			C_in_new = np.concatenate([curr_state, curr_reward]).reshape(1, params['m']+ params['n'])
			max_score = sess.run((C.max_score), feed_dict={C.input_place_holder: C_in_new})
			# update C's weights based on that
			backup = curr_reward + gamma * max_score
			targ_Q[0, curr_action] = backup
			updateC, loss = sess.run((C.updateWeights, C.loss), feed_dict={
																	C.input_place_holder: C_in,
																	C.target_values: targ_Q
																}) 
			losses.append(loss)
			#print 'target'
			#print targ_Q
			#print 'loss'
			#print loss
			if environment.move_c % environment.reset == 0:
				r = np.mean(rewards)
				l = np.mean(losses)
				stats.append([ts, r, l, mloss])
				losses = []
				rewards = []

	average_file = open(filename, 'wb')
	# write trades executed
	w = csv.writer(average_file)
	stats.insert(0, ['Timestep', 'Average Reward', 'C Loss', 'M Loss'])
	w.writerows(stats)
	return (actions, stats)


def CM_sim(params, environment, steps, filename):

	gamma = .99
	e = 1
	sleep_freq = 15

	M = M_RNN(params)
	C = C_NN(params)
	init = tf.global_variables_initializer()
	stats = []
	actions = []
	rewards = []
	losses = []
	mloss = 0
	with tf.Session() as sess:
		# init everything
		sess.run(init)
		curr_state = environment.state()
		curr_reward = environment.reward()
		curr_action = np.array([random.randint(0, params['o'])])	
		M.add_to_history(curr_state, curr_reward, curr_action)
		for ts in range(1, steps + 1):
			print ts
			e = 100 / np.sqrt(ts)
			# given last state, action, reward, predict next action
			M_in, _ = M.compile_output_and_input()
			M_hidden, M_predictions = sess.run((M.final_rnn_state, M.predictions), feed_dict={M.input_place_holder: M_in}) 
			M_pred = M_predictions[-1, :]
			C_in = np.concatenate([curr_state, curr_reward, M_hidden[0,:], M_pred]).reshape(1, 2*params['m']+ 2*params['n'] + params['hidden'])
			predicted_action, Q_est = sess.run((C.max_action, C.predictions), feed_dict={C.input_place_holder: C_in})
			predicted_action = np.array([predicted_action])
			targ_Q = Q_est
			#print 'Q'
			#print Q_est
			# randomly sample another part of the policy
			if np.random.rand(1) < e:
				predicted_action = np.array([random.randint(0, params['o'] - 1)])	
			#print 'action'
			#print predicted_action
			# move and update C, add to M's history
			environment.make_move(predicted_action[0])
			# get next state, reward, action 
			curr_state = environment.state()
			curr_reward = environment.reward()
			curr_action = predicted_action
			actions.append(curr_action[0])
			rewards.append(curr_reward[0])
			# predict arg max value of next state
			M.add_to_history(curr_state, curr_reward, curr_action)
			M_in_new, _ = M.compile_output_and_input()
			M_hidden_new, M_predictions_new = sess.run((M.final_rnn_state, M.predictions), feed_dict={M.input_place_holder: M_in_new}) 
			M_pred_new = M_predictions_new[-1, :]
			C_in_new = np.concatenate([curr_state, curr_reward, M_hidden_new[0,:], M_pred_new]).reshape(1, 2*params['m']+ 2*params['n'] + params['hidden'])
			max_score = sess.run((C.max_score), feed_dict={C.input_place_holder: C_in_new})
			# update C's weights based on that
			backup = curr_reward + gamma * max_score
			targ_Q[0, curr_action] = backup
			updateC, loss = sess.run((C.updateWeights, C.loss), feed_dict={
																	C.input_place_holder: C_in,
																	C.target_values: targ_Q
																}) 
			losses.append(loss)
			#print 'target'
			#print targ_Q
			#print 'loss'
			#print loss
			if ts % sleep_freq == 0: 
				x, y = M.compile_output_and_input()
				updateM, mloss = sess.run((M.updateWeights, M.loss), feed_dict={
																		M.input_place_holder: x,
																		M.target_sequence_placeholder: y
																	})
				print 'MLOSS'
			if environment.move_c % environment.reset == 0:
				r = np.mean(rewards)
				l = np.mean(losses)
				stats.append([ts, r, l, mloss])
				losses = []
				rewards = []

	average_file = open(filename, 'wb')
	# write trades executed
	w = csv.writer(average_file)
	stats.insert(0, ['Timestep', 'Average Reward', 'C Loss', 'M Loss'])
	w.writerows(stats)
	return (actions, stats)
	
if  __name__ == "__main__":
	params = {
		'm': 9,
		'n': 1,
		'o': 7,
		'h_c': 4,
		'window': 10,
		'hidden': 5
	}
	for i in range(10):
		size = np.power(2, i)
		control = size / 100
		env = Grid(size, control, 100)
		filename_CM = 'CM-episodes_{}'.format(size)
		filename_C = 'C-episodes_{}'.format(size)
		C_sim(params, env, 100000, filename_C)
		CM_sim(params, env, 100000, filename_CM)
		tf.reset_default_graph()

	

