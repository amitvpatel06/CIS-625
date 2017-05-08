import random
import numpy as np

class Grid(object):

	def __init__(self, size, movePrecision, reset):
		self.size = size
		self.movePrecision = movePrecision
		self.reset = reset
		self.move_c = 0
		self.position = np.random.rand(3) * self.size
		self.targetPosition = np.random.rand(3) * size
		self.velocity = np.zeros(3)
		self.acceleration = np.zeros(3)


	def make_move(self, move):
		self.acceleration = np.zeros(3)
		# RIGHT
		if move == 0:
			self.acceleration[0] = self.movePrecision
		# LEFT
		elif move == 1:
			self.acceleration[0] = -self.movePrecision
		# FORWARD
		elif move == 2:
			self.acceleration[1] = self.movePrecision
		# BACKWARD
		elif move == 3:
			self.acceleration[1] = -self.movePrecision
		# UP
		elif move == 4:
			self.acceleration[2] = self.movePrecision
		# DOWN
		elif move == 5:
			self.acceleration[2] = -self.movePrecision
		# NOT CONSTANT
		elif move != 6:
			print('illegal move')
		self.move_c += 1
		if self.move_c % self.reset == 0:
			self.position = np.random.rand(3) * self.size
			self.velocity = np.zeros(3)
			self.acceleration = np.zeros(3)
		# Perform the move
		self.move()
		print "State: " 		
		print self.state()
		print "Reward: " 
		print self.reward()
		return

	# State Function
	def state(self):
		# 9x1 Vector of position, velocity, acceleration
		return np.concatenate((self.position, self.velocity, self.acceleration), axis=0)


	# Reward Function
	def reward(self):
		return np.array([-0.01 * sum((self.position - self.targetPosition) ** 2)])


	def move(self):
		for i in xrange(3):
			self.velocity[i] += self.acceleration[i]
			tempPos = self.position[i] + self.velocity[i]
			if tempPos > self.size:
				tempPos = self.size
				self.velocity[i] = 0
			elif tempPos < 0:
				tempPos = 0
			self.position[i] = tempPos
		return



