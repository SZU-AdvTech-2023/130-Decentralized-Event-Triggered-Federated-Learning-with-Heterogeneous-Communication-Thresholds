from copy import deepcopy
from random import random, choices

import torch
import torch.optim as optim

import utils


class Agent():
	"""docstring for Agent"""
	def __init__(self,
				ID,
				bandwidth,  # 带宽
				model,
				criterion,  # 基准
				trainset,
				testset,
				model_dim,
				batch_size						=	1,
				learning_rate_type				=	"iter_decay",
				r								=	1e1,
				threshold_type					=	"heterogeneous",  # 阈值类型：异构
				randomized_gossip_probability	=	None,
				rho_estimate					=	5000
	):
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		# Agent-based properties
		self.ID = ID
		self.bandwidth = bandwidth

		# Learning-based parameters
		self.initial_model = model
		self.w = deepcopy(self.initial_model).to(self.device)  # 深拷贝(deepcopy)： 是对于一个对象所有层次的拷贝(递归)，完全拷贝了父对象及其子对象。
		self.what = deepcopy(self.w).to(self.device)
		self.len_params = len(list(model.parameters()))

		self.criterion = criterion
		self.trainset = trainset
		self.testset = testset
		self.model_dim = model_dim
		self.batch_size = batch_size
		self.learning_rate_type = learning_rate_type

		self.loss = 0
		self.gradient_step = None  # 梯度步长
		self.accuracy = 0

		# Aggregation-based parameters  # 基于聚合的参数
		self.r = r
		self.rho = self.calculate_rho()
		self.threshold_type = threshold_type
		self.rho_estimate = rho_estimate

		self.neighbors = []
		self.aggregation_neighbors = []
		self.v = 0
		self.aggregation_step = None

		# In case agent runs randomized gossip algorithm
		self.randomized_gossip_probability = randomized_gossip_probability

		# Counters
		self.data_processed = 0
		self.aggregation_count = 0


	def calculate_rho(self):
		return 1 / self.bandwidth


	def run(self):
		# Event 4: new data
		self.gradient_step = [0 for _ in range(self.len_params)]

		# self.gradient_step = self.event_new_data(choices(self.trainset, k=self.batch_size))
		start = self.data_processed%len(self.trainset)
		self.gradient_step = self.event_new_data(self.trainset[start : start+self.batch_size])

		# Event 3: receive updates from neighbors
		self.aggregation_step = [0 for _ in range(self.len_params)]
		self.aggregation_neighbors = []

		for neighbor in self.neighbors:
			if neighbor.get_v() == 1 or self.v == 1:
				self.aggregation_neighbors.append(neighbor)
		if self.aggregation_neighbors:
			self.aggregation_step = self.event_update_received(self.aggregation_neighbors)


	def post_run_update(self):
		if self.v == 1:
			self.what = deepcopy(self.w).to(self.device)

		with torch.no_grad():  # PyTorch中的上下文管理器，用于指定一段代码块内部不需要计算梯度。通常在测试集验证或者模型推断时使用。
			param_idx = 0
			for param in self.w.parameters():
				param.data += self.aggregation_step[param_idx] - self.gradient_step[param_idx]
				param_idx += 1

		# Event 2: threshold passed
		self.v = 0

		if self.threshold_type == "randomized_gossip":
			if random() < self.randomized_gossip_probability:
				self.v = 1
		else:
			threshold_w = self.calculate_threshold()
			norm_e = 0
			p = 2
			for param, paramhat in zip(self.w.parameters(), self.what.parameters()):
				norm_e += torch.pow(torch.linalg.vector_norm(param.data - paramhat.data, ord=p), p)

			if torch.pow(norm_e / self.model_dim, 1/p) >= threshold_w:
				self.v = 1


	def event_new_data(self, data):
		self.data_processed += self.batch_size
		return self.gradient_descent(data)


	def event_update_received(self, neighbors):
		aggregation_step = [0 for _ in range(self.len_params)]

		for neighbor in neighbors:
			aggregation_weight = self.calculate_aggregation_weight(neighbor)

			param_idx = 0
			for param, param_neighbor in zip(self.w.parameters(), neighbor.get_w().parameters()):
				aggregation_step[param_idx] += aggregation_weight*(param_neighbor.data - param.data)
				param_idx += 1

		self.aggregation_count += len(neighbors)
		return aggregation_step


	def gradient_descent(self, data):  # 梯度下降
		# We do the the update on a temporary model, so that we can do the learning ...
		# and the aggregation at the same iteration.
		w2 = deepcopy(self.w).to(self.device)
		w2.train()

		train_loader = torch.utils.data.DataLoader(
			data,
			batch_size=self.batch_size,
			shuffle=True
		)
		dataX, dataY = next(iter(train_loader))
		dataX, dataY = dataX.to(self.device), dataY.to(self.device)
		learning_rate = self.calculate_learning_rate()

		optimizer = optim.SGD(w2.parameters(), lr=learning_rate)  # 初始化SGD优化器
		# 调用optimizer.zero_grad()重置模型参数的梯度。如果不设置默认情况下继续累加加起来； 为了防止重复计算，我们在每次迭代时明确地将它们归零。
		optimizer.zero_grad()  # 会遍历模型的所有参数，通过p.grad.detach_()方法截断反向传播的梯度流，再通过p.grad.zero_()函数将每个参数的梯度值设为0，即上一次的梯度记录被清空。因为训练的过程通常使用mini-batch方法，所以如果不将梯度清零的话，梯度会与上一个batch的数据相关，因此该函数要写在反向传播和梯度下降之前。
		output = w2(dataX)
		loss = self.criterion(output, dataY)
		# 通过调用 loss.backward()反向传播预测损失. PyTorch将误差梯度w.r.t.存储在每个参数中。
		loss.backward()  # 在使用后，会一层层的反向传播计算每个w的梯度值，并保存到该w的.grad属性中。如果没有进行 backward() 的话，梯度值将会是None，因此loss.backward()要写在optimizer.step()之前。
		# 一旦我们有了梯度，我们就调用 optimizer.step()通过反向传播中收集的梯度来调整参数。
		optimizer.step() # 作用是执行一次优化步骤，通过梯度下降法来更新参数的值。因为梯度下降是基于梯度的，所以在执行optimizer.step()函数前应先执行loss.backward()函数来计算梯度。

		gradient_step = [0 for _ in range(self.len_params)]

		param_idx = 0
		for param, param2 in zip(self.w.parameters(), w2.parameters()):
			gradient_step[param_idx] = param.data - param2.data
			param_idx += 1

		self.loss = loss
		return gradient_step


	def calculate_learning_rate(self):
		iteration = self.data_processed // self.batch_size
		epoch = self.data_processed // len(self.trainset)
		return utils.calculate_learning_rate(self.learning_rate_type, iteration, epoch, self.batch_size)


	def calculate_aggregation_weight(self, neighbor):
		return 1 / (1 + max(self.get_degree(), neighbor.get_degree()))


	def calculate_threshold(self):
		if self.threshold_type == "heterogeneous":
			coef = self.heterogeneous_coef()
		elif self.threshold_type == "constant":
			coef = self.constant_coef()
		elif self.threshold_type == "zero":
			coef = self.zero_coef()

		return coef * self.calculate_gamma()


	def calculate_gamma(self):  # 学习率调整倍数
		return self.calculate_learning_rate() ** 1

		
	def heterogeneous_coef(self):
		return self.r * self.rho
	def constant_coef(self):
		return self.r * self.rho * 0.4
		# return self.r * self.rho_estimate
	def zero_coef(self):
		return 0


	def calculate_accuracy(self):
		self.accuracy = utils.calculate_accuracy(self.w, self.testset)


	def reset(self, model, threshold_type, randomized_gossip_probability=None):
		# Agent-based properties
		if randomized_gossip_probability:
			self.randomized_gossip_probability = randomized_gossip_probability

		# Learning-based parameters
		# self.w = model.to(self.device)	# generate new random weights
		self.w = deepcopy(self.initial_model).to(self.device)	# reuse initial model every time
		self.what = deepcopy(self.w).to(self.device)
		self.loss = 0

		# Aggregation-based parameters
		self.v = 0
		self.threshold_type = threshold_type

		# Counters
		self.data_processed = 0
		self.aggregation_count = 0


	def cpu_used(self):
		return len(self.aggregation_neighbors)
	def max_cpu_usable(self):
		return self.get_degree()


	def bandwidth_used(self):
		return len(self.aggregation_neighbors)*self.model_dim
	def max_bandwidth_usable(self):
		return self.get_degree()*self.model_dim
			

	def transmission_time_used(self):
		return (self.bandwidth_used() / self.bandwidth) / self.get_degree()
		# return (self.bandwidth_used() / self.bandwidth) / len(self.aggregation_neighbors)
	def max_transmission_time_usable(self):
		return (self.max_bandwidth_usable() / self.bandwidth) / self.get_degree()


	def add_neighbor(self, neighbor):
		self.neighbors.append(neighbor)
	def remove_neighbor(self, neighbor):
		self.neighbors.remove(neighbor)
	def clear_neighbors(self):
		self.neighbors = []


	def set_trainset(self, trainset):
		self.trainset = trainset
	def set_bandwidth(self, bandwidth):
		self.bandwidth = bandwidth


	def get_degree(self):
		return len(self.neighbors)


	def get_w(self):
		return self.w
	def get_v(self):
		return self.v
	def get_loss(self):
		return self.loss
	def get_aggregation_count(self):
		return self.aggregation_count
	def get_accuracy(self):
		return self.accuracy