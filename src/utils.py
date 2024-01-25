import numpy as np
import pandas as pd

import torch
from torchvision import datasets, transforms

from LeNet5 import LeNet5


def aux_info(dataset_name, model_name):
	if dataset_name in ["MNIST", "FMNIST"]:
		num_classes = 10

	# transform = None
	if model_name == "SVM":
		transform = transforms.Compose([
			transforms.ToTensor(),  # 将图像转换为张量
			transforms.Normalize((0.1307,), (0.3081,)),  # 对张量进行标准化，其中，0.1307和0.3081是mnist数据集的均值和标准差，因为mnist数据值都是灰度图，所以图像的通道数只有一个，因此均值和标准差各一个。
			transforms.Lambda(lambda x: torch.flatten(x))  # 将张量展平为一维向量
		])
	if model_name == "LeNet5":
		transform = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.1307,), (0.3081,)),
			transforms.Resize((32, 32))  # 将图像的大小调整为32×32。原来的图像大小是28x28。
		])

	return num_classes, transform


def dataset_info(dataset_name, transform):   # 下载数据集
	if dataset_name == "MNIST":
		trainset = datasets.MNIST('../data', train=True, download=True, transform=transform)
		testset = datasets.MNIST('../data', train=False, download=True, transform=transform)

	if dataset_name == "FMNIST":
		trainset = datasets.FashionMNIST('../data', train=True, download=True, transform=transform)
		testset = datasets.FashionMNIST('../data', train=False, download=True, transform=transform)

	input_dim = calculate_input_dim(trainset[0][0].shape)  # 张量的维度
	return list(trainset), list(testset), input_dim


def model_info(model_name, input_dim, num_classes):
	if model_name == "SVM":
		model = torch.nn.Linear(input_dim, num_classes)
		criterion = torch.nn.MultiMarginLoss()  # 多类别分类的hinge损失函数，用于单标签多分类任务

	if model_name == "LeNet5":
		model = LeNet5(num_classes)
		criterion = torch.nn.CrossEntropyLoss()  # 交叉熵损失函数，用于单标签多分类任务

	model_dim = calculate_model_dim(model.parameters())
	return model, criterion, model_dim


def calculate_input_dim(shape):
	dim = 1   # dim=1，指定列，也就是行不变，列之间的比较
	for ax in shape:
		dim *= ax
	return dim


def calculate_model_dim(model_params):
	model_dim = 0   # dim = 0，指定的是行，那就是列不变，理解成：同一列中每一行之间的比较或者操作，是每一行的比较，因为行是可变的。
	for param in model_params:
		model_dim += len(param.flatten())
	return model_dim


def calculate_learning_rate(learning_rate_type, iteration, epoch, batch_size):  # 学习率类型、迭代次数、轮数、批次大小
	if learning_rate_type == "constant":
		lr = 0.01
	if learning_rate_type == "iter_decay":
		lr = 1 / np.sqrt(1 + iteration)
	if learning_rate_type == "epoch_decay":
		lr = 0.01 / (1 + epoch)
	if learning_rate_type == "data_decay":
		lr = 1 / np.sqrt(1 + iteration*batch_size)

	return lr


def calculate_accuracy(model, testset):
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 判断电脑GPU可不可用，如果可用的话device就采用cuda()即调用GPU，不可用的话就采用cpu()即调用CPU

	model.eval()  # 如果模型中有 BN 层（Batch Normalization）和 Dropout，在 测试时 添加 model.eval()。model.eval() 是保证 BN 层能够用 全部训练数据 的均值和方差，即测试过程中要保证 BN 层的均值和方差不变。对于 Dropout，model.eval() 是利用到了 所有 网络连接，即不进行随机舍弃神经元。
	test_loader = torch.utils.data.DataLoader(  # 对数据进行装载
		testset,  # 决定数据从哪读取或者从何读取
		batch_size=32,  # 每次处理的数据集大小（默认为1）
		shuffle=True  # 每一个epoch是否为乱序（default: False）
	)

	correct = 0
	for dataX, dataY in iter(test_loader):
		dataX, dataY = dataX.to(device), dataY.to(device)  # 把变量放到对应的device上（如果用的是CPU的话就不用这一步了，因为变量默认是存在CPU上的，调用GPU的话要先把变量放到GPU上跑，跑完之后再调回CPU上）
		output = model(dataX) # 处理后的结果
		pred = output.argmax(dim=1)  # 取模型的预测输出output中概率最大的类别作为预测结果
		correct += (pred == dataY).int().sum().item()  # 累计正确的次数，将预测结果和真实标签逐一比较，得到一个布尔类型的张量，统计其中为True的元素个数，即预测正确的数量.加了item只输出tensor里面的result数值，不返回其他东西 若都返回造成运行负担

	return correct / len(testset)  # 正确个数/数据集总数量 = 正确率


def moving_average(x, y, window=32):  # 移动平均的目的是去除噪声
	if len(x) <= window:
		return x, y

	output_x = x[window-1:]

	val = 0
	for i in range(window):
		val += y[i] / window
	output_y = [val]
	for i in range(window, len(y)):
		val += (y[i] - y[i - window]) / window
		output_y.append(val)  # 在列表的末尾添加一个新元素val。这个方法会修改原来的列表，而不会返回一个新的列表。

	return output_x, output_y


def moving_average_df(dfs_dict, index="iters", window=32):
	dfs_dict_ret = {}

	for threshold_type, df in dfs_dict.items():
		dfs_dict_ret[threshold_type] = pd.DataFrame()  # pd.DataFrame是 Pandas 库中的一个类，用于创建和操作数据框（DataFrame）。DataFrame 是 Pandas 的核心数据结构，用于以表格形式和处理数据，类似提供电子表格或数据库表格。

		for column_name in df:
			if column_name == index:
				x, _ = moving_average(df[index], df[index], window)
				dfs_dict_ret[threshold_type][index] = x

			_, y = moving_average(df[index], df[column_name], window)
			dfs_dict_ret[threshold_type][column_name] = y

	return dfs_dict_ret   # 字典
