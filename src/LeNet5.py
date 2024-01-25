import torch
import torch.nn as nn
import torch.nn.functional as F


# 搭建神经网络
class LeNet5(nn.Module):
# 初始化
    def __init__(self, n_classes):  # 定义网络结构
        super(LeNet5, self).__init__()
        
        self.feature_extractor = nn.Sequential(   # 特征提取器，用于提取输入图像的特征
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),  # 卷积层（二维卷积层）
            nn.Tanh(),  # 双曲正切（Tanh）激活函数
            nn.AvgPool2d(kernel_size=2),  # 二维平均池化层
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            nn.Tanh()
        )

        self.classifier = nn.Sequential(  # 分类器，用于将特征映射到类别概率
            nn.Linear(in_features=120, out_features=84),  # 全连接层
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=n_classes),  # 输出样本的大小为类别数
        )

# 定义前向传播方式
    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)  # 将特征展平成一维向量
        logits = self.classifier(x)  # 全连接层（往往是模型的最后一层）的值，一般代码中叫做logits
        probs = F.softmax(logits, dim=1)  # 归一化的值，含义是属于该位置的概率，一般代码中叫做probs
        return logits                     # softmax是一个激活函数，作用是归一化
        # return logits, probs            # 经过使用指数形式的softmax函数能够将差距大的数值距离拉的更大
                                          # dim指的是归一化的方式，如果为0是对列做归一化，1是对行做归一化