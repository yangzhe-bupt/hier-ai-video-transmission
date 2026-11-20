# ======================================================================
# 文件：ANN_泰坦尼克.py
# 位置：practice/basics
# 功能：神经网络入门练习：使用泰坦尼克号表格数据进行生存分类。
# 类与函数汇总：
#   - create_dataset：读取CSV，补全缺失值、编码和标准化特征，划分训练测试数据。
#   - Mymodel：用于表格分类的全连接网络。
#     方法：__init__、forward
#   - train：训练分类模型并保存权重。
#   - evaluate：加载分类模型并统计测试准确率。
# 运行：python D:/learn/pytorch/practice/run.py basics/ANN_泰坦尼克.py
# 路径兼容：通过run.py运行；原导入、函数体与路径表达式保持不变。
# ======================================================================

from sklearn.preprocessing import StandardScaler        #数据标准化
import torch
from torch.utils.data import TensorDataset              #数据集对象
from torch.utils.data import DataLoader                 #数据加载器
import torch.nn as nn                                   #神经网络
import torch.optim as optim                             #数据优化器
from sklearn.model_selection import train_test_split    #划分训练集和测试集
import matplotlib.pyplot as plt                         #绘图
import numpy as np                                      #矩阵计算
import pandas as pd                                     #数据处理
import time                                             #时间模块
from torchinfo import summary                        #模型结构可视化

#1.定义函数，构建数据集
def create_dataset():
    data = pd.read_csv('D:/learn/pytorch/data/泰坦尼克号/train.csv')
    #print(f'data:{data.shape}')
    
    #提取特征，标签
    y = data['Survived']
    x = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']].copy()

    #补全数据
    x['Age'] = x['Age'].fillna(x['Age'].median())        # 年龄用中位数填
    x['Fare'] = x['Fare'].fillna(x['Fare'].median())     # 票价用中位数填
    x['Embarked'] = x['Embarked'].fillna('S')            # 登船港用众数 S 填

    #数据编码
    x['Sex'] = x["Sex"].map({'male':0,'female':1})
    x = pd.get_dummies(x, columns=['Embarked'], drop_first=True)

    #特征转浮点
    x = x.astype(np.float32)

    #切分数据集，训练集
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=3,stratify=y)

    #数据标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    #张量封装
    train_dataset = TensorDataset(torch.tensor(x_train),torch.tensor(y_train.values))
    test_dataset = TensorDataset(torch.tensor(x_test),torch.tensor(y_test.values))

    #返回结果
    return train_dataset,test_dataset,x_train.shape[1],len(np.unique(y))

#2.搭建神经网络
class Mymodel(nn.Module):
    def __init__(self,input_dim,output_dim):
        super().__init__()
        #搭建开始
        #隐藏层 32 - 16
        self.linear1 = nn.Linear(input_dim,32)
        self.linear2 = nn.Linear(32,16)
        #nn.Dropout(0.2)
        #self.linear3 = nn.Linear(64,32)
        #self.linear4 = nn.Linear(32,16)
        self.output =  nn.Linear(16,output_dim)

    def forward(self,x):
        #隐藏层加权求和+激活函数
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        #x = torch.relu(self.linear3(x))
        #x = torch.relu(self.linear4(x))
        x = self.output(x)

        return x

#3.模型训练
def train(train_dataset,input_dim,output_dim):
    #创建数据集加载对象并打乱
    train_loader = DataLoader(train_dataset,batch_size=16,shuffle=True)

    #创建模型
    model = Mymodel(input_dim,output_dim)

    #损失函数
    criterison = nn.CrossEntropyLoss()

    #创建数据优化器
    optimizer = optim.Adam(model.parameters(),0.001,(0.9,0.999))

    #开始训练
    epochs = 200
    for epoch in range(epochs):
        model = model.train()
        start = time.time()
        for x,y in train_loader:
            #预测值
            y_pred = model(x)
            #损失值
            loss = criterison(y_pred,y)
            #梯度清理，反向传播，更新权重
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        #打印每轮用时
        print(f'epoch={epoch+1},time:{time.time()-start}s')

    #保存模型
    torch.save(model.state_dict(),'D:/learn/pytorch/model/Titanic.pth')

#4.模型评估
def evaluate(test_dataset,input_dim,output_dim):
    model = Mymodel(input_dim,output_dim)
    #加载参数
    model.load_state_dict(torch.load('D:/learn/pytorch/model/Titanic.pth'))
    test_loader = DataLoader(test_dataset,batch_size=16,shuffle=False)
    correct = 0
    model.eval()
    for x,y in test_loader:
        
        y_pred = model(x)
        #print(f'y_pred={y_pred}')
        y_pred = torch.argmax(y_pred,dim=1)
        #print(f'y_pred={y_pred}')
        correct+=((y_pred==y).sum())

    #打印准确率
    print(f'准确率：{correct/len(test_dataset):.4f}')

#测试
if __name__=='__main__':
    #创建数据集
    train_dataset,test_dataset,input_dim,output_dim = create_dataset()

    #搭建模型
    model = Mymodel(input_dim,output_dim)
    #summary(model,input_size=(16,input_dim))

    #训练模型
    #train(train_dataset,input_dim,output_dim)

    #模型测试
    evaluate(test_dataset,input_dim,output_dim)