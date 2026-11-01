# ======================================================================
# 文件：CNN_编解码器.py
# 位置：practice/basics
# 功能：早期CNN图像重建练习：使用CIFAR10比较有无特征元素随机置零训练的模型。GRACE仅为练习标签，不代表官方复现；此处尚非真实包级丢失。
# 类与函数汇总：
#   - create_dataset：读取或下载CIFAR10训练集和测试集。
#   - GraceModel：CNN图像自编码器，编码特征可按元素随机置零。
#     方法：__init__、forward
#   - train：训练普通模型或带随机特征丢失训练的模型并保存权重。
#   - evaluate：评测多个特征元素丢失概率下的重建MSE。
#   - visualize：显示原图与两组模型的重建对照。
#   - compare_and_plot：统计两组模型在不同丢失概率下的MSE并绘制曲线。
# 运行：python D:/learn/pytorch/practice/run.py basics/CNN_编解码器.py
# 路径兼容：通过run.py运行；原导入、函数体与路径表达式保持不变。
# ======================================================================

import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt
import numpy as np
import random

#1.准备数据集
def create_dataset():
    #1.获取训练集       数据集地址，是否是训练集，数据预处理->张量数据，是否联网下载
    train_dataset = CIFAR10('D:/learn/pytorch/data',train=True,transform=ToTensor(),download=True)
    #2.获取测试集
    test_dataset = CIFAR10('D:/learn/pytorch/data',train=False,transform=ToTensor(),download=True)
    #3.返回数据集
    return train_dataset,test_dataset

#2.搭建神经网络CNN
class GraceModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        #编码器
        self.conv1 = nn.Conv2d(3,6,3,1,1)
        self.pool1 = nn.MaxPool2d(2,2,0)
        self.conv2 = nn.Conv2d(6,16,3,1,1)
        self.pool2 = nn.MaxPool2d(2,2,0)

        #解码器
        self.deconv1 = nn.ConvTranspose2d(16, 6, 2, stride=2)
        self.deconv2 = nn.ConvTranspose2d(6, 3, 2, stride=2)


    def forward(self,x,mask_ratio):
        #编码
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))

        if mask_ratio > 0:
            mask = torch.bernoulli(torch.ones_like(x) * (1 - mask_ratio))
            x = x * mask

        #解码
        x = torch.relu(self.deconv1(x))
        x = torch.sigmoid(self.deconv2(x)) 

        return x

#3.模型训练
def train(train_dataset,is_grace):
    dataloader = DataLoader(train_dataset,batch_size=32,shuffle=True)

    model = GraceModel()

    criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(),0.001,(0.9,0.999))

    model.train()

    epochs = 100
    for epoch in range(epochs):
        total_loss,total_samples,total_time = 0.0,0,time.time()

        for x,y in dataloader:
            if is_grace:
                ratio = random.uniform(0.0, 0.7)
                x_pred = model(x,mask_ratio=ratio)
            else :
                x_pred = model(x,mask_ratio=0.0)

            loss = criterion(x_pred,x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss+=loss.item()*len(x)
            total_samples += len(x)

        print(f'epoch:{epoch+1},loss={total_loss/total_samples:.5f}, time={time.time()-total_time:.2f}s')

    if is_grace:
        save_path = 'D:/learn/pytorch/model/Grace_model.pth'
    else :
        save_path = 'D:/learn/pytorch/model/Base_model.pth'
    torch.save(model.state_dict(),save_path)

#4.模型测试
def evaluate(test_dataset):
    dataloader = DataLoader(test_dataset,batch_size=32,shuffle=False)
    model = GraceModel()
    model.load_state_dict(torch.load('D:/learn/pytorch/model/Grace_model.pth'))
    model.eval()
    rates = [0.0,0.1,0.2,0.3,0.4,0.5]
    for rate in rates:
        total_mse , total_samples = 0.0,0
        for x,y in dataloader:
            x_pred = model(x,mask_ratio=rate)

            total_mse+=torch.mean((x_pred-x)**2).item() * len(x)
            total_samples+=len(x)
        print(f'丢包率：{rate*100:.1f}%,重建误差：{total_mse/total_samples:.4f}')

# 5. 结果可视化 (三方对比版：原图 vs 普通模型 vs GRACE模型)
def visualize(test_dataset):
    dataloader = DataLoader(test_dataset, batch_size=8, shuffle=True)
    
    # 加载普通模型 (Base)
    model_base = GraceModel()
    model_base.load_state_dict(torch.load('D:/learn/pytorch/model/Base_model.pth'))
    model_base.eval()

    # 加载 GRACE 模型
    model_grace = GraceModel()
    model_grace.load_state_dict(torch.load('D:/learn/pytorch/model/Grace_model.pth'))
    model_grace.eval()

    # 取出 8 张图片
    images, _ = next(iter(dataloader))
    
    # 模拟丢失 20% 的数据包
    with torch.no_grad():
        recon_base = model_base(images, mask_ratio=0.2)
        recon_grace = model_grace(images, mask_ratio=0.2)

    # 画图：3 排，8 列
    plt.figure(figsize=(12, 5.5))
    
    # 辅助函数：将 Tensor 转回 matplotlib 可显示的 Numpy 格式
    def to_img(tensor):
        return tensor.detach().cpu().numpy().transpose(1, 2, 0) 

    for i in range(8):
        # 第一排：原图
        plt.subplot(3, 8, i + 1)
        plt.imshow(to_img(images[i]))
        plt.axis('off')
        if i == 0:
            plt.title('Original', loc='left', fontsize=12)

        # 第二排：普通模型重建图
        plt.subplot(3, 8, i + 9)
        plt.imshow(to_img(recon_base[i]))
        plt.axis('off')
        if i == 0:
            plt.title('Base (20% Loss)', loc='left', fontsize=12)

        # 第三排：GRACE 重建图
        plt.subplot(3, 8, i + 17)
        plt.imshow(to_img(recon_grace[i]))
        plt.axis('off')
        if i == 0:
            plt.title('GRACE (20% Loss)', loc='left', fontsize=12)

    plt.tight_layout()
    plt.show()
    
# 6. 双模型对比折线图
def compare_and_plot(test_dataset):
    dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    mse_base, mse_grace = [], []

    # 加载 Base 模型
    model_base = GraceModel()
    model_base.load_state_dict(torch.load('D:/learn/pytorch/model/Base_model.pth'))
    model_base.eval()

    # 加载 GRACE 模型
    model_grace = GraceModel()
    model_grace.load_state_dict(torch.load('D:/learn/pytorch/model/Grace_model.pth'))
    model_grace.eval()

    print("\n--- Calculating Comparison Data ---")
    with torch.no_grad():
        for rate in rates:
            loss_b, loss_g, total = 0.0, 0.0, 0
            for x, _ in dataloader:
                # 普通模型重建
                pred_b = model_base(x, mask_ratio=rate)
                loss_b += torch.mean((pred_b - x) ** 2).item() * len(x)
                
                # GRACE 模型重建
                pred_g = model_grace(x, mask_ratio=rate)
                loss_g += torch.mean((pred_g - x) ** 2).item() * len(x)
                
                total += len(x)
            
            mse_base.append(loss_b / total)
            mse_grace.append(loss_g / total)
            print(f"Loss Rate {rate*100:>4.0f}% : Base = {mse_base[-1]:.4f} | GRACE = {mse_grace[-1]:.4f}")

    # 画图 (全英文，无乱码)
    plt.figure(figsize=(8, 5))
    plt.plot(rates, mse_base, marker='o', linestyle='--', color='gray', label='Base Model')
    plt.plot(rates, mse_grace, marker='s', linestyle='-', color='green', label='GRACE Model')
    
    plt.title('Reconstruction Error vs Packet Loss Rate', fontsize=14)
    plt.xlabel('Packet Loss Rate', fontsize=12)
    plt.ylabel('MSE (Lower is Better)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 反转 Y 轴，让表现更好的绿线在上方
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

if __name__=='__main__':

    train_dataset,test_dataset = create_dataset()

    train(train_dataset,is_grace=False)
    print('-'*30)
    train(train_dataset,is_grace=True)

    evaluate(test_dataset)

    visualize(test_dataset)

    compare_and_plot(test_dataset)