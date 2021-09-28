import pandas as pd
import math
import numpy as np
import torch
import torch.nn as nn
import scipy
import sys
import time
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
np.random.seed(seed=1024)

class DPMM(nn.Module):
    def __init__(self, variationalT, alphaDP, data, device):
        super(DPMM, self).__init__()
        self.variationalT = variationalT
        self.clusterNum = variationalT
        self.alphaDP = alphaDP
        self.device = device

        # 数据的继承和修改
        self.data = torch.tensor(data=data, dtype=torch.float32, device=device)
        self.dataNumber = data.shape[0]
        self.dataDimension = data.shape[1]

        # Beta分布先验超参
        self.priorBetaGamma1 = torch.ones(1, variationalT, dtype=torch.float32, device=device)
        self.priorBetaGamma2 = torch.ones(1, variationalT, dtype=torch.float32, device=device)

        # 变分分布中Beta分布的参数
        self.variationalBetaGamma1 = torch.ones([1, variationalT], dtype=torch.float32, device=device)
        self.variationalBetaGamma2 = torch.ones([1, variationalT], dtype=torch.float32, device=device)

        # 变分分布中的Categorical分布参数
        self.npPhi = np.random.multinomial(n=1, pvals=(1.0 / float(variationalT) * np.ones(variationalT)), size=self.dataNumber)
        self.variationalPhi = torch.tensor(data=self.npPhi, dtype=torch.float32, device=device)


        # T个高斯成分 * 数据维度
        # 均值的变分超参数
        self.variationalGaussianMeanMu = torch.ones([self.dataDimension, variationalT], dtype=torch.float32, device=device)
        self.variationalGaussianVarKappa = torch.ones([1, variationalT], dtype=torch.float32, device=device)
        # 方差的变分超参数
        self.variationalWishartPsi = torch.ones([self.dataDimension, self.dataDimension, variationalT], dtype=torch.float32, device=device)
        self.variationalWishartNu = torch.ones([1, variationalT], dtype=torch.float32, device=device)

        # 均值的先验超参数
        # self.priorGaussianMeanMu = torch.mean(input=self.data, dim=0).unsqueeze(1)
        self.priorGaussianMeanMu = torch.ones(size=(self.dataDimension, 1))
        self.priorGaussianVarKappa = 0.1 * torch.ones([1], dtype=torch.float32, device=device)
        # 方差的先验超参数
        self.priorWishartPsi = 1.0 * torch.eye(self.dataDimension, dtype=torch.float32, device=device)
        # self.priorWishartNu = float(self.dataDimension + 1) * torch.ones([1], dtype=torch.float32, device=device)
        self.priorWishartNu = 200.0 * torch.ones([1], dtype=torch.float32, device=device)

    # def vbMStep(self, BetaGamma1, BetaGamma2, Phi, GaussianMuMk, GaussianMuKappa, WishartPsi, WishartNu):
    def vbMStep(self, BetaGamma1, Phi):

        # Gamma1参数计算 维度: (1 * t)
        BetaGamma1 = torch.ones_like(input=BetaGamma1, device=self.device) + torch.sum(input=Phi, dim=0, keepdim=True)

        # 计算Phi及其掩码矩阵, 避免后续for循环
        tempGamma2Phi = Phi.unsqueeze(0).repeat(self.variationalT, 1, 1)

        # torch.tril(input=torch.ones([4, 4]), diagonal=0).unsqueeze(2).repeat(1, 1, 4).permute(0, 2, 1)
        # torch.flip(input=u, dims=[0, 2])
        # torch.flip(input=torch.tril(input=torch.ones([self.variationalT, self.variationalT]), diagonal=0).unsqueeze(2).repeat(1, 1, self.dataNumber).permute(0, 2, 1), dims=[0, 2])

        # tempGamma2Mask = torch.flip(input=torch.tril(input=torch.ones([self.variationalT, self.variationalT], device=self.device, dtype=torch.float32), diagonal=0).unsqueeze(2).repeat(1, 1, self.dataNumber).permute(0, 2, 1), dims=[0, 2])

        tempGamma2Mask = torch.flip(input=torch.tril(input=torch.ones([self.variationalT, self.variationalT], device=self.device), diagonal=-1).unsqueeze(2).repeat(1, 1, self.dataNumber).permute(0, 2, 1), dims=[0, 2])

        # Gamma2参数计算 维度: (1 * t)
        BetaGamma2 = (float(self.alphaDP) + torch.sum(input=(tempGamma2Phi * tempGamma2Mask), dim=[1, 2], keepdim=False).T).unsqueeze(0)

        # 指数分布族参数计算
        # 预先分配空内存
        tempExpoentialTaoT11 = torch.zeros([self.dataDimension, self.dataDimension, self.variationalT], device=self.device, dtype=torch.float32)



        # 第一个自然参数的计算
        for i in range(self.variationalT):
            # 一共有t个成分, 打循环:
            temp1ForExpoentialTaoT11 = -0.5 * (self.priorWishartPsi + self.priorGaussianVarKappa * (self.priorGaussianMeanMu) @ (self.priorGaussianMeanMu.T)).view(-1, 1)
            temp2ForExpoentialTaoT11 = -0.5 * ((torch.bmm(input=self.data.unsqueeze(2), mat2=self.data.unsqueeze(1)).view(self.dataNumber, self.dataDimension * self.dataDimension)) * (Phi[:, i].unsqueeze(1).repeat(1, self.dataDimension * self.dataDimension)))
            tempExpoentialTaoT11[:, :, i] = (temp1ForExpoentialTaoT11.squeeze(1) + torch.sum(temp2ForExpoentialTaoT11, dim=0, keepdim=False)).view(self.dataDimension, self.dataDimension)



        # 第二个自然参数的计算 维度: (d * t)
        tempExpoentialTaoT12 = self.priorGaussianVarKappa * (self.priorGaussianMeanMu.repeat(1, self.variationalT)) + self.data.T @ Phi

        # 第三个自然参数的计算 维度: (1 * t)
        tempExpoentialTaoT21 = -0.5 * torch.sum(Phi, dim=0, keepdim=True) - 0.5 * (self.priorWishartNu + self.dataDimension + 2)

        # 第四个自然参数的计算 维度: (1 * t)
        tempExpoentialTaoT22 = self.priorGaussianVarKappa.unsqueeze(1).repeat(1, self.variationalT) + torch.sum(Phi, dim=0, keepdim=True)


        # 后验分布参数
        # 变分高斯分布
        variationalGaussianVarKappa = tempExpoentialTaoT22
        variationalGaussianMeanMu = tempExpoentialTaoT12 / tempExpoentialTaoT22.repeat(self.dataDimension, 1)

        # print('打印设备')
        # print(self.priorGaussianVarKappa.device)

        # 变分威沙特分布
        variationalWishartNu = -2.0 * (tempExpoentialTaoT21) - 2.0 - float(self.dataDimension)


        variationalWishartPsi = -2.0 * tempExpoentialTaoT11 - variationalGaussianVarKappa.unsqueeze(0).repeat(self.dataDimension, self.dataDimension, 1) * torch.bmm(input=variationalGaussianMeanMu.unsqueeze(1).permute(2, 0, 1), mat2=variationalGaussianMeanMu.unsqueeze(1).permute(2, 1, 0)).permute(2, 1, 0)


        return BetaGamma1, BetaGamma2, variationalGaussianMeanMu, variationalGaussianVarKappa, variationalWishartNu, variationalWishartPsi

    def vbEStep(self, BetaGamma1, BetaGamma2, variationalWishartPsi, variationalWishartNu, variationalGaussianMeanMu, variationalGaussianVarKappa):

        # 计算第1项 维度: (n, t)
        tempPhiTerm1 = (torch.digamma(input=(BetaGamma1)) - torch.digamma(input=(BetaGamma1 + BetaGamma2))).repeat(self.dataNumber, 1)

        # 计算第2项 维度: (n, t)
        tempPhiTerm2 = (torch.digamma(input=(BetaGamma2)) - torch.digamma(input=(BetaGamma1 + BetaGamma2))).T

        tempPhiTerm2 = tempPhiTerm2.repeat(1, self.variationalT) * torch.triu(input=torch.ones([self.variationalT, self.variationalT], device=self.device), diagonal=1)

        tempPhiTerm2 = torch.sum(input=tempPhiTerm2, dim=0, keepdim=True).repeat(self.dataNumber, 1)

        # 计算第3项 维度: (t, d, d)
        tempEqEta1 = ((variationalWishartNu.T.unsqueeze(2).repeat(1, self.dataDimension, self.dataDimension)) * torch.inverse(input=variationalWishartPsi.permute(2, 0, 1))).view(self.variationalT, self.dataDimension * self.dataDimension)

        # 计算第3项 维度: (t, d, 1)
        tempEqEta2 = variationalWishartNu.T.unsqueeze(2).repeat(1, self.dataDimension, self.dataDimension) * torch.inverse(input=variationalWishartPsi.permute(2, 0, 1))
        tempEqEta2 = torch.bmm(input=tempEqEta2, mat2=variationalGaussianMeanMu.T.unsqueeze(2))

        # 计算第3项 维度: (1, t)
        tempEqEta3 = -1.0 * torch.digamma(input=0.5 * variationalWishartNu) - self.dataDimension * math.log(2.0) + torch.log(torch.det(variationalWishartPsi.permute(2, 0, 1)))

        #计算第4项

        tempEqEta42 = torch.bmm(input=variationalGaussianMeanMu.T.unsqueeze(1), mat2=torch.inverse(variationalWishartPsi.permute(2, 0, 1)))
        tempEqEta42 = torch.bmm(input=tempEqEta42, mat2=variationalGaussianMeanMu.T.unsqueeze(2)).squeeze(2)

        # 真正的第4项 维度: (1, t)
        tempEqEta4 = -0.5 * float(self.dataDimension) / (variationalGaussianVarKappa) -0.5 * variationalWishartNu * tempEqEta42.T


        tempPhiTerm311 = (tempEqEta1.view(self.variationalT, self.dataDimension * self.dataDimension)).unsqueeze(0).repeat(self.dataNumber, 1, 1)
        tempPhiTerm312 = (-0.5 * torch.bmm(input=self.data.unsqueeze(2), mat2=self.data.unsqueeze(1))).view(self.dataNumber, self.dataDimension * self.dataDimension)
        # 第3项第1个 维度: (n, t)
        tempPhiTerm31 = torch.bmm(input=tempPhiTerm311, mat2=tempPhiTerm312.unsqueeze(2)).squeeze(2)

        # 第3项第2个 维度: (n, t)
        tempPhiTerm32 = tempEqEta2.squeeze(2) @ self.data.T
        # 第3项第3个 维度: (n, t)
        tempPhiTerm33 = -0.5 * tempEqEta3.repeat(self.dataNumber, 1).T

        # 第4项 维度: (n, t)
        tempPhiTerm4 = tempEqEta4.repeat(self.dataNumber, 1)



        finalPhi = tempPhiTerm1 + tempPhiTerm2 + tempPhiTerm31 + tempPhiTerm32.T + tempPhiTerm33.T + tempPhiTerm4


        finalPhiExp = torch.exp(input=finalPhi)

        finalPhiExp = finalPhiExp / torch.sum(input=finalPhiExp, dim=1, keepdim=True).repeat(1, finalPhiExp.shape[1])

        return finalPhiExp



        # 第四个自然参数的计算

    def truncation(self, threshold=0.1):
        # 进行截断操作
        piList = torch.sum(input=self.variationalPhi, dim=0, keepdim=True) / torch.sum(input=self.variationalPhi, dim=[0, 1])
        self.piList = piList
        piListTruncation = piList[piList >= threshold]
        self.clusterNum = piListTruncation.shape[0]

        # torch.sort(input, dim=-1, descending=False, stable=False, *, out=None)
        piList, piIdx = torch.sort(input=piList, dim=-1, descending=True)
        piIdx = piIdx.squeeze(0)

        # 进行重新排序
        self.variationalBetaGamma1 = self.variationalBetaGamma1[:, piIdx]
        self.variationalBetaGamma2 = self.variationalBetaGamma2[:, piIdx]

        # 高斯的变分超参数
        self.variationalGaussianMeanMu = self.variationalGaussianMeanMu[:, piIdx]
        self.variationalGaussianVarKappa = self.variationalGaussianVarKappa[:, piIdx]

        # 威沙特的变分超参数
        self.variationalWishartNu = self.variationalWishartNu[:, piIdx]
        self.variationalWishartPsi = self.variationalWishartPsi[:, :, piIdx]

        # 变分超参数Phi
        self.variationalPhi = self.variationalPhi[:, piIdx]

        # 存储变分超参数
        self.variationalBetaGamma1Old, self.variationalBetaGamma2Old = self.variationalBetaGamma1, self.variationalBetaGamma2
        self.variationalGaussianMeanMuOld, self.variationalGaussianVarKappaOld = self.variationalGaussianMeanMu, self.variationalGaussianVarKappa
        self.variationalWishartNuOld, self.variationalWishartPsiOld = self.variationalWishartNu, self.variationalWishartPsi
        self.variationalPhiOld = self.variationalPhi

        # 截断超参数列表
        piIdxList = torch.tensor(np.arange(self.clusterNum), dtype=torch.long)

        # 进行截断操作

        # Beta的变分超参数
        self.variationalBetaGamma1 = self.variationalBetaGamma1[:, piIdxList]
        self.variationalBetaGamma2 = self.variationalBetaGamma2[:, piIdxList]

        # 高斯的变分超参数
        self.variationalGaussianMeanMu = self.variationalGaussianMeanMu[:, piIdxList]
        self.variationalGaussianVarKappa = self.variationalGaussianVarKappa[:, piIdxList]

        # 威沙特的变分超参数
        self.variationalWishartNu = self.variationalWishartNu[:, piIdxList]
        self.variationalWishartPsi = self.variationalWishartPsi[:, :, piIdxList]

        # 变分超参数Phi
        self.variationalPhi = self.variationalPhi[:, piIdxList]
        return self

    def classPrediction(self):
        predictionClass = torch.argmax(input=self.variationalPhi, dim=1)

        predictionClass = pd.DataFrame(data=predictionClass.cpu().numpy())
        return predictionClass

    def vbEMAlgo(self, epoch=50):
        # BetaGamma1, BetaGamma2, variationalGaussianMeanMu, variationalGaussianVarKappa, variationalWishartNu, variationalWishartPsi

        t = time.time()
        for i in range(epoch):
            i = i + 1
            # BetaGamma1, BetaGamma2, variationalGaussianMeanMu, variationalGaussianVarKappa, variationalWishartNu, variationalWishartPsi
            self.variationalBetaGamma1, self.variationalBetaGamma2, \
            self.variationalGaussianMeanMu, self.variationalGaussianVarKappa, \
            self.variationalWishartNu, self.variationalWishartPsi = self.vbMStep(BetaGamma1=self.priorBetaGamma1,
                                                                                 Phi=self.variationalPhi)

            self.variationalPhi = self.vbEStep(BetaGamma1=self.variationalBetaGamma1,
                                               BetaGamma2=self.variationalBetaGamma2,
                                               variationalWishartPsi=self.variationalWishartPsi,
                                               variationalWishartNu=self.variationalWishartNu,
                                               variationalGaussianMeanMu=self.variationalGaussianMeanMu,
                                               variationalGaussianVarKappa=self.variationalGaussianVarKappa)

        self.runningTime = time.time() - t
        print('variational Bayesian running time:{} s'.format(str(self.runningTime)))

        return self

    def prediction(self, dataX, BetaGamma1, BetaGamma2, variationalWishartPsi, variationalWishartNu, variationalGaussianMeanMu, variationalGaussianVarKappa):

        '''

        # 变分分布中Beta分布的参数
        self.variationalBetaGamma1 = torch.ones([1, variationalT], dtype=torch.float32, device=device)
        self.variationalBetaGamma2 = torch.ones([1, variationalT], dtype=torch.float32, device=device)

        # 变分分布中的Categorical分布参数
        self.npPhi = np.random.multinomial(n=1, pvals=(1.0 / float(variationalT) * np.ones(variationalT)), size=self.dataNumber)
        self.variationalPhi = torch.tensor(data=self.npPhi, dtype=torch.float32, device=device)


        # T个高斯成分 * 数据维度
        # 均值的变分超参数
        self.variationalGaussianMeanMu = torch.ones([self.dataDimension, variationalT], dtype=torch.float32, device=device)
        self.variationalGaussianVarKappa = torch.ones([1, variationalT], dtype=torch.float32, device=device)
        # 方差的变分超参数
        self.variationalWishartPsi = torch.ones([self.dataDimension, self.dataDimension, variationalT], dtype=torch.float32, device=device)
        self.variationalWishartNu = torch.ones([1, variationalT], dtype=torch.float32, device=device)
        '''


        dataX = torch.tensor(data=dataX, dtype=torch.float32)
        dataXNumber, dimX = dataX.shape[0], dataX.shape[1]
        variationalGaussianMeanMu = variationalGaussianMeanMu[0: dimX, :]
        variationalWishartPsi = variationalWishartPsi[0: dimX, 0: dimX, :]
        # 计算第1项 维度: (n, t)
        tempPhiTerm1 = (torch.digamma(input=(BetaGamma1)) - torch.digamma(input=(BetaGamma1 + BetaGamma2))).repeat(dataXNumber, 1)

        # 计算第2项 维度: (n, t)
        tempPhiTerm2 = (torch.digamma(input=(BetaGamma2)) - torch.digamma(input=(BetaGamma1 + BetaGamma2))).T

        tempPhiTerm2 = tempPhiTerm2.repeat(1, self.clusterNum) * torch.triu(input=torch.ones([self.clusterNum, self.clusterNum], device=self.device), diagonal=1)

        tempPhiTerm2 = torch.sum(input=tempPhiTerm2, dim=0, keepdim=True).repeat(dataXNumber, 1)

        # 计算第3项 维度: (t, d, d)
        tempEqEta1 = ((variationalWishartNu.T.unsqueeze(2).repeat(1, dimX, dimX)) * torch.inverse(input=variationalWishartPsi.permute(2, 0, 1))).view(self.clusterNum, dimX * dimX)

        # 计算第3项 维度: (t, d, 1)
        tempEqEta2 = variationalWishartNu.T.unsqueeze(2).repeat(1, dimX, dimX) * torch.inverse(input=variationalWishartPsi.permute(2, 0, 1))
        tempEqEta2 = torch.bmm(input=tempEqEta2, mat2=variationalGaussianMeanMu.T.unsqueeze(2))

        # 计算第3项 维度: (1, t)
        tempEqEta3 = -1.0 * torch.digamma(input=0.5 * variationalWishartNu) - dimX * math.log(2.0) + torch.log(torch.det(variationalWishartPsi.permute(2, 0, 1)))

        # 计算第4项

        tempEqEta42 = torch.bmm(input=variationalGaussianMeanMu.T.unsqueeze(1),
                                mat2=torch.inverse(variationalWishartPsi.permute(2, 0, 1)))
        tempEqEta42 = torch.bmm(input=tempEqEta42, mat2=variationalGaussianMeanMu.T.unsqueeze(2)).squeeze(2)

        # 真正的第4项 维度: (1, t)
        tempEqEta4 = -0.5 * float(dimX) / (
            variationalGaussianVarKappa) - 0.5 * variationalWishartNu * tempEqEta42.T

        tempPhiTerm311 = (tempEqEta1.view(self.clusterNum, dimX * dimX)).unsqueeze(0).repeat(dataXNumber, 1, 1)
        tempPhiTerm312 = (-0.5 * torch.bmm(input=dataX.unsqueeze(2), mat2=dataX.unsqueeze(1))).view(dataXNumber, dimX * dimX)
        # 第3项第1个 维度: (n, t)
        tempPhiTerm31 = torch.bmm(input=tempPhiTerm311, mat2=tempPhiTerm312.unsqueeze(2)).squeeze(2)

        # 第3项第2个 维度: (n, t)
        tempPhiTerm32 = tempEqEta2.squeeze(2) @ dataX.T
        # 第3项第3个 维度: (n, t)
        tempPhiTerm33 = -0.5 * tempEqEta3.repeat(dataXNumber, 1).T

        # 第4项 维度: (n, t)
        tempPhiTerm4 = tempEqEta4.repeat(dataXNumber, 1)

        finalPhi = tempPhiTerm1 + tempPhiTerm2 + tempPhiTerm31 + tempPhiTerm32.T + tempPhiTerm33.T + tempPhiTerm4

        finalPhiExp = torch.exp(input=finalPhi)

        finalPhiExp = finalPhiExp / torch.sum(input=finalPhiExp, dim=1, keepdim=True).repeat(1, finalPhiExp.shape[1])

        muY = self.variationalGaussianMeanMu[dimX:, :]
        muX = self.variationalGaussianMeanMu[0: dimX, :]
        sigmaRecovered = DPMMTest1.variationalWishartPsi / ((DPMMTest1.variationalWishartNu - DPMMTest1.dataDimension - 1.0).unsqueeze(0).repeat(DPMMTest1.dataDimension, DPMMTest1.dataDimension, 1) )
        sigmaX = sigmaRecovered[0:dimX, 0:dimX, :]
        sigmaY = sigmaRecovered[dimX:, dimX:, :]

        sigmaYX = sigmaRecovered[0:dimX, dimX:, :]

        xForPred = dataX.unsqueeze(0).repeat(self.clusterNum, 1, 1)
        muXForPred = muX.T.unsqueeze(1).repeat(1, dataXNumber, 1)
        standardXForPred = (xForPred - muXForPred).permute(0, 2, 1)
        tempPredTerm21 = torch.bmm(input=(torch.inverse(input=sigmaX.permute(2, 0, 1))), mat2=standardXForPred)
        tempPredTerm2 = torch.bmm(input=sigmaYX.permute(2, 1, 0), mat2=tempPredTerm21).squeeze(1)



        tempYPred = muY.view(-1, self.clusterNum).repeat(dataXNumber, 1) + tempPredTerm2.T

        yPred = torch.sum(input=tempYPred * finalPhiExp, dim=1, keepdim=True)

        return yPred



#
# data = pd.read_csv('iris.csv', sep=',', header=None, index_col=None)
# data = data.iloc[:, 1:3]
# print(data)

# 开始干活儿
data = pd.read_csv('testSampleForRegression.csv', sep=',', header=0, index_col=0)
# print(data)
# GPU不支持批次矩阵求逆操作
DEVICE = torch.device('cpu')

DPMMTest1 = DPMM(variationalT=10, alphaDP=1.0/1000.0, data=np.array(data), device=DEVICE).to(DEVICE).vbEMAlgo(epoch=200)



DPMMTest1.truncation(threshold=0.10)

print('Before Truncation:')
print(DPMMTest1.variationalT)
print('After Truncation:')
print(DPMMTest1.clusterNum)
print('Phi value list:')
print(DPMMTest1.piList)

dataTest = pd.read_csv('testData.csv', sep=',', header=0, index_col=0)

yPred = DPMMTest1.prediction(dataX=np.array(dataTest.iloc[:, 0:2]),
                             BetaGamma1=DPMMTest1.variationalBetaGamma1,
                             BetaGamma2=DPMMTest1.variationalBetaGamma2,
                             variationalWishartPsi=DPMMTest1.variationalWishartPsi,
                             variationalWishartNu=DPMMTest1.variationalWishartNu,
                             variationalGaussianMeanMu=DPMMTest1.variationalGaussianMeanMu,
                             variationalGaussianVarKappa=DPMMTest1.variationalGaussianVarKappa)



dataY = np.array(dataTest.iloc[:, -1]).reshape((-1, 1))


r2 = r2_score(dataY, yPred)
rmse = math.sqrt(mean_squared_error(dataY, yPred))
print('Valuation Indices, r2: {}, rmse: {}'.format(r2, rmse))




f, ax = plt.subplots(figsize=(6, 6))
ax.scatter(dataY, yPred, c='#2728d6')
ax.plot([-20, 20], [-20, 20], ls="--", c='black')
ax.set(xlim=(-20, 20), ylim=(-20, 20))
ax.set_title('Prediction Scatter Plot, R2: ' + str(round(r2, 5)))
plt.show()


