%% 设置种子
clc;
clear all;
close all;
% 全局种子
rng(1024)
%% 
% 模态百分比
percentage = [0.3, 0.4, 0.3];
% 总数据
TotalNumber = 1000;
mode_number = size(percentage, 2);
% 因子数
Factor_Num = 3;
% X的维度
dim_X = 5;
% Y的维度
dim_Y = 1;
TotalData_list = [];
% 每一个模态种子数
seed_Number = poissrnd(1024,1, mode_number);
% 每一个模态采样数
sample_number = TotalNumber * percentage;
for h = 1 : mode_number

[TotalData, A, C]  = data_generation(Factor_Num,dim_X, dim_Y, seed_Number(1, h), sample_number(1, h), h);
TotalData_list = [TotalData_list;TotalData];
end
save TotalData_list;
%% 搞事情函数
function [TotalData, A, C]  = data_generation(Factor_Num,dim_X, dim_Y, seed_Number, sample_number, mode_number)
%% 卡种子
rng(seed_Number);
%% 超参数
mu_t = poissrnd(mode_number, Factor_Num,1);
% mu_t = zeros(Factor_Num, 1);
sigma_t = eye(Factor_Num);
%% 生成样本
sample_t = mvnrnd(mu_t,sigma_t,sample_number);

% 生成x的噪音
mu_e =  zeros(dim_X, 1);
sigma_e = 0.1 * linspace(1, dim_X, dim_X);
sigma_e = sigma_e .* sigma_e;
sigma_e = diag(sigma_e);
sample_e = mvnrnd(mu_e,sigma_e,sample_number);

% A矩阵
A = randn(dim_X, Factor_Num);
A = rand(dim_X, Factor_Num);

% 生成X的数据
data_X = A * sample_t' + sample_e';

% 生成y的噪音
mu_f =  zeros(dim_Y, 1);
sigma_f = 0.1 * ones(dim_Y, 1);
sigma_f = sigma_f .* sigma_f;
sigma_f = diag(sigma_f);
sample_f = mvnrnd(mu_f,sigma_f,sample_number);

% C矩阵
C = rand(dim_Y, Factor_Num);

% 生成y的数据
data_Y = C * sample_t' + sample_f';

% 最终搞起
TotalData = [data_X; data_Y]';
end
%% 存储数据

% save TotalData;