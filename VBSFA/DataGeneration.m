%% 设置种子
clc;
clear all;
close all;
rng(1024);
%% 超参数
Factor_Num = 3;
dim_X = 5;
dim_Y = 1;
sample_number = 100;
mu_t = zeros(Factor_Num, 1);
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
%% 存储数据
save TotalData;