%%
clc;
clear;
close all;
% 卡住种子
rng(1024);
% 加载数据集


load('U06_data_Scale.mat')
TotalData = U06_data((53000 + 1): (53000 + 95000), :);
% X的维度和Y的维度来定位
XNo = 10;
YNo = 1;


Data_X = TotalData(:, 1:XNo);
Data_Y = TotalData(:, (XNo + 1): (XNo + YNo));
%% 训练数据


%% 组合数据集
F = TotalData;
dim_X = XNo;
dim_Y = YNo;


%% 超参数
Factor_Num = 4;
nMaxIter = 200;
minibatch =  100;


%% 给先验
% 这里X的组成是P * N(P为X的自变量维度，N为数据条数)
X = F';
piror.K = Factor_Num;
piror.m_s = zeros(piror.K, minibatch);
piror.sigma_s = 0.0 * ones(piror.K, piror.K);
piror.m_a = zeros(piror.K, size(X, 1));
piror.sigma_a = ones(piror.K, piror.K, size(X, 1));

piror.alpha_a = 1.0e-5*ones(piror.K, 1);
piror.alpha_b = 1.0e-5*ones(piror.K, 1);

piror.phi_a = 1.0e-6*ones(size(X, 1), 1);
piror.phi_b = 1.0e-6*ones(size(X, 1), 1);
%[q, piror, ELBO_list] = VBSFA(X, piror, nMaxIter);
%% 开始搞活
% 程序开始时间
tic
% 软测量在线
[output_list, ~, ELBO_list_summary ] = StreamingSoftSensorVBSFA(F, piror,dim_X, dim_Y, minibatch, nMaxIter);
% 结果评估
[evaluate_results] = evaluate_function(output_list);
% 结果画图
Results_Paint(output_list, minibatch, evaluate_results)
% 程序结束时间
toc
%% 流式变分贝叶斯软测量
function [output_list, regressor, ELBO_list_summary ] = StreamingSoftSensorVBSFA(F, piror,dim_X, dim_Y, minibatch, nMaxIter)
[N, P] = size(F);
LoopNo = ceil(N/minibatch);

F = reshape(F', [P, minibatch, LoopNo]);
F = permute(F, [2, 1, 3]);

X = F(:, 1 : dim_X, :);
Y = F(:, (dim_X + 1):(dim_X + dim_Y), :);

ELBO_list_summary = zeros(nMaxIter, LoopNo);

output_list = zeros((N - minibatch), 2*dim_Y);
% temp_output = zeros(minibatch, 2*dim_Y);
for iter = 1 : LoopNo
    temp_F = F(:, :, iter);
    if iter == 1
        % 第一次更新参数
        last_data = temp_F;
        [last_data, mean_last, variance_last] = auto(last_data);
        % 获得参数
        mean_X_last = mean_last(:, 1:dim_X);
        variance_X_last = variance_last(:, 1:dim_X);
        mean_Y_last = mean_last(:, (dim_X + 1):(dim_X + dim_Y));
        variance_Y_last = variance_last(:, (dim_X + 1):(dim_X + dim_Y));
        % 变分推断(一般的变分贝叶斯)
        [q_last, piror, ELBO_list] = VBSFA(last_data', piror, nMaxIter);
        ELBO_list_summary(:, iter) = ELBO_list(2:end,:);
    else
        % 以后的参数更新
        last_data = temp_F;
        [last_data, mean_last, variance_last] = auto(last_data);
        % 获得参数
        mean_X_last = mean_last(:, 1:dim_X);
        variance_X_last = variance_last(:, 1:dim_X);
        mean_Y_last = mean_last(:, (dim_X + 1):(dim_X + dim_Y));
        variance_Y_last = variance_last(:, (dim_X + 1):(dim_X + dim_Y));
        % 变分推断(这里就是流式变分贝叶斯了)
        [q_new, ELBO_list] = StreamingVBSFA(last_data', q_last, piror, nMaxIter);
        ELBO_list_summary(:, iter) = ELBO_list(2:end,:);
        
    end
    if iter > 1
        this_X = X(:, :, iter);
        % 软测量模型的构建
        [Y_pred, regressor] = SoftSensorVBSFA(this_X, q_last, dim_X, dim_Y, mean_X_last, variance_X_last, mean_Y_last, variance_Y_last);
        this_Y = Y(:, :, iter);
        
        output_list((iter-2)*minibatch + 1 : (iter-1)*minibatch, 1:dim_Y) = this_Y;
        output_list((iter-2)*minibatch + 1 : (iter-1)*minibatch, (dim_Y+1):2*dim_Y) = Y_pred;
        q_last = q_new;
    end
      
end

end


%% 标准化数据
% 标准化函数区
function [Xsc, mx, sx] = auto(Xtrain)
% 这里X的组成是P * N(P为X的自变量维度，N为数据条数)
% Xtrain = Xtrain';
mx = zeros(1, size(Xtrain,2));
sx = zeros(1, size(Xtrain,2));
Xsc = zeros(size(Xtrain,1), size(Xtrain,2));
for i = 1 : size(Xtrain, 2)
    mx(1, i) = mean(Xtrain(:, i));
    sx(1, i) = std(Xtrain(:, i));
end
 % sx = ones(1, size(Xtrain,2));
for i = 1 : size(Xtrain, 2)
    Xsc(:, i) = (Xtrain(:, i) - mx(1, i))/ sx(1, i);
end
end

% 标准化函数区
function X_sc = auto_test(X_test, mean_X, variance_X)
% 这里X的组成是P * N(P为X的自变量维度，N为数据条数)
X_sc = zeros(size(X_test,1), size(X_test,2));
for i = 1 : size(X_test, 2)
    X_sc(:, i) = (X_test(:, i) - mean_X(1, i))/ variance_X(1, i);
end
end

function Y_pred = inv_test(Y_test, mean_Y, variance_Y)
% 这里X的组成是P * N(P为X的自变量维度，N为数据条数)
Y_pred = zeros(size(Y_test,1), size(Y_test,2));
for i = 1 : size(Y_test, 2)
    Y_pred(:, i) = Y_test(:, i) * variance_Y(1, i) + mean_Y(1, i);
end
end
%% 软测量模型构建(一般的SFA软测量)
function [Y_pred, regressor] = SoftSensorVBSFA(X_Test, q, dim_X, dim_Y, mean_X, variance_X, mean_Y, variance_Y)
% 数据标准化
X_Test = auto_test(X_Test, mean_X, variance_X);
% [X_Test, ~, ~] = auto(X_Test);
X_Test = X_Test';
% 构造回归器
regressor.sigma_s = q.sigma_s;
regressor.phi_exp = q.phi_a ./ q.phi_b;
regressor.phi_X = diag(regressor.phi_exp(1: dim_X, :));
regressor.phi_Y = diag(regressor.phi_exp((dim_X + 1): (dim_X + dim_Y), :));
% 回归器要的AX
regressor.AX = q.m_a(:, 1 : dim_X);
% 计算因子s
s_new = regressor.sigma_s * regressor.AX * regressor.phi_X * X_Test;
regressor.m_s = s_new;
% 回归要的AY
regressor.AY = q.m_a(:, (dim_X + 1) : (dim_X + dim_Y));


% 回归器的预测y
% y_new = regressor.AY' * s_new  + diag(regressor.phi_Y);
y_new = regressor.AY' * s_new ;

% 重新搞回来
y_new = y_new';
Y_pred = inv_test(y_new, mean_Y, variance_Y);
end
%% 流式变分
function [q_new, ELBO_list] = StreamingVBSFA(X, q_last, piror, nMaxIter)
K = q_last.K; [P, N] = size(X);
q_new = q_last;

q_new.sigma_s = piror.sigma_s;
q_new.m_s = piror.m_s;
q_new.sigma_a = piror.sigma_a;

ELBO_list = zeros(nMaxIter+1, 1);
ELBO_list(1, 1) = 1.0e3;

for iter = 1: nMaxIter  
    % 推断q(s)
    % 推断q.sigma_s
    phi_exp = q_new.phi_a ./ q_new.phi_b ;
    sigma_s_temp = zeros(K,K);
    for p = 1: P
        sigma_s_temp = sigma_s_temp ...
            + phi_exp(p, 1) * q_new.m_a(:, p) * q_new.m_a(:, p)' + q_new.sigma_a(:, :, p);
             % + phi_exp(p, 1) * q.m_a(p, :)' * q.m_a(p, :) + q.sigma_a(:, :, p);      
    end
    q_new.sigma_s = inv(eye(K) + sigma_s_temp);
    % 推断q.m_s
     q_new.m_s = q_new.sigma_s * q_new.m_a * diag(phi_exp) * X;

    % 推断q(A)
    % 推断q.sigma_A
    phi_exp = q_new.phi_a ./ q_new.phi_b ;
    % 计算<si*siT>q(s)
    temp_phi = zeros(K, K);
    % msmst = sum( pagemtimes(reshape(q.ms, [K, 1, N]), reshape(q.ms, [1, K, N])), 1);
    temp_phi = temp_phi + N * q_new.sigma_s + sum( pagemtimes(reshape(q_new.m_s, [K, 1, N]), reshape(q_new.m_s, [1, K, N])), 3);
 
    for p = 1: P
        % 推断q.sigma_a
        q_new.sigma_a(:, :, p) = inv(phi_exp(p, 1) * temp_phi + inv(q_last.sigma_a(:, :, p)));
        % 推断q.ma
        q_new.m_a(:, p) = q_new.sigma_a(:, :, p) * (phi_exp(p, 1) * sum( (bsxfun(@times, X(p, :) , q_new.m_s)), 2)...
            + ( q_last.sigma_a(:, :, p) \  q_last.m_a(:, p)) ) ;   
    end
% + inv(q_last.sigma_a(:, :, p)) *  q_last.m_a(:, p) );    
%     % 推断q(alpha)
%     q.alpha_a = piror.alpha_a + P/2;  
%     temp_alpha_b = sum(q.sigma_a, 3) + sum( pagemtimes(reshape(q.m_a, [K, 1, P]), reshape(q.m_a, [1, K, P])), 3);
%     q.alpha_b = piror.alpha_b + 0.5 * diag(temp_alpha_b);
    
    % 推断q(phi)
    
    q_new.phi_a = q_last.phi_a + N/2;
    a_aT = q_new.sigma_a + pagemtimes(reshape(q_new.m_a, [K, 1, P]), reshape(q_new.m_a, [1, K, P]));
    phi_sigma_s = reshape(repmat(q_new.sigma_s, 1, N), [K, K, N]);
    phi_mu_s = pagemtimes(reshape(q_new.m_s, [K, 1, N]), reshape(q_new.m_s, [1, K, N]));
    phi_sisiT = phi_sigma_s + phi_mu_s;
    for p = 1:P
        % 计算二次型
        temp_quadratic1 = sum((X(p, :) .* X(p, :)), 2);
        temp_binary = 2 * sum((bsxfun(@times, X(p, :), ( q_new.m_a(:, p)' * q_new.m_s) ) ) , 2);
        % 计算乘法项
        aj_ajT = reshape(repmat(a_aT(:, :, p), 1, N), [K, K, N]) ;
        temp_quadratic2 = pagemtimes(aj_ajT, phi_sisiT);
        temp_quadratic2 = sum(temp_quadratic2, 3);
        temp_quadratic2 = trace(temp_quadratic2);
        q_new.phi_b(p, 1) = q_last.phi_b(p, 1) + 0.5 * (temp_quadratic1 - temp_binary + temp_quadratic2);
    end
   
    ELBO_list(iter+1, 1) = StreamingFreeEnergy(q_new, q_last, X);
    % if ((ELBO_list(iter+1, 1) - ELBO_list(iter, 1)) /ELBO_list(iter, 1) * 100 < 1.0e-9) && (iter ~= 1)
%     if ((ELBO_list(iter+1, 1) - ELBO_list(iter, 1)) < 1.0e-4) && (iter ~= 1)
%         % ELBO_list = ELBO_list(2:end, :);
%         break; 
%         
%     end  
 
    % ELBO_list = ELBO_list(2:end, :);
end

end
%% Streaming的ELBO的计算
function ELBO = StreamingFreeEnergy(q_new, q_last, X)
K = q_last.K; [P, N] = size(X);
% 计算条件概率的期望
temp_logPhi_exp = psi(q_new.phi_a) - log(q_new.phi_b);
temp_Phi_exp = q_new.phi_a ./ q_new.phi_b ;
J = -0.5 * N * P * log(2 * pi) + 0.5 * N * sum(temp_logPhi_exp, 1) + ... 
    + sum( (temp_Phi_exp .* q_last.phi_b - q_new.phi_a), 1);

% 计算KL散度
% KL(q(s)||p(s))
temp_KL_s = sum( pagemtimes(reshape(q_new.m_s, [K, 1, N]), reshape(q_new.m_s, [1, K, N])), 3);
KL_s = 0.5 * N + 0.5 * N * trace(eye(K) - q_new.sigma_s) - 0.5 * trace(temp_KL_s);

% KL(q(A) || p(A|alpha))
% alpha_exp = q_new.alpha_a ./ q_new.alpha_b;


temp_KL_A = 0;
for p = 1:P
    temp_KL_A = temp_KL_A + 0.5 * log(det(q_last.sigma_a(:, :, p)) / det(q_new.sigma_a(:, :, p))) ...
        - 0.5 * K + 0.5 *  trace(q_last.sigma_a(:, :, p) \ q_new.sigma_a(:, :, p)) ...
        + 0.5 * (q_new.m_a(:, p) - q_last.m_a(:, p))'* (q_last.sigma_a(:, :, p) \ (q_new.m_a(:, p) - q_last.m_a(:, p))) ;
end
% - 0.5 * K + 0.5 *  trace(inv(q_last.sigma_a(:, :, p)) * q_new.sigma_a(:, :, p)) 
% + 0.5 * (q_new.m_a(:, p) - q_last.m_a(:, p))'* inv(q_last.sigma_a(:, :, p)) * (q_new.m_a(:, p) - q_last.m_a(:, p))
KL_A = temp_KL_A;


% % KL(q(alpha) || p(alpha))
% temp_KL_alpha = q_new.alpha_a .* log(q_new.alpha_b) - q_last.alpha_a .* log(q_last.alpha_b) ...  
%    - (gammaln(q_new.alpha_a) - gammaln(q_last.alpha_a)) ... 
%     + q_last.alpha_b .* (q_new.alpha_a ./ q_new.alpha_b) - q_new.alpha_a ... 
%     + (q_new.alpha_a - q_last.alpha_a) .* (psi(q_new.alpha_a) - log(q_new.alpha_b));
% KL_alpha = sum(temp_KL_alpha, 1);

% KL(q(phi) || p(phi))
temp_KL_phi = q_new.phi_a .* log(q_new.phi_b) - q_last.phi_a .* log(q_last.phi_b)  ...  
     - (gammaln(q_new.phi_a) - gammaln(q_last.phi_a)) ...    
     + q_last.phi_b .* (q_new.phi_a ./ q_new.phi_b) - q_new.phi_a... 
     + (q_new.phi_a - q_last.phi_a) .* (psi(q_new.phi_a) - log(q_new.phi_b));
KL_phi = sum(temp_KL_phi, 1);

ELBO = J - KL_s - KL_A  - KL_phi;
end

%% 调用函数区
function [q, piror, ELBO_list] = VBSFA(X, piror, nMaxIter)
K = piror.K; [P, N] = size(X);
q.K = piror.K;
q.sigma_s = piror.sigma_s;
q.m_s = piror.m_s;

q.m_a = 1 * ones(piror.K, size(X, 1));
% piror.m_a = zeros(piror.K, size(X, 1));
q.sigma_a = piror.sigma_a;

q.phi_a = piror.phi_a;
q.phi_b = piror.phi_b;

q.alpha_a = piror.alpha_a;
q.alpha_b = piror.alpha_b;
% nMaxIter = 30;

ELBO_list = zeros(nMaxIter+1, 1);
ELBO_list(1, 1) = 1.0e3;

for iter = 1: nMaxIter  
    % 推断q(s)
    % 推断q.sigma_s
    phi_exp = q.phi_a ./ q.phi_b ;
    sigma_s_temp = zeros(K,K);
    for p = 1: P
        sigma_s_temp = sigma_s_temp ...
            + phi_exp(p, 1) * q.m_a(:, p) * q.m_a(:, p)' + q.sigma_a(:, :, p);
             % + phi_exp(p, 1) * q.m_a(p, :)' * q.m_a(p, :) + q.sigma_a(:, :, p);      
    end
    q.sigma_s = inv(eye(K) + sigma_s_temp);
    % 推断q.m_s
     q.m_s = q.sigma_s * q.m_a * diag(phi_exp) * X;
    %q.m_s = q.sigma_s * q.m_a * diag(phi_exp) * X;
    % 推断q(A)
    % 推断q.sigma_A
    phi_exp = q.phi_a ./ q.phi_b ;
    % 计算<si*siT>q(s)
    temp_phi = zeros(K, K);
    % msmst = sum( pagemtimes(reshape(q.ms, [K, 1, N]), reshape(q.ms, [1, K, N])), 1);
    temp_phi = temp_phi + N * q.sigma_s + sum( pagemtimes(reshape(q.m_s, [K, 1, N]), reshape(q.m_s, [1, K, N])), 3);
    alpha_exp = q.alpha_a ./ q.alpha_b ;
    for p = 1: P
        % 推断q.sigma_a
        q.sigma_a(:, :, p) = inv(phi_exp(p, 1) * temp_phi + diag(alpha_exp));
        % 推断q.ma
        q.m_a(:, p) = q.sigma_a(:, :, p) * phi_exp(p, 1) * sum( (bsxfun(@times, X(p, :) , q.m_s)), 2);
    end
    
    % 推断q(alpha)
    q.alpha_a = piror.alpha_a + P/2;  
    temp_alpha_b = sum(q.sigma_a, 3) + sum( pagemtimes(reshape(q.m_a, [K, 1, P]), reshape(q.m_a, [1, K, P])), 3);
    q.alpha_b = piror.alpha_b + 0.5 * diag(temp_alpha_b);
    
    % 推断q(phi)
    q.phi_a = piror.phi_a + N/2;
    a_aT = q.sigma_a + pagemtimes(reshape(q.m_a, [K, 1, P]), reshape(q.m_a, [1, K, P]));
    phi_sigma_s = reshape(repmat(q.sigma_s, 1, N), [K, K, N]);
    phi_mu_s = pagemtimes(reshape(q.m_s, [K, 1, N]), reshape(q.m_s, [1, K, N]));
    phi_sisiT = phi_sigma_s + phi_mu_s;
    for p = 1:P
        % 计算二次型
        temp_quadratic1 = sum((X(p, :) .* X(p, :)), 2);
        temp_binary = 2 * sum((bsxfun(@times, X(p, :), ( q.m_a(:, p)' * q.m_s) ) ) , 2);
        % 计算乘法项
        aj_ajT = reshape(repmat(a_aT(:, :, p), 1, N), [K, K, N]) ;
        temp_quadratic2 = pagemtimes(aj_ajT, phi_sisiT);
        temp_quadratic2 = sum(temp_quadratic2, 3);
        temp_quadratic2 = trace(temp_quadratic2);
        q.phi_b(p, 1) = piror.phi_b(p, 1) + 0.5 * (temp_quadratic1 - temp_binary + temp_quadratic2);
    end
   
    ELBO_list(iter+1, 1) = FreeEnergy(q, piror, X);
    % if ((ELBO_list(iter+1, 1) - ELBO_list(iter, 1)) /ELBO_list(iter, 1) * 100 < 1.0e-9) && (iter ~= 1)
%     if ((ELBO_list(iter+1, 1) - ELBO_list(iter, 1)) < 1.0e-4) && (iter ~= 1)
%         % ELBO_list = ELBO_list(2:end, :);
%         break; 
%         
%     end  
 
    % ELBO_list = ELBO_list(2:end, :);
end

end

%% ELBO的计算
function ELBO = FreeEnergy(q, piror, X)
K = piror.K; [P, N] = size(X);
% 计算条件概率的期望
temp_logPhi_exp = psi(q.phi_a) - log(q.phi_b);
temp_Phi_exp = q.phi_a ./ q.phi_b ;
J = -0.5 * N * P * log(2 * pi) + 0.5 * N * sum(temp_logPhi_exp, 1) + ... 
    + sum( (temp_Phi_exp .* piror.phi_b - q.phi_a), 1);

% 计算KL散度
% KL(q(s)||p(s))
temp_KL_s = sum( pagemtimes(reshape(q.m_s, [K, 1, N]), reshape(q.m_s, [1, K, N])), 3);
KL_s = 0.5 * N + 0.5 * N * trace(eye(K) - q.sigma_s) - 0.5 * trace(temp_KL_s);

% KL(q(A) || p(A|alpha))
alpha_exp = q.alpha_a ./ q.alpha_b;
temp_KL_A_I = reshape(repmat(eye(K), 1, P), [K, K, P]);
temp_KL_A_A = q.sigma_a + pagemtimes(reshape(q.m_a, [K, 1, P]), reshape(q.m_a, [1, K, P]));
temp_KL_A_alpha = reshape(repmat(diag(alpha_exp), 1, P), [K, K, P]);
temp_KL_A_inner = pagemtimes(temp_KL_A_A, temp_KL_A_alpha);
temp_KL_A_trace = trace(sum((temp_KL_A_I - temp_KL_A_inner), 3));

temp_KL_A_sigma = 0;
for p = 1: P
    temp_KL_A_sigma = temp_KL_A_sigma + log(det(q.sigma_a(:, :, p)));
end
KL_A = 0.5 * P * (sum((psi(q.alpha_a) - log(q.alpha_b)), 1)) ...
    + 0.5 * (temp_KL_A_sigma + temp_KL_A_trace);

% KL(q(alpha) || p(alpha))
temp_KL_alpha = q.alpha_a .* log(q.alpha_b) - piror.alpha_a .* log(piror.alpha_b) ...  
   - (gammaln(q.alpha_a) - gammaln(piror.alpha_a)) ... 
    + piror.alpha_b .* (q.alpha_a ./ q.alpha_b) - q.alpha_a ... 
    + (q.alpha_a - piror.alpha_a) .* (psi(q.alpha_a) - log(q.alpha_b));
KL_alpha = sum(temp_KL_alpha, 1);

% KL(q(phi) || p(phi))
temp_KL_phi = q.phi_a .* log(q.phi_b) - piror.phi_a .* log(piror.phi_b)  ...  
     - (gammaln(q.phi_a) - gammaln(piror.phi_a)) ...    
     + piror.phi_b .* (q.phi_a ./ q.phi_b) - q.phi_a... 
     + (q.phi_a - piror.phi_a) .* (psi(q.phi_a) - log(q.phi_b));
KL_phi = sum(temp_KL_phi, 1);

ELBO = J - KL_s - KL_A - KL_alpha - KL_phi;
end
%% 评价函数
function [evaluate] = evaluate_function(output_list)
temp_rmse = (output_list(:, 1) - output_list(:, 2)) .* (output_list(:, 1) - output_list(:, 2));
rmse = sqrt(1/size(output_list, 1) * sum(temp_rmse , 1));
evaluate.rmse = rmse;


SSE = (output_list(:, 1)-output_list(:, 2)).^2;
SST = (output_list(:, 1)-mean(output_list(:, 2))).^2;
r2 = 1 - ( sum(SSE)/sum(SST) );
evaluate.r2 = r2;
end
%% 结果画图
function Results_Paint(output_list, minibatch, evaluate)
output_length = linspace(1, size(output_list, 1), size(output_list, 1));
figure;
plot(output_length , output_list(:, 1), 'b', 'LineWidth',2.5);
hold on
plot(output_length , output_list(:, 2), 'r', 'LineWidth',2.5);
hold on
legend({'real value','predicted by S-VBSFA'},'Location','northwest');
name = 'S_VBSFA';
RMSE = evaluate.rmse;
R2 = evaluate.r2;
name = [name, '_minibatch=', num2str(minibatch), '_R2=', num2str(R2),'_RMSE=', num2str(RMSE),'.fig'];
saveas(gcf, name) ;
end