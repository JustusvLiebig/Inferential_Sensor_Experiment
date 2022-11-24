class SystemIdentification(nn.Module):
    def __init__(self, data_dim, hidden_dim, ):
        super(SystemIdentification, self).__init__()
        self.data_dim = data_dim
        self.hidden_dim = hidden_dim

        self.transition_matrix = nn.init.xavier_uniform_(torch.zeros([hidden_dim, hidden_dim]))
        temp_transition_noise = torch.ones([hidden_dim, 1])
        self.transition_noise = torch.eye(hidden_dim)

        self.observation_matrix = nn.init.xavier_uniform_(torch.zeros([data_dim, hidden_dim]))
        temp_observation_noise = torch.ones([data_dim, 1])
        self.observation_noise = torch.eye(data_dim)

        self.init_mean = torch.zeros([hidden_dim, 1])
        self.init_cov = torch.eye(hidden_dim)

        self.filtering_mean_list, self.filtering_cov_list = [], []
        self.smoothing_mean_list, self.smoothing_cov_list = [], []

        self.filtering_mean_list.append(self.init_mean)
        self.filtering_cov_list.append(self.init_cov)

    def subspace_initialize(self, data, hidden_dim, stacking_order=5):
        data_number, data_dim = data.shape[1], data.shape[0]
        hankel_matrix = hankel = scipy.linalg.hankel([x for x in range(stacking_order)],
                                                     [x for x in range(stacking_order - 1, data_dim, 1)])
        hankel_list = list(chain(*hankel_matrix.T.tolist()))
        expand_data = data[:, hankel_list]
        svd_data = expand_data.reshape((int(data_dim * stacking_order), -1))
        [u, s, v] = torch.svd(input=svd_data)
        s = torch.diag(s)
        big_c = u[:data_dim, :hidden_dim]
        big_z = s[:hidden_dim, :hidden_dim] @ v[:, :hidden_dim].transpose(1, 0)
        big_a = big_z[:, 1:] @ torch.linalg.pinv(big_z[:, 0:-1])
        return big_a, big_c, big_z

    def _log_gauss(self, x, mu, sigma):
        """
        :param x: 观测变量
        :param mu: 均值
        :param sigma: 协方差
        :return:
        """
        residual, data_dimension = x - mu, mu.shape[0]

        # sigma_inverse = torch.linalg.inv(sigma)
        log_exp_term = 0.5 * residual.transpose(1, 0) @ torch.linalg.solve(sigma, residual)
        # print(f"[Info] The sigma inverse is {sigma_inverse}")
        previous_term = -1.0 * (0.5 * data_dimension * np.log(2 * np.pi) + 0.5 * torch.det(sigma))
        total_term = log_exp_term.squeeze() + previous_term.squeeze()
        return total_term
    def _self_inverse(self, matrix):
        temp_matrix = torch.linalg.inv(torch.linalg.cholesky(matrix))
        return temp_matrix @ temp_matrix.T

    def kalman_filter(self, data):
        """
        :param data:
        :return:
        """
        temp_filtering_mean, temp_filtering_cov = [], []
        likelihood_list = []
        temp_filtering_mean.append(self.init_mean)
        temp_filtering_cov.append(self.init_cov)


        for t in range(self.seq_length):

            # 先验转移
            prior_mean = self.transition_matrix @ temp_filtering_mean[-1]
            prior_cov = self.transition_matrix @ temp_filtering_cov[-1] @ self.transition_matrix.transpose(1, 0) + self.transition_noise

            # 观测方程
            obs_mean = self.observation_matrix @ prior_mean
            obs_cov = self.observation_matrix @ prior_cov @ self.observation_matrix.transpose(1, 0) + self.observation_noise

            # 新息
            residual = data[:, t:t+1] - obs_mean
            # kalman_gain = prior_cov @ self.observation_matrix.transpose(1, 0) @ torch.linalg.inv(obs_cov)
            # kalman_gain = prior_cov @ self.observation_matrix.transpose(1, 0) @ self._self_inverse(obs_cov)

            # 更新滤波均值、滤波方差
            filtered_mean = prior_mean + prior_cov @ self.observation_matrix.transpose(1, 0) @ torch.linalg.solve(obs_cov, residual)
            # filtered_cov = prior_cov - kalman_gain @ obs_cov @ kalman_gain.transpose(1, 0)
            filtered_cov = prior_cov - prior_cov @ self.observation_matrix.transpose(1, 0) @ torch.linalg.inv(obs_cov) @ self.observation_matrix @ prior_cov
            temp_filtering_mean.append(filtered_mean)
            temp_filtering_cov.append(filtered_cov)

            likelihood_list.append(self._log_gauss(x=data[:, t:t+1],
                                                   mu=obs_mean,
                                                   sigma=obs_cov))

        return temp_filtering_mean, temp_filtering_cov, likelihood_list

    def rts_smoother(self, filtering_mean, filtering_cov):

        temp_smoothing_mean, temp_smoothing_cov, gain_list = [], [], []
        temp_smoothing_mean.append(filtering_mean[-1])
        temp_smoothing_cov.append(filtering_cov[-1])
        # print(len(filtering_mean))

        for t in range(self.seq_length - 1, -1, -1):
            # print(f'[Info] Time {t} start smoothing!')
            # 先验均值、方差
            prior_mean = self.transition_matrix @ filtering_mean[t]
            prior_cov = self.transition_matrix @ filtering_cov[t] @ self.transition_matrix.transpose(1, 0) + self.transition_noise

            # rts gain
            residual_mean = temp_smoothing_mean[-1] - prior_mean
            residual_cov = temp_smoothing_cov[-1] - prior_cov
            gain_matrix = filtering_cov[t] @ self.transition_matrix.transpose(1, 0) @ torch.linalg.inv(prior_cov)

            # 平滑均值、方差
            smooth_mean = filtering_mean[t] + filtering_cov[t] @ self.transition_matrix.transpose(1, 0) @ torch.linalg.solve(prior_cov, residual_mean) #residual_mean
            smooth_cov = filtering_cov[t] + gain_matrix @ residual_cov @ gain_matrix.transpose(1, 0)

            temp_smoothing_mean.append(smooth_mean)
            temp_smoothing_cov.append(smooth_cov)
            gain_list.append(gain_matrix)

        temp_smoothing_mean = list(reversed(temp_smoothing_mean))
        temp_smoothing_cov = list(reversed(temp_smoothing_cov))
        return temp_smoothing_mean, temp_smoothing_cov, gain_list

    def temp_params(self, smooth_mean_list, smooth_cov_list, gain_list, data):
        big_sigma = 1.0 / self.seq_length * sum([smooth_cov_list[idx] + smooth_mean_list[idx] @ smooth_mean_list[idx].transpose(1, 0) for idx in range(self.seq_length)])
        big_phi = 1.0 / self.seq_length * sum([smooth_cov_list[:-1][idx] + smooth_mean_list[:-1][idx] @ smooth_mean_list[:-1][idx].transpose(1, 0) for idx in range(self.seq_length)])

        big_b = 1.0 / self.seq_length * sum([data[:, idx:idx+1] @ smooth_mean_list[1:][idx].transpose(1, 0) for idx in range(self.seq_length)])
        big_c = 1.0 / self.seq_length * sum([smooth_cov_list[1:][idx] @ gain_list[idx].transpose(1, 0) + smooth_mean_list[1:][idx] @ smooth_mean_list[:-1][idx].transpose(1, 0) for idx in range(self.seq_length)])
        big_d = 1.0 / self.seq_length * sum([data[:, idx:idx+1] @ data[:, idx:idx + 1].transpose(1, 0) for idx in range(self.seq_length)])
        return big_sigma, big_phi, big_b, big_c, big_d

    def param_optimization(self, big_sigma, big_phi, big_b, big_c, big_d):
        self.transition_matrix = big_c @ torch.linalg.inv(big_phi)
        self.transition_noise = big_sigma - big_c @ self.transition_matrix.transpose(1, 0) - self.transition_matrix @ big_c.transpose(1, 0) + self.transition_matrix @ big_phi @ self.transition_matrix.transpose(1, 0)
        self.observation_matrix = big_b @ torch.linalg.inv(big_sigma)
        self.observation_noise = big_d - self.observation_matrix @ big_b.transpose(1, 0) - big_b @ self.observation_matrix.transpose(1, 0) + self.observation_matrix @ big_sigma @ self.observation_matrix.transpose(1, 0)

    def expectation_maximization(self, data, iter_time):
        filtering_mean, filtering_cov, likelihood_list = self.kalman_filter(data=data)
        smoothing_mean, smoothing_cov, gain_list = self.rts_smoother(filtering_mean=filtering_mean,
                                                                     filtering_cov=filtering_cov)


        print(f'[Info] The iter time is {iter_time + 1}, likelihood list is {sum(likelihood_list).data}')
        params_tuple = self.temp_params(smooth_mean_list=smoothing_mean,
                                                smooth_cov_list=smoothing_cov,
                                                gain_list=gain_list, data=data)
        big_sigma, big_phi, big_b, big_c, big_d = params_tuple[0], params_tuple[1], params_tuple[2], params_tuple[3], params_tuple[4]
        self.param_optimization(big_sigma=big_sigma, big_phi=big_phi, big_b=big_b, big_c=big_c, big_d=big_d)

        # print(f"[Info] 滤波序列长度: {len(filtering_mean), len(filtering_cov), len(likelihood_list)}")
        # print(f"[Info] 平滑序列长度: {len(smoothing_mean), len(smoothing_cov), len(gain_list)}")
        #
        # print("[Info] 干活完事儿了!")

        return sum(likelihood_list).data

    def forward(self, data):
        """
        :param data: [input_dim, seq_length]
        :return:
        """

        # self.transition_matrix, self.observation_matrix, _ = self.subspace_initialize(data=data,
        #                                                                               hidden_dim=self.hidden_dim,
        #                                                                               stacking_order=3)

        self.seq_length = data.shape[0]
        self.likelihood_list = []
        for iter_time in range(100):

            likelihood_value = self.expectation_maximization(data=data, iter_time=iter_time)
            self.likelihood_list.append(likelihood_value)

            if (iter_time >= 1):
                if (np.abs(self.likelihood_list[-2] - self.likelihood_list[-1]) / np.abs(self.likelihood_list[-2]) <= 1.0e-4):
                    break

        return self
