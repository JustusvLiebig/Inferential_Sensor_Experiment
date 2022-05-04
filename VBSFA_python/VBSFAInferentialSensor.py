import numpy as np
import pandas as pd
import scipy.special
from scipy import linalg
import copy
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

class VariationalHyperparameters(object):
    def __init__(self, x_dimension, y_dimension, factor_dimension, data_number):
        """
        变分超参数结构体
        :param x_dimension:
        :param y_dimension:
        :param factor_dimension:
        :param data_number:
        """
        super(VariationalHyperparameters, self).__init__()

        self.x_dimension = x_dimension
        self.y_dimension = y_dimension
        self.factor_dimension = factor_dimension

        self.data_number = data_number

        # p dimension: [x_dimension + y_dimension]
        self.p_dimension = x_dimension + y_dimension

        # k_dimension: [factor_dimension + 1]
        self.k_dimension = factor_dimension + 1

        # A: (data dimension * factor dimension + 1) dimension, each column gaussian
        self.a_mean = 0.0 * np.ones((self.k_dimension, self.p_dimension))
        self.a_mean[-1, :] = 1.0

        # prior_A_sigma: [factor_dimension + 1, factor_dimension + 1, data_dimension]
        # temp_sigma = np.eye(self.k_dimension)
        temp_sigma = np.ones((self.k_dimension, self.k_dimension))
        temp_sigma[self.k_dimension - 1, self.k_dimension - 1] = 0.0
        self.a_sigma = np.concatenate([np.expand_dims(temp_sigma, axis=2) for p in range(self.p_dimension)], axis=-1)

        # alpha a [factor_dimension + 1, 1], alpha b [factor_dimension + 1, 1]
        self.alpha_a = 1.0e-5 * np.ones((self.k_dimension, 1))
        self.alpha_b = 1.0e-5 * np.ones((self.k_dimension, 1))

        # prior phi a [data_dimension, data_dimension]
        self.phi_a = 1.0e-6 * np.ones((self.p_dimension, 1))
        self.phi_b = 1.0e-6 * np.ones((self.p_dimension, 1))

        # latent factor
        self.s_mean = np.zeros((self.k_dimension, self.data_number))
        self.s_mean[self.k_dimension-1, :] = np.ones((1, self.data_number))
        # self.s_sigma = np.ones((self.k_dimension - 1, self.k_dimension - 1))
        self.s_sigma = 0.0 * np.eye(self.k_dimension - 1)
        self.s_sigma = np.concatenate([self.s_sigma, np.ones((1, self.factor_dimension))], axis=0)
        self.s_sigma = np.concatenate([self.s_sigma, np.ones((self.k_dimension, 1))], axis=1)



class VariationalBayesianFactorAnalysis(object):
    def __init__(self, data, x_dimension, y_dimension, factor_dimension):
        super(VariationalBayesianFactorAnalysis, self).__init__()
        # data dimension
        self.data = data
        self.data_number = data.shape[1]
        self.x_dimension = x_dimension
        self.y_dimension = y_dimension
        self.factor_dimension = factor_dimension

        # p dimension: [x_dimension + y_dimension]
        self.p_dimension = x_dimension + y_dimension

        # k_dimension: [factor_dimension + 1]
        self.k_dimension = factor_dimension + 1

        # prior hyper-parameters
        self.prior_hyperparameters = VariationalHyperparameters(x_dimension=x_dimension, y_dimension=y_dimension, factor_dimension=factor_dimension, data_number=self.data_number)

        # posterior hyper-parameters
        self.posterior_hyperparameters = VariationalHyperparameters(x_dimension=x_dimension, y_dimension=y_dimension, factor_dimension=factor_dimension, data_number=self.data_number)

        self.elbo_list = list()


    def inverse_matrix(self, matrix):
        """
        using cholesky decomposition for the inverse matrix
        :param matrix:
        :return:
        """
        temp_matrix = scipy.linalg.inv(scipy.linalg.cholesky(matrix))
        return temp_matrix @ temp_matrix.T
    """"""

    # def inverse_matrix(self, matrix):
    #
    #     return np.linalg.inv(matrix)


    def vb_e_step(self, posterior):
        """
        :param prior: posterior class
        :return: posterior
        """

        # inference on s [factor dimension + 1, data number]

        # [data dimension, 1]
        exp_phi = posterior.phi_a / posterior.phi_b

        # [factor dimension, factor dimension]
        # the sigma dimension
        s_sigma_temp = np.zeros((self.factor_dimension, self.factor_dimension))

        for p in range(self.p_dimension):
            # 扩展了一行, 所以这里a应该是倒数后两个
            s_sigma_temp = s_sigma_temp + exp_phi[p, 0] \
                           * np.expand_dims(posterior.a_mean[:self.factor_dimension, p], axis=1) \
                           @ np.expand_dims(posterior.a_mean[:self.factor_dimension, p], axis=0) \
                           + posterior.a_sigma[:self.factor_dimension, :self.factor_dimension, p]

        s_sigma_temp = (np.eye(self.factor_dimension) + s_sigma_temp)

        # the cholesky inverse matrix
        # try:
        #
        # except:
        #     print(s_sigma_temp)


        # update the sigma s in the posterior hyper-parameters
        # update the s mean
        # posterior_s_sigma = np.linalg.solve(s_sigma_temp, np.eye(self.factor_dimension))
        posterior_s_sigma = self.inverse_matrix(s_sigma_temp)
        # posterior_s_mean = np.dot(np.dot(np.dot(posterior_s_sigma, posterior.a_mean[:self.factor_dimension, :]), np.diag(exp_phi.reshape((-1)))), self.data)
        # solve method
        posterior_s_mean = scipy.linalg.solve(s_sigma_temp, (posterior.a_mean[:self.factor_dimension, :] @ np.diag(exp_phi.reshape((-1))) @ self.data ))

        # posterior_s_mean = posterior_s_sigma @ (posterior.a_mean[:self.factor_dimension, :] @ np.diag(exp_phi.reshape((-1))) @ self.data)



        posterior_s_mean = np.concatenate([posterior_s_mean, np.ones((1, self.data_number))], axis=0)

        posterior_s_sigma = np.concatenate(
            [np.concatenate([posterior_s_sigma, np.zeros((self.factor_dimension, 1))], axis=1),
             np.zeros((1, self.k_dimension))], axis=0)

        # return the posterior hyper parameters
        posterior.s_sigma = posterior_s_sigma
        posterior.s_mean = posterior_s_mean

        return posterior

    def vb_m_step(self, prior, posterior):
        """
        :param prior: The prior class
        :param posterior: The posterior class
        :return: posterior distribution
        """
        # inference on A [data dimension, data dimension]
        exp_phi = posterior.phi_a / posterior.phi_b

        temp_phi = np.zeros((self.k_dimension, self.k_dimension))
        temp_phi = temp_phi + self.data_number * posterior.s_sigma \
                   + np.sum(np.einsum("ijk, ikn -> ijn", np.expand_dims(posterior.s_mean.T, axis=2),
                                      np.expand_dims(posterior.s_mean.T, axis=1)), axis=0)

        exp_alpha = posterior.alpha_a / posterior.alpha_b
        # exp_alpha = np.concatenate(exp_alpha)

        # inference a
        for p in range(self.p_dimension):
            # a_sigma

            temp_a_sigma_inv = (exp_phi[p, 0] * temp_phi + np.diag(exp_alpha.reshape((-1))))

            # posterior.a_sigma[:, :, p] = np.linalg.solve(temp_a_sigma_inv, np.eye(self.k_dimension))

            posterior.a_sigma[:, :, p] = self.inverse_matrix(temp_a_sigma_inv)

            # a_mean
            posterior.a_mean[:, p] = exp_phi[p, 0] * scipy.linalg.solve(temp_a_sigma_inv, (np.sum((self.data[p:p+1, :].repeat(self.k_dimension, axis=0) * posterior.s_mean), axis=1)))
            # posterior.a_mean[:, p] = exp_phi[p, 0] * self.inverse_matrix(temp_a_sigma_inv) @ (np.sum((self.data[p:p + 1, :].repeat(self.k_dimension, axis=0) * posterior.s_mean), axis=1))


        # inference alpha
        posterior.alpha_a[:self.factor_dimension, :] = prior.alpha_a[:self.factor_dimension, :] + 0.5 * self.p_dimension
        temp_alpha_b = np.sum(posterior.a_sigma[:self.factor_dimension, :self.factor_dimension, :], axis=2) \
                       + np.sum(np.einsum("ijk, ikn -> ijn", np.expand_dims(posterior.a_mean[:self.factor_dimension, :].T, axis=2),
                                          np.expand_dims(posterior.a_mean[:self.factor_dimension, :].T, axis=1)),
                                axis=0)
        posterior.alpha_b[:self.factor_dimension, :] = prior.alpha_b[:self.factor_dimension, :] + 0.5 * np.expand_dims(np.diagonal(temp_alpha_b), axis=1)

        # inference phi
        posterior.phi_a = prior.phi_a + 0.5 * self.data_number
        a_quadratic = posterior.a_sigma[:self.factor_dimension, :self.factor_dimension] \
                      + np.einsum("ijk, ikn -> ijn", np.expand_dims(posterior.a_mean[:self.factor_dimension, :].T, axis=2),
                                  np.expand_dims(posterior.a_mean[:self.factor_dimension, :].T, axis=1)).transpose((1, 2, 0))

        phi_s_sigma = np.expand_dims(posterior.s_sigma[:self.factor_dimension, :self.factor_dimension], axis=2).repeat(self.data_number, axis=2)

        phi_s_mu = np.einsum("ijk, ikn -> ijn", np.expand_dims(posterior.s_mean[:self.factor_dimension, :].T, axis=2),
                             np.expand_dims(posterior.s_mean[:self.factor_dimension, :].T, axis=1)).transpose((1, 2, 0))

        # phi_s_mu = np.matmul(posterior.s_mean[:self.factor_dimension, :].T.reshape((self.data_number, self.factor_dimension, 1)),
        #                      posterior.s_mean[:self.factor_dimension, :].T.reshape((self.data_number, 1, self.factor_dimension))).transpose((1, 2, 0))

        phi_s_quadric = phi_s_mu + phi_s_sigma

        for p in range(self.p_dimension):
            temp_quadratic1 = np.sum((self.data[p:p+1, :] * self.data[p:p+1, :]), axis=1)
            temp_binary = 2 * np.sum(self.data[p, :].reshape((1, -1)) * (np.expand_dims(posterior.a_mean[:self.factor_dimension, p], axis=0) @ posterior.s_mean[:self.factor_dimension, :]), axis=1)

            # times term
            aj_quadric = np.expand_dims(a_quadratic[:, :, p], axis=2).repeat(self.data_number, axis=2)
            temp_quadratic2 = np.matmul(aj_quadric.transpose((2, 0, 1)), phi_s_quadric.transpose((2, 0, 1)))
            temp_quadratic2 = np.sum(temp_quadratic2, axis=0)
            temp_quadratic2 = np.trace(temp_quadratic2)
            posterior.phi_b[p, 0] = prior.phi_b[p, 0] + 0.5 * (temp_quadratic1 - temp_binary + temp_quadratic2)

        return posterior

    def elbo_calculation(self, posterior, prior):

        temp_log_exp_phi = scipy.special.digamma(posterior.phi_a) - np.log(posterior.phi_b)
        temp_exp_phi = posterior.phi_a / posterior.phi_b
        j_value = -0.5 * self.data_number * self.p_dimension * np.log(2 * np.pi) \
                  + 0.5 * self.data_number * np.sum(temp_log_exp_phi) \
                  + np.sum((temp_exp_phi * prior.phi_b * posterior.phi_a), axis=0)

        temp_kl_s = np.sum(np.einsum("ijk, ikn -> ijn", np.expand_dims(posterior.s_mean.T, axis=2),
                                     np.expand_dims(posterior.s_mean.transpose(1, 0), axis=1)),
                           axis=2)
        kl_s = 0.5 * self.data_number \
               + 0.5 * self.data_number * np.trace(np.eye(self.k_dimension) - posterior.s_sigma) \
               - 0.5 * np.trace(temp_kl_s)

        alpha_exp = posterior.alpha_a / posterior.alpha_b
        temp_kl_A_I = np.expand_dims(np.eye(self.k_dimension), axis=2).repeat(self.p_dimension, axis=2)
        temp_kl_A_A = posterior.a_sigma \
                      + np.matmul(np.expand_dims(posterior.a_mean.T, axis=2),
                                  np.expand_dims(posterior.a_mean.T, axis=1)).transpose((1, 2, 0))
        temp_kl_A_alpha = np.expand_dims(np.diag(alpha_exp.reshape((-1))), axis=2).repeat(self.p_dimension, axis=2)
        temp_kl_A_inner = np.matmul(temp_kl_A_A.transpose(2, 0, 1), temp_kl_A_alpha.transpose(2, 0, 1)).transpose((1, 2, 0))
        temp_kl_A_trace = np.trace((np.sum(temp_kl_A_I - temp_kl_A_inner, axis=2)))

        temp_kl_a_sigma = 0
        for p in range(self.p_dimension):
            temp_kl_a_sigma = temp_kl_a_sigma + np.log(np.linalg.det(posterior.a_sigma[:, :, p]))
        kl_a = 0.5 * self.p_dimension * np.sum((scipy.special.digamma(posterior.alpha_a) - np.log(posterior.alpha_b)), axis=0) \
               + 0.5 * (temp_kl_a_sigma + temp_kl_A_trace)


        temp_kl_alpha = posterior.alpha_a * np.log(posterior.alpha_b) - prior.alpha_a * np.log(prior.alpha_b) \
                        - (scipy.special.gammaln(posterior.alpha_a) - scipy.special.gammaln(prior.alpha_a)) \
                        + prior.alpha_b * (posterior.alpha_a / posterior.alpha_b) - posterior.alpha_a \
                        + (posterior.alpha_a - prior.alpha_a) \
                        * (scipy.special.digamma(posterior.alpha_a) - np.log(posterior.alpha_b))
        kl_alpha = np.sum(temp_kl_alpha, axis=0)

        temp_kl_phi = posterior.phi_a * np.log(posterior.phi_b) \
                      - prior.phi_a * np.log(prior.phi_b) \
                      - (scipy.special.gammaln(posterior.phi_a) - scipy.special.gammaln(prior.phi_a)) \
                      + (prior.phi_b * (posterior.phi_a / posterior.phi_b)) - posterior.phi_a \
                      + (posterior.phi_a - prior.phi_a) * (scipy.special.digamma(posterior.phi_a) - np.log(posterior.phi_b))

        kl_phi = np.sum(temp_kl_phi, axis=0)

        # sum them up
        elbo = j_value - kl_s - kl_a - kl_alpha - kl_phi

        return elbo



    def fit(self, loop_number):
        for loop in range(loop_number):
            self.posterior_hyperparameters = self.vb_e_step(posterior=self.posterior_hyperparameters)

            self.posterior_hyperparameters = self.vb_m_step(prior=self.prior_hyperparameters, posterior=self.posterior_hyperparameters)

            elbo_value = self.elbo_calculation(prior=self.prior_hyperparameters, posterior=self.posterior_hyperparameters)
            self.elbo_list.append(elbo_value)
            if loop >= 1:
                if np.abs((self.elbo_list[-1] - self.elbo_list[-2]) / (self.elbo_list[-2])) <= 1.0e-9:
                    break

        #     print(f'Iteration {loop + 1}, the evidence lower bound value is {elbo_value}')
        # print('[Info] The optimization finished')

        return self

class VBSFASoftSensor(object):
    def __init__(self, x_dimension, y_dimension, factor_dimension, loop_number):
        super(VBSFASoftSensor, self).__init__()




        self.x_dimension = x_dimension
        self.y_dimension = y_dimension
        self.factor_dimension = factor_dimension
        self.loop_number = loop_number

    def fit_stream(self, data, past_posterior):
        """
        :param data:
        :return:
        """
        data_x = data[:, :self.x_dimension]
        data_y = data[:, self.x_dimension:].reshape((-1, self.y_dimension))

        self.input_scaler = StandardScaler()
        self.output_scaler = StandardScaler()

        data_x_scaled = self.input_scaler.fit_transform(data_x)
        data_y_scaled = self.output_scaler.fit_transform(data_y)
        scaled_data = np.concatenate([data_x_scaled, data_y_scaled], axis=1)
        """
        scaled_data = np.concatenate([data_x, data_y], axis=1)
        """
        self.VBSFAClass = VariationalBayesianFactorAnalysis(data=scaled_data.T,
                                                            x_dimension=self.x_dimension,
                                                            y_dimension=self.y_dimension,
                                                            factor_dimension=self.factor_dimension)
        self.VBSFAClass.prior_hyperparameters = past_posterior
        self.VBSFAClass = self.VBSFAClass.fit(loop_number=self.loop_number)
        self.regressor_hyperparameters = copy.deepcopy(self.VBSFAClass.posterior_hyperparameters)
        return self


    def fit(self, data):
        """
        :param data:
        :return:
        """
        data_x = data[:, :self.x_dimension]
        data_y = data[:, self.x_dimension:].reshape((-1, self.y_dimension))

        self.input_scaler = StandardScaler()
        self.output_scaler = StandardScaler()

        data_x_scaled = self.input_scaler.fit_transform(data_x)
        data_y_scaled = self.output_scaler.fit_transform(data_y)
        scaled_data = np.concatenate([data_x_scaled, data_y_scaled], axis=1)
        """
        scaled_data = np.concatenate([data_x, data_y], axis=1)
        """
        self.VBSFAClass = VariationalBayesianFactorAnalysis(data=scaled_data.T,
                                                            x_dimension=self.x_dimension,
                                                            y_dimension=self.y_dimension,
                                                            factor_dimension=self.factor_dimension).fit(loop_number=self.loop_number)
        self.regressor_hyperparameters = copy.deepcopy(self.VBSFAClass.posterior_hyperparameters)
        return self



    def predict(self, input_data):
        """
        :param input_data: data_number * data dimension
        :return: output_data
        """
        test_data_number = input_data.shape[0]

        scaled_x = self.input_scaler.transform(input_data).T
        """
        scaled_x = input_data.T
        """
        s_sigma = self.regressor_hyperparameters.s_sigma[:self.factor_dimension, :self.factor_dimension]
        phi_exp = self.regressor_hyperparameters.phi_a / self.regressor_hyperparameters.phi_b
        x_phi = np.diag(phi_exp[:self.x_dimension, :].reshape((-1)))

        # y_phi = np.diag(phi_exp[self.x_dimension:, :].reshape((-1)))

        A_x = self.regressor_hyperparameters.a_mean[:self.factor_dimension, :self.x_dimension]
        A_y = self.regressor_hyperparameters.a_mean[:, self.x_dimension:]

        s_tilde = s_sigma @ A_x @ x_phi @ scaled_x
        s_tilde = np.concatenate([s_tilde, np.ones((1, test_data_number))], axis=0)

        y_predict = (A_y.T @ s_tilde).T
        """"""
        y_predict = self.output_scaler.inverse_transform(y_predict)

        return y_predict



if __name__ == '__main__':
    import datetime

    np.random.seed(seed=1024)

    total_x_dimension = 6
    data_number = 1000
    loop_number = 100
    plot_elbo = False

    train_percentage = 0.6

    x_data = np.random.randn(data_number, total_x_dimension)

    y_data = x_data @ np.random.randint(0, 2, total_x_dimension).reshape((-1, 1))
    total_data = np.concatenate([x_data, y_data], axis=1)

    print(total_data.shape)

    x_dimension, y_dimension = x_data.shape[1], y_data.shape[1]
    factor_dimension = 3

    start_time = datetime.datetime.now()
    sfa_model = VariationalBayesianFactorAnalysis(data=total_data.T, x_dimension=x_dimension,
                                                  y_dimension=y_dimension, factor_dimension=factor_dimension)
    sfa_model.fit(loop_number=loop_number)

    end_time = datetime.datetime.now()
    print(f'We use {(end_time - start_time).seconds} seconds')


    if plot_elbo:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(list(range(len(sfa_model.elbo_list))), sfa_model.elbo_list, color='r', label='elbo value')
        plt.legend()
        plt.show()

    train_data = total_data[:int(data_number * train_percentage), :]
    test_data = total_data[int(data_number * train_percentage):, :]

    soft_sensor_model = VBSFASoftSensor(x_dimension=x_dimension, y_dimension=y_dimension,
                                        factor_dimension=factor_dimension, loop_number=loop_number)
    soft_sensor_model.fit(data=train_data)
    test_data_x = test_data[:, :x_dimension]
    predict_test = soft_sensor_model.predict(input_data=test_data_x)

    print(f'[Info] The predict test shape is {predict_test.shape}')

    real_test = test_data[:, x_dimension:].reshape((-1, y_dimension))

    test_rmse = np.sqrt(mean_squared_error(predict_test, real_test))
    test_r2 = r2_score(predict_test, real_test)

    print(f'[Info] The output data shape is {predict_test.shape}, r2 is {test_r2}, rmse is {test_rmse}')




