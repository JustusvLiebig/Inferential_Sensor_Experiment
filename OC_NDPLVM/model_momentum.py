import numpy as np
import torch
import torch.nn as nn
from model_transition import PlanarFlow



class TransitionModel(nn.Module):
    def __init__(self, transition_dim, feature_dim):
        super(TransitionModel, self).__init__()
        self.transition_dim = transition_dim
        self.feature_dim = feature_dim

        self.transition_network = list()
        # the hidden transiton
        for i in range(transition_dim):
            self.transition_network.append(PlanarFlow())
            self.add_module(f"plannar_transition_at_{i + 1}", self.transition_network[-1])

        self.feature_projection = nn.Linear(feature_dim, transition_dim)
        self.transition_dim = list(range(transition_dim))

    def forward(self, mu_past, sigma_past, covariate_feature):
        """
        :param mu_past: [batch_size, transtion_dim]
        :param sigma_past: [batch_size, transtion_dim]
        :param covariate_feature: [batch_size, feature_dim]
        :return:
        """

        # 均值和方差的term干出来
        mu_and_sigma_list = torch.cat(list(map(lambda x: self.transition_network[x](mu_past[:, x:x+1]),
                                               self.transition_dim)), dim=-1)

        # 计算期望term
        mu_future_term = mu_and_sigma_list[:, 0, :]
        mu_future = mu_future_term + self.feature_projection(covariate_feature) + mu_past

        # 计算方差term
        sigma_future_coefficient = mu_and_sigma_list[:, 1, :]
        # sigma_future = sigma_future_coefficient * sigma_past * sigma_future_coefficient
        sigma_future = 2 * sigma_future_coefficient * sigma_past + sigma_past + 1.0

        return mu_future, sigma_future


class DynamicVAEModelWeak(nn.Module):
    def __init__(self, covariate_dim, transition_dim, output_dim, control_dim, rnn_dim, rnn_layers):
        super(DynamicVAEModelWeak, self).__init__()
        self.covariate_dim = covariate_dim
        self.transition_dim = transition_dim
        self.rnn_dim = rnn_dim
        self.output_dim = output_dim
        self.control_dim = control_dim
        self.rnn_layers = rnn_layers


        self.rnn_network = nn.GRU(input_size=covariate_dim, hidden_size=rnn_dim,
                                  num_layers=rnn_layers, batch_first=True)

        self.control_network = nn.Sequential(nn.Linear(output_dim, control_dim),
                                             nn.Tanh(),
                                             nn.Linear(control_dim, transition_dim * 2))

        self.control_network_mu = nn.Sequential(nn.Linear(transition_dim, transition_dim))
        self.control_network_var = nn.Sequential(nn.Linear(transition_dim, transition_dim))

        self.transition_network = TransitionModel(transition_dim=transition_dim, feature_dim=rnn_dim)

        self.output_network = nn.Sequential(nn.Linear(transition_dim, output_dim))

    def _update_state(self, mu_prior, var_prior, mu_posterior, var_posterior):
        denominator = (var_prior * var_posterior) / (var_posterior + var_prior)
        sum_temp = (var_posterior + var_prior)
        weight_prior, weight_posterior = var_posterior / sum_temp, var_prior / sum_temp
        mu_weighted = weight_prior * mu_prior + weight_posterior * mu_posterior
        # mu_weighted = mu_prior + mu_posterior
        return mu_weighted, denominator


    def forward(self, past_input, past_label, future_input):

        past_length = past_input.shape[1]
        future_length = future_input.shape[1]
        batch_size = past_input.shape[0]

        # rnn start token length
        rnn_start = torch.zeros([self.rnn_layers, batch_size, self.rnn_dim]).to(past_input.device)
        covariate_feature, _ = self.rnn_network(torch.cat([past_input, future_input], dim=1), rnn_start)

        output_prediction_list = list()

        prior_mu_list = list()
        prior_var_list = list()

        prior_mu_list.append(torch.zeros([batch_size, self.transition_dim]).to(past_input.device))
        prior_var_list.append(torch.ones([batch_size, self.transition_dim]).to(past_input.device))

        posterior_mu_list = list()
        posterior_var_list = list()

        posterior_mu_list.append(torch.zeros([batch_size, self.transition_dim]).to(past_input.device))
        posterior_var_list.append(torch.ones([batch_size, self.transition_dim]).to(past_input.device))

        control_mu_list = list()
        control_var_list = list()

        for idx_term in range(past_length):

            mu_past = posterior_mu_list[-1]
            var_past = posterior_var_list[-1]

            # prediction
            mu_prior, var_prior = self.transition_network(mu_past=mu_past, sigma_past=var_past,
                                                          covariate_feature=covariate_feature[:, idx_term, :])
            # append result
            prior_mu_list.append(mu_prior)
            prior_var_list.append(var_prior)

            # sample and prediction
            if self.training:
                sampled_hidden = mu_prior # + torch.sqrt(var_prior) * torch.randn_like(mu_prior)
                output_prediction = self.output_network(sampled_hidden)
                output_prediction_list.append(output_prediction)
            else:
                sampled_hidden = mu_prior # + torch.sqrt(var_prior) * torch.randn_like(mu_prior)
                output_prediction = self.output_network(sampled_hidden)
                output_prediction_list.append(output_prediction)

            # backward control
            [control_mu, control_log_std] = self.control_network(past_label[:, idx_term, :]).chunk(chunks=2, dim=-1)
            control_mu = control_mu - self.control_network_mu(mu_prior)
            control_log_std = control_log_std # + self.control_network_var(var_prior)
            control_std = nn.functional.softplus(control_log_std)
            # control_log_var = torch.tanh(control_log_var)

            # update state
            # mu_posterior = mu_prior + control_mu
            # var_posterior = var_prior + torch.exp(control_log_var)

            mu_posterior, var_posterior = self._update_state(mu_prior=mu_prior, var_prior=var_prior,
                                                             mu_posterior=control_mu,
                                                             var_posterior=control_std)
            control_mu_list.append(control_mu)
            control_var_list.append(control_std)

            # append tensor
            posterior_mu_list.append(mu_posterior)
            posterior_var_list.append(var_posterior)



        # future
        for idx_term in range(future_length):
            mu_past = posterior_mu_list[-1]
            sigma_past = posterior_var_list[-1]

            # prediction
            mu_prior, var_prior = self.transition_network(mu_past=mu_past, sigma_past=sigma_past,
                                                          covariate_feature=covariate_feature[:, idx_term + past_length, :])
            # append result
            prior_mu_list.append(mu_prior)
            prior_var_list.append(var_prior)

            # sample and prediction
            if self.training:
                sampled_hidden = mu_prior # + torch.sqrt(var_prior) * torch.randn_like(mu_prior)
                output_prediction = self.output_network(sampled_hidden)
                output_prediction_list.append(output_prediction)
            else:
                sampled_hidden = mu_prior # + torch.sqrt(var_prior) * torch.randn_like(mu_prior)
                output_prediction = self.output_network(sampled_hidden)
                output_prediction_list.append(output_prediction)


            # backward control
            [control_mu, control_log_std] = self.control_network(output_prediction).chunk(chunks=2, dim=-1)
            control_mu = control_mu - self.control_network_mu(mu_prior)
            control_log_std = control_log_std # + self.control_network_var(var_prior)
            control_std = nn.functional.softplus(control_log_std)
            # control_log_var = torch.tanh(control_log_var)

            # update state
            # mu_posterior = mu_prior + control_mu
            # var_posterior = var_prior + torch.exp(control_log_var)
            mu_posterior, var_posterior = self._update_state(mu_prior=mu_prior, var_prior=var_prior,
                                                             mu_posterior=control_mu,
                                                             var_posterior=control_std)

            control_mu_list.append(control_mu)
            control_var_list.append(control_std)

            # append the list
            posterior_mu_list.append(mu_posterior)
            posterior_var_list.append(var_posterior)

        output_prediction_sequence = torch.cat(list(map(lambda x: x.unsqueeze(1), output_prediction_list)), dim=1)
        posterior_mu_sequence = torch.cat(list(map(lambda x: x.unsqueeze(1), control_mu_list)), dim=1)[:, 1:, :]
        posterior_var_sequence = torch.cat(list(map(lambda x: x.unsqueeze(1), control_var_list)), dim=1)[:, 1:, :]
        return output_prediction_sequence, posterior_mu_sequence, posterior_var_sequence


