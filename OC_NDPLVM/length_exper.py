from utils import display_results, FasterDataset, setup_seed, create_file, display_window_results, obtain_window_results
from utils import obtain_train_valid_test
# from model_dynamic import DynamicVAEModel, DynamicVAEModelControl, DynamicVAEModelWeak
from model_momentum import DynamicVAEModelWeak
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.base import BaseEstimator, RegressorMixin
import torch
import torch.nn as nn
import numpy as np
import joblib
import pandas as pd
import os
from torch.utils.data import DataLoader


def construct_test_tensor(test_x, test_y, past_length, future_length):
    total_length = past_length + future_length

    iterator_list = list(range(test_x.shape[0] - total_length + 1))

    test_x_3d = list(map(lambda x: test_x[x: x + total_length, :], iterator_list))
    test_y_3d = list(map(lambda x: test_y[x: x + total_length, :], iterator_list))

    test_x_np = np.stack(test_x_3d, axis=0)
    test_y_np = np.stack(test_y_3d, axis=0)
    past_input, future_input = test_x_np[:, :past_length, :], test_x_np[:, past_length:, :]
    start_token, label_token = test_y_np[:, :past_length, :], test_y_np[:, past_length:, :]
    return past_input, future_input, start_token, label_token


class DynamicVAEModelInit(BaseEstimator, RegressorMixin):
    def __init__(self, input_dim, output_dim, transition_dim, control_dim, rnn_dim, rnn_layers,
                 past_length, future_length, num_epochs, batch_size, learning_rate, device, seed=1024):
        super(DynamicVAEModelInit, self).__init__()

        # Set seed
        setup_seed(seed=seed)
        create_file(model_name="DynamicVAE")

        # Parameter assignment
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.transition_dim = transition_dim
        self.control_dim = control_dim
        self.rnn_dim = rnn_dim
        self.rnn_layers = rnn_layers
        self.past_length = past_length
        self.future_length = future_length
        self.total_length = past_length + future_length
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device
        self.seed = seed

        self.model = DynamicVAEModelWeak(covariate_dim=input_dim, transition_dim=transition_dim,
                                         output_dim=output_dim, control_dim=control_dim,
                                         rnn_dim=rnn_dim, rnn_layers=rnn_layers).to(device)

        self.likelihood_loss_list = list()
        self.control_loss_list = list()
        self.total_loss_list = list()
        self.valid_loss_list = list()
        self.valid_loss_list.append(1.0e10)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        self.prediction_loss = nn.MSELoss(reduction="sum")

        self.input_scaler = StandardScaler()
        self.output_scaler = StandardScaler()

    # def self_loss_function(self, y_pred, y_true, control_mu, control_var):
    #     batch_size = y_true.shape[0]
    #     prediction_error = self.prediction_loss(input=y_true, target=y_pred) / batch_size
    #     l2_norm_control = control_mu * control_mu + control_var
    #     control_loss = torch.sum((l2_norm_control)) / (batch_size * (self.past_length + self.future_length))
    #     # control_loss = torch.sum((control_mu * control_mu)) / (
    #     #             batch_size * (self.past_length + self.future_length))
    #     total_loss = prediction_error + control_loss
    #     # print(total_loss.shape)
    #     return prediction_error, control_loss, total_loss

    def self_loss_function(self, y_pred, y_true, control_mu, control_var):
        batch_size = y_true.shape[0]
        prediction_error = self.prediction_loss(input=y_true, target=y_pred) / batch_size
        l2_norm_control = control_mu * control_mu + control_var
        control_loss = torch.sum((l2_norm_control)) / (batch_size * (self.past_length + self.future_length))
        total_loss = prediction_error + control_loss
        # print(total_loss.shape)
        return prediction_error, control_loss, total_loss

    def fit(self, input_data, output_data, valid_input, valid_output):

        scaled_input = self.input_scaler.fit_transform(input_data)
        scaled_output = self.output_scaler.fit_transform(output_data.reshape((-1, self.output_dim)))

        joblib.dump(self.input_scaler,
                    f'./DynamicVAE/input_scaler_{self.seed}_{self.batch_size}_{self.learning_rate}.pkl')
        joblib.dump(self.output_scaler,
                    f'./DynamicVAE/output_scaler_{self.seed}_{self.batch_size}_{self.learning_rate}.pkl')

        scaled_valid_input = self.input_scaler.transform(valid_input)
        scald_valid_output = self.output_scaler.transform(valid_output)

        input_iterator_list = list(range(input_data.shape[0] - self.total_length + 1))
        valid_iterator_list = list(range(valid_input.shape[0] - self.total_length + 1))

        scaled_input_3d_list = list(map(lambda x: scaled_input[x: x + self.total_length, :], input_iterator_list))
        scaled_output_3d_list = list(map(lambda x: scaled_output[x: x + self.total_length, :], input_iterator_list))

        valid_input_3d_list = list(map(lambda x: scaled_valid_input[x: x + self.total_length, :], valid_iterator_list))
        valid_output_3d_list = list(map(lambda x: scald_valid_output[x: x + self.total_length, :], valid_iterator_list))

        scaled_input_3d = np.stack(scaled_input_3d_list, 0)
        scaled_output_3d = np.stack(scaled_output_3d_list, 0)
        scaled_valid_input_3d = np.stack(valid_input_3d_list, 0)
        scaled_valid_output_3d = np.stack(valid_output_3d_list, 0)

        train_dataset = FasterDataset(data=torch.tensor(scaled_input_3d, dtype=torch.float32, device=self.device),
                                      label=torch.tensor(scaled_output_3d, dtype=torch.float32, device=self.device))

        valid_dataset = FasterDataset(data=torch.tensor(scaled_valid_input_3d, dtype=torch.float32, device=self.device),
                                      label=torch.tensor(scaled_valid_output_3d, dtype=torch.float32,
                                                         device=self.device))

        valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False)

        for epoch in range(self.num_epochs):
            total_loss_value = 0.0
            total_control_loss = 0.0
            total_likelihood_loss = 0.0
            train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=self.batch_size)
            self.model.train()
            for idx, (batch_x, batch_y) in enumerate(train_loader):
                start_token = batch_y[:, :self.past_length, :]
                label = batch_y[:, self.past_length:, :]

                past_input = batch_x[:, :self.past_length, :]
                future_input = batch_x[:, self.past_length:, :]

                prediction, posterior_mu, posterior_var = self.model(past_input=past_input,
                                                                     past_label=start_token,
                                                                     future_input=future_input)
                self.optimizer.zero_grad()
                prediction_loss, control_loss, energy_loss = self.self_loss_function(
                    y_pred=prediction[:, self.past_length:, :],
                    y_true=batch_y[:, self.past_length:, :],
                    control_mu=posterior_mu,
                    control_var=posterior_var)

                energy_loss.backward()
                self.optimizer.step()
                total_loss_value = total_loss_value + energy_loss.item()
                total_control_loss = total_control_loss + control_loss.item()
                total_likelihood_loss = total_likelihood_loss + prediction_loss.item()
            self.total_loss_list.append(total_loss_value)
            self.control_loss_list.append(total_control_loss)
            self.likelihood_loss_list.append(total_likelihood_loss)
            print(f"Epoch: {epoch + 1}, ",
                  f"total energy loss: {np.round(total_loss_value, 6)}, ",
                  f"total likelihood loss: {np.round(total_likelihood_loss, 6)}, ",
                  f"total control loss: {np.round(total_control_loss, 6)}. ")
            if ((epoch + 1) % 5 == 0):
                self.model.eval()
                with torch.no_grad():
                    total_valid_loss = 0.0
                    for batch_idx, (batch_x, batch_y) in enumerate(valid_loader):
                        batch_size = batch_x.shape[0]

                        start_token = batch_y[:, :self.past_length, :]
                        label = batch_y[:, self.past_length:, :]
                        past_input = batch_x[:, :self.past_length, :]
                        future_input = batch_x[:, self.past_length:, :]

                        prediction, _, _ = self.model(past_input=past_input,
                                                      past_label=start_token,
                                                      future_input=future_input)

                        valid_loss = self.prediction_loss(input=prediction[:, self.past_length:, :],
                                                          target=label) / batch_size
                        total_valid_loss = total_valid_loss + valid_loss.item()

                    print(f"[Info] Validating, loss: {np.round(total_valid_loss, 5)}")

                    if (total_valid_loss < self.valid_loss_list[-1]) & (epoch >= 20):
                        state = {'net': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(),
                                 'epoch': epoch}

                        torch.save(state,
                                   f'./DynamicVAE/DVAE_{self.seed}_{self.batch_size}_{self.learning_rate}_model.pth')
                        self.valid_loss_list.append(total_valid_loss)

        return self

    def predict(self, past_input, future_input, start_token):
        past_length = past_input.shape[1]
        future_length = future_input.shape[1]

        input_scaler = joblib.load(f'./DynamicVAE/input_scaler_{self.seed}_{self.batch_size}_{self.learning_rate}.pkl')
        output_scaler = joblib.load(
            f'./DynamicVAE/output_scaler_{self.seed}_{self.batch_size}_{self.learning_rate}.pkl')
        checkpoint = torch.load(f'./DynamicVAE/DVAE_{self.seed}_{self.batch_size}_{self.learning_rate}_model.pth')

        test_model = self.model
        test_model.load_state_dict(checkpoint['net'])

        scaled_past_input = list(map(lambda x: input_scaler.transform(past_input[:, x, :]), list(range(past_length))))
        scaled_past_input = np.stack(scaled_past_input, axis=1)
        scaled_future_input = list(
            map(lambda x: input_scaler.transform(future_input[:, x, :]), list(range(future_length))))
        scaled_future_input = np.stack(scaled_future_input, axis=1)

        scaled_start_token = list(
            map(lambda x: output_scaler.transform(start_token[:, x, :]), list(range(past_length))))
        scaled_start_token = np.stack(scaled_start_token, axis=1)

        past_input_tensor = torch.tensor(scaled_past_input, device=self.device, dtype=torch.float32)
        future_input_tensor = torch.tensor(scaled_future_input, device=self.device, dtype=torch.float32)
        start_token_tensor = torch.tensor(scaled_start_token, device=self.device, dtype=torch.float32)

        test_model.eval()
        with torch.no_grad():
            y, _, _ = test_model(past_input=past_input_tensor, past_label=start_token_tensor,
                                 future_input=future_input_tensor)

            y = y[:, self.past_length:, :].cpu().numpy()

        result = list(map(lambda x: output_scaler.inverse_transform(y[:, x, :]), list(range(future_length))))
        result_np = np.stack(result, axis=1)
        return result_np


if __name__ == "__main__":
    data = pd.read_csv('Debutanizer_Data.txt', index_col=None, header=0, sep='\s+')
    train_index = int(data.shape[0] * 0.6)
    valid_index = int(data.shape[0] * 0.7)

    SEED = 4096
    BATCH_SIZE = 32
    OUTPUT_DIM = 1
    TRANSITION_DIM = 4
    CONTROL_DIM = 4
    RNN_DIM = 4
    RNN_LAYERS = 2
    INPUT_DIM = data.shape[1] - 1
    PAST_LENGTH = 8
    FUTURE_LENGTH = 4
    CUDA_NUMBER = 4
    EPOCH = 100
    LEARNING_RATE = 0.005
    # DEVICE = torch.device(f'cuda:{CUDA_NUMBER}' if torch.cuda.is_available() else 'cpu')
    DEVICE = torch.device('cpu')



    """
     input_dim, output_dim, transition_dim, control_dim, rnn_dim, rnn_layers,
                 past_length, future_length, num_epochs, batch_size, learning_rate, device, seed=1024
    """
    """
    # 原先的代码
    mdl = DynamicVAEModelInit(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM, control_dim=CONTROL_DIM, rnn_dim=RNN_DIM,
                              rnn_layers=RNN_LAYERS, past_length=PAST_LENGTH, transition_dim=TRANSITION_DIM,
                              future_length=FUTURE_LENGTH, num_epochs=EPOCH,
                              batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE,
                              device=DEVICE, seed=SEED).fit(input_data=train_x, output_data=train_y,
                                                            valid_input=valid_x, valid_output=valid_y)

    prediction = DynamicVAEModelInit(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM,
                                             control_dim=CONTROL_DIM, rnn_dim=RNN_DIM,
                                             rnn_layers=RNN_LAYERS, past_length=PAST_LENGTH,
                                             transition_dim=TRANSITION_DIM,
                                             future_length=FUTURE_LENGTH, num_epochs=EPOCH,
                                             batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE,
                                             device=DEVICE, seed=SEED).predict(past_input=test_past_input,
                                                                               future_input=test_future_input,
                                                                               start_token=test_start_token)
    """

    seed_list = [256, 512, 1024, 2048, 4096]
    future_length_list = [2, 4, 8, 16]
    result_list = []
    try:
        os.makedirs("./future_length_experiment")
    except:
        pass

    for future_length in future_length_list:
        print(f"[Info] We are testing future length: {future_length}")



        for selected_seed in seed_list:
            train_x, train_y, valid_x, valid_y, test_x, test_y = obtain_train_valid_test(data=data,
                                                                                         train_index=train_index,
                                                                                         valid_index=valid_index,
                                                                                         past_length=PAST_LENGTH,
                                                                                         future_length=future_length,
                                                                                         output_dim=OUTPUT_DIM)
            mdl = DynamicVAEModelInit(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM, control_dim=CONTROL_DIM, rnn_dim=RNN_DIM,
                              rnn_layers=RNN_LAYERS, past_length=PAST_LENGTH, transition_dim=TRANSITION_DIM,
                              future_length=future_length, num_epochs=100,
                              batch_size=64, learning_rate=0.005,
                              device=DEVICE, seed=selected_seed).fit(input_data=train_x, output_data=train_y,
                                                            valid_input=valid_x, valid_output=valid_y)

            test_past_input, test_future_input, test_start_token, test_label = construct_test_tensor(test_x=test_x,
                                                                                             test_y=test_y,
                                                                                             past_length=PAST_LENGTH,
                                                                                             future_length=future_length)

            prediction = DynamicVAEModelInit(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM,
                                     control_dim=CONTROL_DIM, rnn_dim=RNN_DIM,
                                     rnn_layers=RNN_LAYERS, past_length=PAST_LENGTH,
                                     transition_dim=TRANSITION_DIM,
                                     future_length=future_length, num_epochs=100,
                                     batch_size=BATCH_SIZE, learning_rate=0.005,
                                     device=DEVICE, seed=selected_seed).predict(past_input=test_past_input,
                                                                       future_input=test_future_input,
                                                                       start_token=test_start_token)

            display_window_results(pred=prediction, true=test_label, seed=selected_seed)

            mean_rmse, mean_mae, mean_mape = obtain_window_results(pred=prediction, true=test_label)

            result_list.append(pd.DataFrame(
                {"model": "OC-NDPLVM", "seed": selected_seed, "H": future_length, "rmse": mean_rmse, "mae": mean_mae,
                 "mape": mean_mape}, index=[0]))
    result_dataframe = pd.concat(result_list, axis=0)
    result_dataframe.to_csv("./future_length_experiment/OC-NDPLVM.csv", index=None)









