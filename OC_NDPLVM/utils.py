import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import os
import random
import torch
import joblib


def obtain_train_valid_test(data, train_index, valid_index, past_length, future_length, output_dim):
    train_x = np.array(data.iloc[:train_index, :-1])
    train_y = np.array(data.iloc[:train_index, -1]).reshape((-1, 1))

    valid_x = np.array(data.iloc[train_index - (past_length + future_length):valid_index, :-1])
    valid_y = np.array(data.iloc[train_index - (past_length + future_length):valid_index, -1]).reshape((-1, output_dim))

    test_x = np.array(data.iloc[valid_index - (past_length + future_length):, :-1])
    test_y = np.array(data.iloc[valid_index - (past_length + future_length):, -1]).reshape((-1, output_dim))

    return train_x, train_y, valid_x, valid_y, test_x, test_y



def percentage_error(actual, predicted):

    def _judge_value(actual, prediction, mean):
        if actual !=0:
            return (actual - prediction) / actual
        else:
            return prediction / mean

    mean_value = np.mean(actual)
    iterator_list = list(range(actual.shape[0]))
    result_list = list(map(lambda x: _judge_value(actual=actual[x, :],
                                                  prediction=predicted[x, :],
                                                  mean=mean_value), iterator_list))

    res = np.array(result_list)
    return res


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs(percentage_error(np.asarray(y_true), np.asarray(y_pred)))) * 100



def display_results(pred, true, seed):
    test_rmse = np.sqrt(mean_squared_error(y_pred=pred, y_true=true))
    test_r2 = r2_score(y_pred=pred, y_true=true)
    test_mae = mean_absolute_error(y_pred=pred, y_true=true)
    test_mape = mean_absolute_percentage_error(y_true=true, y_pred=pred)
    print('seed = ', seed,
          ', test_rmse = ', round(test_rmse, 6),
          ', r2 = ', round(test_r2, 6),
          ', mape = ', round(test_mape, 6),
          ', mae = ', round(test_mae, 6))


def setup_seed(seed):
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def create_file(model_name):
    try:
        os.makedirs('./' + model_name)
    except:
        pass

class FasterDataset(Dataset):
    def __init__(self, data, label):
        self.data, self.label = data, label

    # Get item
    def __getitem__(self, index):
        return self.data[index, :, :], self.label[index, :, :]

    # Get length
    def __len__(self):
        return self.data.shape[0]


def display_window_results(pred, true, seed):
    batch_size = pred.shape[0]
    test_rmse_list = list(map(lambda x: np.sqrt(mean_squared_error(y_pred=pred[x, :, :].reshape((-1, 1)), y_true=true[x, :, :].reshape((-1, 1)))), list(range(batch_size))))
    test_r2_list = list(map(lambda x: r2_score(y_pred=pred[x, :, :].reshape((-1, 1)), y_true=true[x, :, :].reshape((-1, 1))), list(range(batch_size))))
    test_mae_list = list(map(lambda x: mean_absolute_error(y_pred=pred[x, :, :].reshape((-1, 1)), y_true=true[x, :, :].reshape((-1, 1))), list(range(batch_size))))
    test_mape_list = list(map(lambda x: mean_absolute_percentage_error(y_pred=pred[x, :, :].reshape((-1, 1)), y_true=true[x, :, :].reshape((-1, 1))), list(range(batch_size))))

    mean_rmse = np.mean(test_rmse_list)
    mean_r2 = np.mean(test_r2_list)
    mean_mae = np.mean(test_mae_list)
    mean_mape = np.mean(test_mape_list)

    print('This is for window results!')
    print('seed = ', seed,
          ', test_rmse = ', round(mean_rmse, 6),
          ', r2 = ', round(mean_r2, 6),
          ', mape = ', round(mean_mae, 6),
          ', mae = ', round(mean_mape, 6))


def display_save_window_results(pred, true, seed, hyper_learning_rate, hyper_batch_size, hyper_noise_level):
    batch_size = pred.shape[0]
    test_rmse_list = list(map(lambda x: np.sqrt(mean_squared_error(y_pred=pred[x, :, :].reshape((-1, 1)), y_true=true[x, :, :].reshape((-1, 1)))), list(range(batch_size))))
    test_r2_list = list(map(lambda x: r2_score(y_pred=pred[x, :, :].reshape((-1, 1)), y_true=true[x, :, :].reshape((-1, 1))), list(range(batch_size))))
    test_mae_list = list(map(lambda x: mean_absolute_error(y_pred=pred[x, :, :].reshape((-1, 1)), y_true=true[x, :, :].reshape((-1, 1))), list(range(batch_size))))
    test_mape_list = list(map(lambda x: mean_absolute_percentage_error(y_pred=pred[x, :, :].reshape((-1, 1)), y_true=true[x, :, :].reshape((-1, 1))), list(range(batch_size))))

    mean_rmse = np.mean(test_rmse_list)
    mean_r2 = np.mean(test_r2_list)
    mean_mae = np.mean(test_mae_list)
    mean_mape = np.mean(test_mape_list)

    result_dict = {'seed': seed, 'learning_rate': hyper_learning_rate, 'batch_size': hyper_batch_size,
                   'noise_level': hyper_noise_level, 'test_rmse': mean_rmse, 'r2': mean_r2, 'mape': mean_mae, 'mae': mean_mape}
    return result_dict


def obtain_window_results(pred, true):
    batch_size = pred.shape[0]
    test_rmse_list = list(map(lambda x: np.sqrt(mean_squared_error(y_pred=pred[x, :, :].reshape((-1, 1)), y_true=true[x, :, :].reshape((-1, 1)))), list(range(batch_size))))
    test_r2_list = list(map(lambda x: r2_score(y_pred=pred[x, :, :].reshape((-1, 1)), y_true=true[x, :, :].reshape((-1, 1))), list(range(batch_size))))
    test_mae_list = list(map(lambda x: mean_absolute_error(y_pred=pred[x, :, :].reshape((-1, 1)), y_true=true[x, :, :].reshape((-1, 1))), list(range(batch_size))))
    test_mape_list = list(map(lambda x: mean_absolute_percentage_error(y_pred=pred[x, :, :].reshape((-1, 1)), y_true=true[x, :, :].reshape((-1, 1))), list(range(batch_size))))

    mean_rmse = np.mean(test_rmse_list)
    mean_r2 = np.mean(test_r2_list)
    mean_mae = np.mean(test_mae_list)
    mean_mape = np.mean(test_mape_list)

    return mean_rmse, mean_mae, mean_mape





