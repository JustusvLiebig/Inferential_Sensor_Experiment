from VBSFAInferentialSensor import VBSFASoftSensor
import math
import scipy
import scipy.io as sio
import numpy as np
import pandas as pd
import copy


def streaming_generator(dataset, mini_batch):
    dataset_size = dataset.shape[0]
    index_list = list(map(lambda x: mini_batch * x, range(math.floor(dataset_size / mini_batch)))) + [dataset_size]

    start_index = index_list[:-1]
    end_index = index_list[1:]

    data_list = list(map(lambda x, y: dataset[x:y, :], start_index, end_index))
    return data_list




def StreamingVariationalBayes(dataset, x_dimension, y_dimension, factor_dimension, max_iter, mini_batch, use_stream):

    data_list = streaming_generator(dataset=dataset, mini_batch=mini_batch)

    # x_dimension, y_dimension, factor_dimension, loop_number

    soft_sensor_model = VBSFASoftSensor(x_dimension=x_dimension, y_dimension=y_dimension,
                                        factor_dimension=factor_dimension, loop_number=max_iter)

    no_update_model = VBSFASoftSensor(x_dimension=x_dimension, y_dimension=y_dimension,
                                      factor_dimension=factor_dimension, loop_number=max_iter)

    # fit the first mini batch dataset
    soft_sensor_model.fit(data=data_list[0])
    no_update_model.fit(data=data_list[0])

    label_list = list()
    predict_list = list()


    for i in range(1, len(data_list)):
        # get the stream dataset
        stream_dataset = data_list[i]

        # get the input data and output data
        test_data_x = stream_dataset[:, :x_dimension]
        predict_test = soft_sensor_model.predict(input_data=test_data_x)

        label_list.append(stream_dataset[:, x_dimension:].reshape((-1, y_dimension)))
        predict_list.append(predict_test)

        if i != (len(data_list)-1):
            #soft_sensor_model = VBSFASoftSensor(x_dimension=x_dimension, y_dimension=y_dimension, factor_dimension=factor_dimension, loop_number=max_iter)
            if use_stream:
                # last mini batch global params
                old_posterior_hyperparameters = copy.deepcopy(soft_sensor_model.VBSFAClass.posterior_hyperparameters)
                old_posterior_hyperparameters.s_mean = copy.deepcopy(
                    no_update_model.VBSFAClass.prior_hyperparameters.s_mean)
                old_posterior_hyperparameters.s_sigma = copy.deepcopy(
                    no_update_model.VBSFAClass.prior_hyperparameters.s_sigma)

                # soft_sensor_model.VBSFAClass.posterior_hyperparameters = old_posterior_hyperparameters
                soft_sensor_model = VBSFASoftSensor(x_dimension=x_dimension,
                                                    y_dimension=y_dimension,
                                                    factor_dimension=factor_dimension,
                                                    loop_number=max_iter).fit_stream(data=stream_dataset,
                                                                                     past_posterior=old_posterior_hyperparameters)


            else:
                soft_sensor_model = VBSFASoftSensor(x_dimension=x_dimension,
                                                    y_dimension=y_dimension,
                                                    factor_dimension=factor_dimension,
                                                    loop_number=max_iter).fit(data=stream_dataset)

    return {'prediction_value': np.concatenate(predict_list, axis=0), 'real_value': np.concatenate(label_list, axis=0)}
  
  
  

if __name__ == '__main__':


    original_data = sio.loadmat('./data/U06_data.mat')
    numerical_data = np.array(pd.DataFrame(original_data.get('U06_data')))[53000:53000 + 95000, :]



    # for i in range(len(output_index)):
    #     print(f'The index value is {i+1}, the output shape is {output_index[i].shape}')

    x_dimension = 10
    y_dimension = 1
    factor_dimension = 3


    max_iter = 30
    mini_batch = 25
    use_stream = True
    import datetime
    start_time = datetime.datetime.now()
    # x_dimension, y_dimension, factor_dimension, max_iter, mini_batch, use_stream
    result_dict = StreamingVariationalBayes(dataset=numerical_data[:, :], x_dimension=x_dimension,
                                            y_dimension=y_dimension, factor_dimension=factor_dimension,
                                            max_iter=max_iter, mini_batch=mini_batch, use_stream=use_stream)
    end_time = datetime.datetime.now()

    # print(result_dict.get('prediction_value').shape)
    # print(result_dict.get('real_value').shape)

    predict_test = result_dict.get('prediction_value')
    real_test = result_dict.get('real_value')

    from sklearn.metrics import r2_score, mean_squared_error
    test_rmse = np.sqrt(mean_squared_error(predict_test, real_test))
    test_r2 = r2_score(predict_test, real_test)

    print(f'[Info] The rmse is {test_rmse}, r2 is {test_r2}, time cost {(end_time - start_time).seconds}')

    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(list(range(real_test.shape[0])), real_test.tolist(), color='b', label='real label')
    plt.plot(list(range(predict_test.shape[0])), predict_test.tolist(), color='r', label='predict value')
    plt.legend()
    plt.title("Streaming Factor Analysis at {mini_batch} \n RMSE: {rmse}, R2 {r2}".format(mini_batch=mini_batch, rmse=np.round(test_rmse, 6), r2=np.round(test_r2, 6)))
    plt.show()

