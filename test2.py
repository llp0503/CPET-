

import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.preprocessing import MinMaxScaler

def slide_and_cut(X, Y, output_pid=False):
    """
    如果数据已经是正确的形状，这个函数可以简化为直接返回输入数据。
    如果需要滑动窗口操作，可以在这里实现。
    """
    if output_pid:
        return X, Y, np.arange(len(X))
    else:
        return X, Y

def scale_data(X_train, X_val, X_test):
    """
    对每个通道的数据进行 MinMaxScaler 归一化。
    """
    # 获取原始数据的形状
    n_samples_train, n_channels_train, n_time_steps_train = X_train.shape
    n_samples_val, n_channels_val, n_time_steps_val = X_val.shape
    n_samples_test, n_channels_test, n_time_steps_test = X_test.shape

    # 初始化 MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))

    # 对每个通道分别归一化
    X_train_scaled = np.zeros_like(X_train)
    X_val_scaled = np.zeros_like(X_val)
    X_test_scaled = np.zeros_like(X_test)

    for c in range(n_channels_train):
        # 展平每个通道的数据
        X_train_2d = X_train[:, c, :].reshape(-1, n_time_steps_train)
        X_val_2d = X_val[:, c, :].reshape(-1, n_time_steps_val)
        X_test_2d = X_test[:, c, :].reshape(-1, n_time_steps_test)

        # 对训练集进行拟合和转换
        X_train_scaled_2d = scaler.fit_transform(X_train_2d)

        # 对验证集和测试集仅进行转换
        X_val_scaled_2d = scaler.transform(X_val_2d)
        X_test_scaled_2d = scaler.transform(X_test_2d)

        # 将归一化后的数据恢复为三维数组
        X_train_scaled[:, c, :] = X_train_scaled_2d.reshape(n_samples_train, n_time_steps_train)
        X_val_scaled[:, c, :] = X_val_scaled_2d.reshape(n_samples_val, n_time_steps_val)
        X_test_scaled[:, c, :] = X_test_scaled_2d.reshape(n_samples_test, n_time_steps_test)

    return X_train_scaled, X_val_scaled, X_test_scaled

def read_data_physionet_4_with_val():
    with open('/data/llp/resnet1d-master/merged_data_with_labels.pkl', 'rb') as fin:
        res = pickle.load(fin)

    # 将数据加载为 NumPy 数组
    all_data = np.array([item[0] for item in res], dtype=np.float64)
    all_label = np.array([item[1] for item in res], dtype=int)

    # 确保 all_data 的形状为 (n_samples, n_channels, n_time_steps)
    print("Original data shape:", all_data.shape)
    if all_data.shape[1] != 30:  # 如果通道数不在第二个维度，进行转置
        all_data = np.transpose(all_data, (0, 2, 1))
        print("Transposed data shape:", all_data.shape)

    # 检查数据形状是否正确
    if all_data.shape[1] != 30:
        raise ValueError(f"Expected data shape to have 30 channels, but got shape {all_data.shape}")

    # split train val test
    X_train, X_test, Y_train, Y_test = train_test_split(all_data, all_label, test_size=0.2, random_state=0)
    X_val, X_test, Y_val, Y_test = train_test_split(X_test, Y_test, test_size=0.5, random_state=0)

    # slide and cut
    print('before: ')
    print(Counter(Y_train), Counter(Y_val), Counter(Y_test))
    X_train, Y_train, pid_train = slide_and_cut(X_train, Y_train, output_pid=True)
    X_val, Y_val, pid_val = slide_and_cut(X_val, Y_val, output_pid=True)
    X_test, Y_test, pid_test = slide_and_cut(X_test, Y_test, output_pid=True)
    print('after: ')
    print(Counter(Y_train), Counter(Y_val), Counter(Y_test))

    # shuffle train
    shuffle_pid = np.random.permutation(Y_train.shape[0])
    X_train = X_train[shuffle_pid]
    Y_train = Y_train[shuffle_pid]

    # Scale data
    X_train, X_val, X_test = scale_data(X_train, X_val, X_test)

    # 打印数据形状和统计信息
    print("Scaled X_train shape:", X_train.shape)
    print("Scaled X_val shape:", X_val.shape)
    print("Scaled X_test shape:", X_test.shape)
    print("X_train min/max:", X_train.min(), X_train.max())
    print("X_val min/max:", X_val.min(), X_val.max())
    print("X_test min/max:", X_test.min(), X_test.max())

    return X_train, X_val, X_test, Y_train, Y_val, Y_test, pid_train, pid_val, pid_test

# 调用函数
X_train, X_val, X_test, Y_train, Y_val, Y_test, pid_train, pid_val, pid_test = read_data_physionet_4_with_val()

