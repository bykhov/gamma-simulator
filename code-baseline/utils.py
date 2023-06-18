from scipy.io import savemat, loadmat
import numpy as np
import os
from tensorflow.keras.utils import to_categorical

# %% Data loading (binary classification)
def load_train_data(data_directory='D:/Google Drive/Research/4 Tom/database/', num_classes=6):
    """Load data from .mat file
    return: X_train, Y_train, sample_length
    """
    mat = loadmat(data_directory + 'train.mat')
    X_train = mat['X_train_all'] / 1000. - 0.5  # normalization
    # Y_train = to_categorical(mat['Y_train_all'], num_classes=num_classes)
    Y_train = mat['Y_train_all']
    sample_length = X_train.shape[-1]  # frame length
    return X_train, Y_train, sample_length


def load_test_data(k: int, data_directory='D:/Google Drive/Research/4 Tom/database/', num_classes=6):
    """Load data from .mat file
    k: 10, 20, 30, 40, 50, 60
    returns: X_test, Y_test, Y_test_i
    """
    mat = loadmat(data_directory + 'data_test_' + str(k) + '.mat')
    X_test = mat['X_test'] / 1000. - 0.5
    Y_test = mat['Y_test']
    # Y_test = to_categorical(Y_test_i, num_classes=num_classes)
    # Y_test = mat['Y_test'].astype(bool).astype(np.uint8)
    return X_test, Y_test


def save_test_data(k: int, Y_pred, data_directory='D:/Google Drive/Research/4 Tom/database/'):
    """Save test data to .mat file"""
    file_name = os.getcwd() + '\\data_pred_' + str(k) + '.mat'
    savemat(file_name, {'Y_pred': Y_pred})

