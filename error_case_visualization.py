import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import firstHarmonicsAnalysis as fh
from scipy.stats import kurtosis
from scipy.stats import iqr
from scipy.stats import skew
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

# train test divide
def train_test_divide(x_data, y_data, ratio = 0.7):#mean_data, std_data, skew_data, median_data, amp_acc, amp_lin
    num_sample, num_feature = x_data.shape
    x_data, y_data = shuffle(x_data, y_data)

    train_size = int(num_sample*ratio)
    train_x_data = x_data[:train_size, :]
    test_x_data = x_data[train_size:, :]

    train_y_data = y_data[:train_size]
    test_y_data = y_data[train_size:]

    return train_x_data, train_y_data, test_x_data, test_y_data

#feature extraction: mean(7), std(7), skewness(7), median(7), sqrt acc(1), sqrt gyro(1)
def feature_extraction(data, starting_idx_rank, ending_idx_rank, sampling_size, weight, bmi):

    #first filter the given data
    row, col = data.shape
    data = fh.data_removal_trial(data)
    #data= fh.datameanremoval(data)
    data = fh.filter_data(data)
    data = data[int(row*starting_idx_rank) : int(row*ending_idx_rank), :]

    new_row, new_col = data.shape
    num_features = int(new_row / sampling_size)

    #we consider mean, std, skewness, median, magnitude for acc and gyro
    mean_features = []
    std_features = []
    skew_features = []
    max_data = []
    avg_var = []
    kurt_features = []
    median_features = []
    sqrt_acc_features = []
    sqrt_gx_features = []
    r_cor1 = []
    r_cor2 = []
    r_cor3 = []

    r_cor4 = []
    r_cor5 = []
    r_cor6 = []
    r_cor7 = []
    r_cor8 = []
    r_cor9 = []

    cf = []
    pitch = []
    roll = []
    yaw = []
    iqr_data = []
    angleacc = []
    angler = []

    for idx in range(num_features):
        cor = data[idx * sampling_size:(idx + 1) * sampling_size, :]
        angleacc.append(fh.anglefinderacc(cor, bmi))
        angler.append(fh.anglefinder(cor))
        r_cor1.append(fh.correlation_feature(cor[:, 0], cor[:, 3]))
        r_cor2.append(fh.correlation_feature(cor[:, 1], cor[:, 4]))
        r_cor3.append(fh.correlation_feature(cor[:, 2], cor[:, 5]))

        r_cor4.append(fh.correlation_feature(cor[:, 0], cor[:, 1]))
        r_cor5.append(fh.correlation_feature(cor[:, 1], cor[:, 2]))
        r_cor6.append(fh.correlation_feature(cor[:, 2], cor[:, 1]))

        r_cor7.append(fh.correlation_feature(cor[:, 3], cor[:, 4]))
        r_cor8.append(fh.correlation_feature(cor[:, 4], cor[:, 5]))
        r_cor9.append(fh.correlation_feature(cor[:, 5], cor[:, 3]))


        pitch.append(fh.pitch_calculation(cor))
        roll.append(fh.roll_calculation(cor))
        yaw.append(fh.yaw_calculation(cor))

        mean_features.append(np.mean(data[idx*sampling_size:(idx+1)*sampling_size, :], axis=0))

        _temp = (np.mean(data[idx * sampling_size:(idx + 1) * sampling_size, :], axis=0))

        kurt_features.append(kurtosis(data[idx*sampling_size:(idx+1)*sampling_size, :], axis=0))

        std_features.append(np.std(data[idx*sampling_size:(idx+1)*sampling_size, :], 0))

        skew_features.append(skew(data[idx*sampling_size:(idx+1)*sampling_size, :], axis=0, bias=True))
        median_features.append(np.median(data[idx*sampling_size:(idx+1)*sampling_size, :], axis = 0))

        square_matrix = np.square(data[idx*sampling_size:(idx+1)*sampling_size, :])
        acc_square_matrix = square_matrix[:, [0, 1, 2]]
        gx_square_matrix = square_matrix[:,[3, 4, 5]]
        acc_square_matrix = np.mean(np.sum(acc_square_matrix, axis = 1))
        gx_square_matrix = np.mean(np.sum(gx_square_matrix,axis=1))
        sqrt_acc = (np.sqrt(acc_square_matrix)) * weight
        sqrt_gx = (np.sqrt(gx_square_matrix))
        sqrt_acc_features.append(sqrt_acc)
        sqrt_gx_features.append(sqrt_gx)

    new_mean_feature = [[x[0], x[3]] for x in mean_features]
    new_std_feature = [[x[0], x[3]] for x in std_features]
    new_skew_feature = [[x[0], x[3]] for x in skew_features]
    new_kurt_feature = [[x[0], x[3]] for x in kurt_features]
    new_cf_feature = [[x[0], x[3]] for x in cf]

    return [np.array(mean_features),
            np.array(std_features),
            np.array(skew_features),
            np.array(kurt_features),
            np.array(sqrt_acc_features).reshape(-1, 1),
            np.array(pitch).reshape(-1, 1),
            np.array(roll).reshape(-1, 1),

            np.array(r_cor1).reshape(-1, 1),
            np.array(r_cor2).reshape(-1, 1),
            np.array(r_cor3).reshape(-1, 1),

            np.array(r_cor4).reshape(-1, 1),
            np.array(r_cor5).reshape(-1, 1),
            np.array(r_cor6).reshape(-1, 1),

            np.array(r_cor7).reshape(-1, 1),
            np.array(r_cor8).reshape(-1, 1),
            np.array(r_cor9).reshape(-1, 1),
            np.array(yaw).reshape(-1,1)]

# feature extraction & labeling"""
def activity_feature_extraction(path, starting_idx_rank, ending_idx_rank, sampling_size):
    #path: path to a certain activity
    data_file_list = os.listdir(path)
    print(data_file_list)
    for idx, file_name in enumerate(data_file_list):

        data_file_path = os.path.join(path, file_name)
        str_one = file_name
        k = (str_one.split("_"))
        weigh = k[2]
        height = k[1]
        height = float(height)
        height = height/100
        weigh = weigh.split(".")
        weight = float(weigh[0])
        bmi = weight/(height* height)
        data = pd.read_csv(data_file_path, delimiter = ',').values[1:, 1:]
        data = data.astype(float)
        features = feature_extraction(data, starting_idx_rank, ending_idx_rank, sampling_size, weight, bmi)

        for feature_idx in range(len(features)):
            if feature_idx == 0:
                concatenated_features = features[feature_idx]

            else:
                concatenated_features = np.concatenate((concatenated_features, features[feature_idx]), axis = 1)

        if idx == 0:
            activity_features = concatenated_features
            num_row, _ = features[0].shape
            label = np.zeros((num_row, 2))
            label[:, 0] = idx
            label[:, 1] = range(num_row)
            labels = label

        else:
            activity_features = np.concatenate((activity_features, concatenated_features), axis = 0)
            num_row, _ = features[0].shape
            label = np.zeros((num_row, 2))
            label[:, 0] = idx
            label[:, 1] = range(num_row)
            labels = np.concatenate((labels, label), axis = 0)

    return activity_features, labels


#load feature, label data
walk_features = np.load('walk_features_vis.npy')
walk_labels = np.load('walk_labels_vis.npy')

#set the # of estimators
num_estimators = 500

#divide train and test data (labels)
train_x_data, train_y_data, test_x_data, test_y_data = train_test_divide(walk_features, walk_labels)
train_y_data = train_y_data[:, 0]
test_y_data, test_y_data_idx = test_y_data[:, 0], test_y_data[:, 1]

#number of unique users in our system
num_participants = len(np.unique(walk_labels))

train_y_data = train_y_data.ravel()
test_y_data = test_y_data.ravel()

print('Train Start!')
rfc = RandomForestClassifier(n_estimators=num_estimators, n_jobs = -1)
rfc.fit(train_x_data, train_y_data)
print('Train End!')

train_score = rfc.score(train_x_data, train_y_data)
test_score = rfc.score(test_x_data, test_y_data)
print("rfc train score: ", train_score)
print("rfc test score: ", test_score)

p = rfc.predict(test_x_data)

user_name_list = test_y_data[np.where(p != test_y_data)[0]]
user_data_idx = test_y_data_idx[np.where(p != test_y_data)[0]]

img_save_path = 'img/error'

if not os.path.exists(img_save_path):
    os.makedirs(img_save_path)

walk_data_path = 'processed_data/walk'
starting_idx_rank = 0.05# Cuting off the First 5% of Data
ending_idx_rank = 0.95# Cutting off the End 5% of Data
sampling_size = 100#maybe you should use different sampling size according to the type of activities

data_file_list = os.listdir(walk_data_path)

for user_name, data_idx in zip(user_name_list, user_data_idx):

    if data_idx >= 3:

        file_name = data_file_list[int(user_name)]

        data = pd.read_csv(os.path.join(walk_data_path, file_name), delimiter = ',').values[1:, 1:]
        data = data.astype(float)
        row, col = data.shape

        data = fh.data_removal_trial(data)
        data = fh.filter_data(data)
        data = data[int(row*starting_idx_rank) : int(row*ending_idx_rank), :]

        figure, axes = plt.subplots(6, sharex = True)
        data_name = ['Acc X', 'Acc Y', 'Acc Z', 'Gyro X', 'Gyro Y', 'Gyro Z']#, 'Heart Rate'

        for idx, ax in enumerate(axes):
            new_data = data[int((int(data_idx)-2.5))*sampling_size:int((int(data_idx)+2.5))*sampling_size, idx]
            ax.plot(new_data)
            ax.set_title(data_name[idx])
            ax.set_yticks([int(new_data.min()), int(new_data.max())])
            ax.axvline(100, color = 'k', linestyle = 'dotted', label = 'Period')
            ax.axvline(200, color = 'k', linestyle = 'dotted')
            ax.axvline(300, color = 'k', linestyle = 'dotted')
            ax.axvline(400, color = 'k', linestyle = 'dotted')

        plt.tight_layout()
        plt.savefig(os.path.join(img_save_path, file_name.split('.')[0] + '_' + str(data_idx) + '.pdf'))
        plt.savefig(os.path.join(img_save_path, file_name.split('.')[0] + '_' + str(data_idx) + '.png'))
        plt.close()









































#end
