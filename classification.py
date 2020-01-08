import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import firstHarmonicsAnalysis as fh
from scipy import stats
from scipy.stats import skew
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import kurtosis
from scipy.stats import iqr

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
    #still need to consider which features to extract from heart rate
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
    #anglegyro = []
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

    return [
            np.array(mean_features),
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
            np.array(yaw).reshape(-1,1)
            ]
# feature extraction & labeling"""
def activity_feature_extraction(path, starting_idx_rank, ending_idx_rank, sampling_size):
    #path: path to a certain activity
    data_file_list = os.listdir(path)

    for idx, file_name in enumerate(data_file_list):
        data_file_path = os.path.join(path, file_name)
        # print(data_file_path + ' Time:' + time.ctime())
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
            label = np.zeros((num_row, 1))
            label[:, :] = idx
            labels = label

        else:
            activity_features = np.concatenate((activity_features, concatenated_features), axis = 0)
            num_row, _ = features[0].shape
            label = np.zeros((num_row, 1))
            label[:, :] = idx
            labels = np.concatenate((labels, label), axis = 0)

    return activity_features, labels

walk_data_path = 'processed_data/walk/'
starting_idx_rank = 0.05# Cuting off the First 5% of Data
ending_idx_rank = 0.95# Cutting off the End 5% of Data
sampling_size = 100#maybe you should use different sampling size according to the type of activities

if os.path.exists('walk_features.npy'):
    walk_features = np.load('walk_features.npy')
    walk_labels = np.load('walk_labels.npy')

else:
    print('Feature Extraction Start!')
    walk_features, walk_labels = activity_feature_extraction(walk_data_path, starting_idx_rank, ending_idx_rank, sampling_size)
    np.save('walk_features.npy', walk_features)
    np.save('walk_labels.npy', walk_labels)
    print('Feature Extraction End!')

num_iteration = 10
num_estimators_unit = 10
num_estimators_range = 19

running_time = []

for range_idx in range(1, num_estimators_range + 1):

    num_estimators = num_estimators_unit*range_idx
    num_running_time = []

    for _ in range(num_iteration):

        train_x_data, train_y_data, test_x_data, test_y_data = train_test_divide(walk_features, walk_labels)
        num_participants = len(np.unique(walk_labels))

        train_y_data = train_y_data.ravel()
        test_y_data = test_y_data.ravel()

        print('Train Start!')
        rfc = RandomForestClassifier(n_estimators=num_estimators, n_jobs = -1)
        train_start_time = time.time()
        rfc.fit(train_x_data, train_y_data)
        train_end_time = time.time()

        time4train = train_end_time - train_start_time

        importances = rfc.feature_importances_
        std = np.std([tree.feature_importances_ for tree in rfc.estimators_], axis=0)
        indices = np.argsort(importances)[::-1]
        print('Train End!')

        # Print the feature ranking
        print("Feature ranking:")
        for f in range(train_x_data.shape[1]):
            print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

        train_score = rfc.score(train_x_data, train_y_data)
        test_score = rfc.score(test_x_data, test_y_data)
        print("rfc train score: ", train_score)
        print("rfc test score: ", test_score)

        p = rfc.predict(test_x_data)
        tem = f1_score(test_y_data, p, average=None)

        prediction_start_time = time.time()
        predict_one = rfc.predict(test_x_data[0].reshape(1, -1))
        prediction_end_time = time.time()

        time4prediction = prediction_end_time - prediction_start_time

        f1_score_result = f1_score(test_y_data, p, average=None).mean()
        print("dt F1 score: ", f1_score_result)

        confusion_matrix = np.zeros((num_participants, num_participants))

        for row in range(num_participants):

            prediction_user_row = p[p == row]
            comparison_target = test_y_data[p == row]

            num_prediction_user_row = len(prediction_user_row)

            score_row = []

            for col in range(num_participants):

                score = len(comparison_target[comparison_target == col])
                '''/ num_prediction_user_row'''
                score_row.append(score)
            confusion_matrix[row, :] = np.array(score_row)
        print(confusion_matrix)

        num_running_time.append([time4train, time4prediction, test_score])

    running_time.append(num_running_time)

np.save('running_time_detail.npy', np.array(running_time))





































#end
