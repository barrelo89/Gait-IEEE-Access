import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.stats import skew
from sklearn.utils import shuffle

def border_line_finding(data, activity_list):

    result = []

    for row_idx, row in enumerate(data.values):
        for col in row:
            if col in activity_list:
                result.append(row_idx)

    return sorted(list(set(result)))
#csv file name change
def name_change(base_path, processed_data_path):

    if not os.path.exists(processed_data_path):
        os.makedirs(processed_data_path)

    file_list = os.listdir(base_path)
    columns = ['accx', 'accy', 'accz', 'gyrox', 'gyroy', 'gyroz', 'heart rate']

    for file_name in file_list:
        file_path = os.path.join(base_path, file_name)
        data = pd.read_csv(file_path, names=columns)
        user_info = data.iloc[0, :]
        add_on = user_info[0] + '_' + user_info[2] + '_' + user_info[3] + '.csv'
        data = data.iloc[1:]
        data.to_csv(processed_data_path + add_on)
#csv file divide according to the activies
def activity_divide(processed_data_path, activity_list, walk_data_path, run_data_path, open_data_path, type_data_path):
    file_list = os.listdir(processed_data_path)

    if not os.path.exists(walk_data_path):
        os.makedirs(walk_data_path)

    if not os.path.exists(run_data_path):
        os.makedirs(run_data_path)

    if not os.path.exists(open_data_path):
        os.makedirs(open_data_path)

    if not os.path.exists(type_data_path):
        os.makedirs(type_data_path)

    for file_name in file_list:
        file_path = os.path.join(processed_data_path, file_name)
        data = pd.read_csv(file_path, delimiter = ',').iloc[:, 1:]
        activity_idx = border_line_finding(data, activity_list)

        walk_data = data.iloc[activity_idx[0]:activity_idx[1], :]
        walk_data.to_csv(walk_data_path + file_name)
        run_data = data.iloc[activity_idx[1]:activity_idx[2], :]
        run_data.to_csv(run_data_path + file_name)
        open_data = data.iloc[activity_idx[2]:activity_idx[3], :]
        open_data.to_csv(open_data_path + file_name)
        type_data = data.iloc[activity_idx[3]:, :]
        type_data.to_csv(type_data_path + file_name)

base_path = 'original_data'

activity_list = ['walk', 'run', 'open', 'type']

processed_data_path = 'processed_data/name_change/'

walk_data_path = 'processed_data/walk/'
run_data_path = 'processed_data/run/'
open_data_path = 'processed_data/open/'
type_data_path = 'processed_data/type/'
starting_idx_rank = 0.05
ending_idx_rank = 0.95
sampling_size = 100 #maybe you should use different sampling size according to the type of activities

name_change(base_path, processed_data_path)
activity_divide(processed_data_path, activity_list, walk_data_path, run_data_path, open_data_path, type_data_path)















#end
