import os
import pandas as pd
import matplotlib.pyplot as plt

path = "processed_data/"
activity_name = ['walk']#['open', 'run', 'type', 'walk']
col = ['accx','accy','accz','gyrox','gyroy','gyroz']#, 'HR'

img_save_path = 'img'
type_path = activity_name[0]
target_path = os.path.join(img_save_path, type_path)

if not os.path.exists(target_path):
    os.makedirs(target_path)

for activity in activity_name:
    print(activity)
    folder_path = os.path.join(path, activity)
    file_list = os.listdir(folder_path)

    for file_name in file_list:
        print(file_name)
        file_path = os.path.join(folder_path, file_name)
        data = pd.read_csv(file_path, names=col, delimiter=',').values[2:, :]
        data = data.astype(float)

        #accx = data[:, 0]
        #accy = data[:, 1]
        #accz = data[:, 2]
        #gyrox = data[:, 3]
        #gyroy = data[:, 4]
        #gyroz = data[:, 5]
        #heart_rate = data[:, 6]

        figure, axes = plt.subplots(6, sharex = True)
        data_name = ['Acc X', 'Acc Y', 'Acc Z', 'Gyro X', 'Gyro Y', 'Gyro Z']#, 'Heart Rate'

        for idx, ax in enumerate(axes):
            ax.plot(data[500:20000, idx])#[500:5000, idx]
            ax.set_title(data_name[idx])
            ax.set_yticks([int(data[:, idx].min()), int(data[:, idx].max())])
            ax.axvline(100, color = 'k', linestyle = 'dotted', label = 'Period')
            ax.axvline(200, color = 'k', linestyle = 'dotted')
            ax.axvline(300, color = 'k', linestyle = 'dotted')
            ax.axvline(400, color = 'k', linestyle = 'dotted')
            ax.axvline(500, color = 'k', linestyle = 'dotted')
            ax.axvline(600, color = 'k', linestyle = 'dotted')
            ax.axvline(700, color = 'k', linestyle = 'dotted')
            ax.axvline(800, color = 'k', linestyle = 'dotted')

        plt.tight_layout()
        plt.savefig(os.path.join(target_path, file_name.split('.')[0] + '.pdf'))

        plt.close()




#end
