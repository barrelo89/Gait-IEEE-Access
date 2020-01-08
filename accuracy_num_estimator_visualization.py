import numpy as np
import matplotlib.pyplot as plt

#num_estimators: 200 ~ 3000 in an increment of 200
#num_iteration = 10
#training time, test time, accuracy
def visualization_macro():

    data = np.load('running_time_new.npy').mean(axis = 1)
    num_estimators_unit = 200

    training_time = data[:, 0]
    test_time = data[:, 1]
    accuracy = data[:, 2]

    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    x_range = range(1*num_estimators_unit, len(training_time)*num_estimators_unit+1, num_estimators_unit)

    plot_1 = ax1.plot(x_range, training_time, label = 'Training Time', c = 'g')
    plot_2 = ax1.plot(x_range, test_time, label = 'Prediction Time', c = 'b')

    ax2 = ax1.twinx()
    plot_3 = ax2.plot(x_range, 100*accuracy, label = 'Accuracy', c = 'r')

    plot = plot_1 + plot_2 + plot_3
    legends = [l.get_label() for l in plot]

    ax1.grid()
    ax1.set_xlabel('Number of Estimators', fontsize = 15)
    ax1.set_ylabel('Time (s)', fontsize = 15)
    ax1.legend(plot, legends, loc = 2, fontsize = 15)
    ax1.set_xticks([200, 1000, 2000, 3000])

    ax2.set_ylabel('Accuracy (%)', color='r', fontsize = 15)
    ax2.tick_params('y', colors='r')
    ax2.set_ylim(80, 100)

    plt.tight_layout()
    plt.show()

    return np.array(x_range), data

def visualization_micro():

    data_detail = np.load('running_time_detail.npy').mean(axis = 1)
    num_estimators_unit = 10

    training_time_detail = data_detail[:, 0]
    test_time_detail = data_detail[:, 1]
    accuracy_detail = data_detail[:, 2]

    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    x_range = range(1*num_estimators_unit, len(training_time_detail)*num_estimators_unit+1, num_estimators_unit)

    plot_1 = ax1.plot(x_range, training_time_detail, label = 'Training Time', c = 'g')
    plot_2 = ax1.plot(x_range, test_time_detail, label = 'Test Time', c = 'b')

    ax2 = ax1.twinx()
    plot_3 = ax2.plot(x_range, 100*accuracy_detail, label = 'Accuracy', c = 'r')

    plot = plot_1 + plot_2 + plot_3
    legends = [l.get_label() for l in plot]

    ax1.grid()
    ax1.set_xlabel('Number of Estimators', fontsize = 15)
    ax1.set_ylabel('Time (s)', fontsize = 15)
    ax1.set_xticks([10, 50, 100, 150, 200])
    ax1.legend(plot, legends, loc = 2, fontsize = 15)

    ax2.set_ylabel('Accuracy (%)', color='r', fontsize = 15)
    ax2.tick_params('y', colors='r')
    ax2.set_ylim(80, 100)

    plt.tight_layout()
    plt.show()

    return np.array(x_range), data_detail

x_range_macro, data = visualization_macro()
x_range_micro, data_detail = visualization_micro()

x_range = np.concatenate([x_range_micro, x_range_macro], axis = 0)
final_data = np.concatenate([data_detail, data], axis = 0)

training_time = final_data[:, 0]
test_time = final_data[:, 1]
accuracy = final_data[:, 2]

fig = plt.figure()
ax1 = fig.add_subplot(111)

plot_1 = ax1.plot(x_range, training_time, label = 'Training Time', c = 'g')
plot_2 = ax1.plot(x_range, test_time, label = 'Prediction Time', c = 'b')
ax1.vlines(400, ymin = 0, ymax = 20, linestyle = 'dashed', label = 'N = 400')

ax2 = ax1.twinx()
plot_3 = ax2.plot(x_range, 100*accuracy, label = 'Accuracy', c = 'r')

plot = plot_1 + plot_2 + plot_3
legends = [l.get_label() for l in plot]

ax1.grid()
ax1.set_xlabel('Number of Estimators', fontsize = 15)
ax1.set_ylabel('Time (s)', fontsize = 15)
ax1.tick_params('x', labelsize = 15)
ax1.tick_params('y', labelsize = 15)
ax1.set_xticks([10, 400, 1000, 2000, 3000])
ax1.set_yticks([0, 5, 10, 15, 20])
ax1.legend(plot, legends, loc = 2, fontsize = 15)

ax2.set_ylabel('Accuracy (%)', color='r', fontsize = 15)
ax2.tick_params('y', colors='r', labelsize = 15)
ax2.set_yticks([80, 85, 90, 95, 100])
ax2.set_ylim(80, 100)

plt.tight_layout()
plt.show()

































#end
