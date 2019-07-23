"""
This code analysis the recorded training log from recorded txt
"""
import os
import numpy as np
from numpy import genfromtxt

import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle

import argparse
parser = argparse.ArgumentParser(description='Co-Prune')
args = parser.parse_args()

# -----------------
n_row = 2
# -----------------

method_list = {
    # Set the path and name, color for your desired visualization training log here, as example below shown.
    './Results/CIFARNet-CIFAR9-STL9/runs-STL9/CR1.35':
        {'name': 'Co-Prune-1.35', 'color': 'k'},
}


##################
# Retrieve alpha #
##################
alpha_changing_point_dict = dict()
for method_path, method_vis_info in method_list.items():

    method_name = method_vis_info['name']

    if os.path.exists('%s/alpha_change_point.txt' %method_path):
        alpha_changing_point_dict[method_name] = \
            genfromtxt('%s/alpha_change_point.txt' % (method_path), delimiter=',')

#################
# Visualization #
#################
plt.figure()
file_name_list = ['loss', 'lr', 'train-acc', 'test-acc']
for idx, data_name in enumerate(file_name_list):

    ax = plt.subplot(n_row, 2, idx+1)

    for method_path, method_vis_info in method_list.items():

        color = method_vis_info['color']
        label_name = method_vis_info['name']

        data = genfromtxt('%s/%s.txt' % (method_path, data_name), delimiter=',')[:, 1]
        xaxis = range(len(data))
        ax.plot(xaxis, data, color=color, linestyle='-', label=label_name, markersize=1)

        ###########################
        # Draw where alpha change #
        ###########################

        if label_name in alpha_changing_point_dict:
            if 'test' in data_name:
                # 36 for STL9 dataset for 36 batches in one epoch
                change_timestep = [int(alpha / 36) for alpha in alpha_changing_point_dict[label_name]]
            else:
                change_timestep = [int(alpha) for alpha in alpha_changing_point_dict[label_name]]
            # print(change_timestep)
            change_line = np.array(0, max(data))
            plt.vlines(change_timestep, 0, max(data), linestyle='--', color=color)

    ax.set_ylabel(data_name)
    if data_name in ['loss']:
        ax.set_yscale('log')
    ax.grid()
    if data_name == 'loss':
        ax.legend()

plt.show()