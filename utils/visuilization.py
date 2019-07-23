import torch
import numpy as np
# import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt


def plot_array(weight, fig_name = None):
    f = plt.figure()
    plt.ylim(np.min(weight)*1.1, np.max(weight)*1.1)
    plt.scatter(range(weight.shape[0]), weight, marker='o', s=0.1)
    if fig_name is not None:
        plt.savefig(fig_name)
    plt.close(f)


def plot_mask(mask, save_path = None):
    f = plt.figure()

    if len(mask.shape) == 4:
        mask = np.reshape(mask, [mask.shape[0], mask.shape[1]*mask.shape[2]*mask.shape[2]])
    plt.imshow(mask)
    plt.colorbar()
    if save_path is not None:
        plt.savefig(save_path)
    plt.close(f)


def draw_hist(mat, save_path=None, show=False):
    f = plt.figure()
    min = np.min(mat) - 0.01
    max = np.max(mat) + 0.01
    n_mat = mat.shape
    # n_interval = int((max - min) / n_mat)
    # plt.hist(mat, np.linspace(min, max, n_interval))
    plt.hist(mat.reshape(-1), np.linspace(min, max, 1000))
    if show:
        plt.show()
    if save_path is not None:
        plt.savefig(save_path)
    plt.close(f)