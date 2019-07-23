CR_ratio_1_35 = {
    'conv1': 0.2, # 0.4 ==> 40%
    'conv2': 0.01, # 0.02 ==> 2%
    'conv3': 0.009, # 0.018 ==> 1.8%
    'fc1': 0.009, # 0.018 ==> 1.8%
    'fc2': 0.08 # 0.16 ==> 16%
}

CR_ratio_3_07 = {
    'conv1': 0.15458,
    'conv2': 0.00816,
    'conv3': 0.00742,
    'fc1':   0.00944,
    'fc2':   0.06597
}


def get_sparse_param_analysis(net, CR_ratio, verbose=False):
    total_weights = 0
    total_left = 0
    for layer_name, CR in CR_ratio.items():
        layer = getattr(net, layer_name)
        if verbose:
            print(layer.weight.shape)
        layer_shape = layer.weight.shape
        if len(layer_shape) == 4 and verbose:
            print('Filter size: %d' %(layer_shape[1] * layer_shape[2] * layer_shape[3]))
        n_weights = layer.weight.data.numel()
        n_left = int(n_weights * CR)
        total_left += n_left
        total_weights += n_weights
        if verbose:
            print('[%s] Remaining weights: %d/%d' %(layer_name, n_left, n_weights))
    # if verbose:
    print('Overall CR: %.3f' %(total_left / total_weights))
    return total_left / total_weights

if __name__ == '__main__':

    pass