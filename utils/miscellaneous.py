"""
An utils function for those helper function without specific categorized
"""

import numpy as np
import torch
import os
import sys
import operator

def compare_layer_order(la, lb):
    """
    Return -1 if la < lb, 1 if la > lb it follows the rules as:
    layer4.0.downsample.0 > layer4.0.conv1
    fc > layer4.0.downsample.0
    """
    flag = True
    # return la <= lb
    # la = la[0]
    # lb = lb[0]
    if 'classifier' in la and 'classifier' in lb:
        if sys.version_info[0] == 2:
            flag = cmp(la, lb)
        else:
            flag = operator.gt(la, lb)
    elif 'classifier' in la:
        flag = 1
    elif 'classifier' in lb:
        flag = -1
    elif 'fc' in la and 'fc' in lb:
        flag = 0
    elif 'fc' in la:
        flag = 1
    elif 'fc' in lb:
        flag = -1
    else:
        if sys.version_info[0] == 2:
            flag = cmp(la, lb)
        else:
            flag = operator.gt(la, lb)

    return flag


def generate_layer_name_collections(net, model_name=None, idx_list=None):
    """
    This function generate an order [(conv_name, bn_name)] list,
    for examples: [('conv1, bn1'),('layer1.0.conv1', 'layer1.0.bn1')]
    :param net:
    :return:
    """

    layer_collection_list = list()

    for layer_name in net.state_dict().keys():
        # if 'conv1.weight' in layer_collection and layer_collection != 'conv1.weight':
        if 'conv1.weight' == layer_name or 'module.conv1.weight' == layer_name:
            layer_collection_list.append((layer_name[0: -7], layer_name[:-12] + 'bn1')) # [conv1, bn1] or [module.conv1, module.bn1]
        elif 'conv1.weight' in layer_name: # layer1.0.conv1.weight
            layer_name_prefix = layer_name[0: -12]
            layer_collection_list.append((layer_name_prefix + 'conv1', layer_name_prefix + 'bn1'))
        elif 'conv2.weight' in layer_name:
            layer_name_prefix = layer_name[0: -12]
            layer_collection_list.append((layer_name_prefix + 'conv2', layer_name_prefix + 'bn2'))
        elif 'conv3.weight' in layer_name:
            layer_name_prefix = layer_name[0: -12]
            layer_collection_list.append((layer_name_prefix + 'conv3', layer_name_prefix + 'bn3'))
        elif 'downsample.0.weight' in layer_name: # layer4.0.downsample.0.weight
            layer_name_prefix = layer_name[0: -8]
            layer_collection_list.append((layer_name_prefix + '0', layer_name_prefix + '1'))
        elif 'shortcut.0.weight' in layer_name: # layer4.0.shortcut.0.weight
            layer_name_prefix = layer_name[0: -8]
            layer_collection_list.append((layer_name_prefix + '0', layer_name_prefix + '1'))
        # linear.weight, module.linear.weight, classifier.1.weight
        elif 'linear.weight' in layer_name or 'fc.weight' in layer_name or \
                ('classifier' in layer_name and 'weight' in layer_name):
            layer_name_prefix = layer_name[0: -7]
            layer_collection_list.append((layer_name_prefix, ''))
        # Specify for VGG and AlexNet
        elif 'VGG' in model_name or 'AlexNet' in model_name:
            assert (idx_list is not None)
            for idx in idx_list:
                if 'features.%d.' %idx in layer_name:
                    layer_name_prefix = layer_name[0: -7]
                    layer_collection_list.append((layer_name_prefix, ''))
                    break
    return layer_collection_list


def generate_trainable_parameters(all_trainable_param, quantized_conv_name, model_name):
    """
    This function generate trainable parameters given specific quantized layer, it follows the rules as:
    1) Include weights except already quantized one
    2) Include all bn parameters

    all_trainable_param: an iterator containing all trainable parameters and their names
    quantized_conv_name:
        'layer1.2.conv2.weight', 'layer2.0.downsample.0'
        'module.layer3.1.conv1.weight'
    """
    trainable_parameters = []
    trainable_param_names = []
    stop_flag = True
    for layer_name, layer_param in all_trainable_param:

        # Include bn parameters
        if 'bn' in layer_name or 'shortcut.1' in layer_name:
            trainable_param_names.append(layer_name)
            trainable_parameters.append(layer_param)
        # Include bias, If is AlexNetBN (including the short one and long one), include all bias
        elif 'AlexNetBN' in model_name and 'bias' in layer_name:
            trainable_param_names.append(layer_name)
            trainable_parameters.append(layer_param)
        # Include first and last layer, For AlexNet-short, we don't quantize the first and last layer
        elif model_name == 'AlexNetBN-short' and ('features.0' in layer_name or 'classifier.6' in layer_name):
              trainable_param_names.append(layer_name)
              trainable_parameters.append(layer_param)
        # Include bn parameters, For AlexNetBN, because bn name is mixed with conv name, we specify the bn parameters
        elif ('AlexNetBN' in model_name) and ('features.1.' in layer_name or 'features.5.' in layer_name or 'features.9.' in layer_name or \
             'features.12' in layer_name or 'features.15' in layer_name):
            trainable_param_names.append(layer_name)
            trainable_parameters.append(layer_param)
        # Include bn parameters in VGG16-bn
        elif ('VGG16' in model_name) and ('features.1.' in layer_name or 'features.4.' in layer_name or \
                'features.8.' in layer_name or 'features.11.' in layer_name or 'features.15.' in layer_name or \
                'features.18.' in layer_name or 'features.21.' in layer_name or 'features.25.' in layer_name or \
                'features.28.' in layer_name or 'features.31.' in layer_name or 'features.35.' in layer_name or \
                'features.38.' in layer_name or 'features.41.' in layer_name):
            trainable_param_names.append(layer_name)
            trainable_parameters.append(layer_param)
        # Include unquantized layers, Normal action, until it reach the layer
        elif not stop_flag:
            trainable_param_names.append(layer_name)
            trainable_parameters.append(layer_param)

        # It will not include new layer until we reach the quantized layer
        if layer_name == quantized_conv_name:
            stop_flag = False

    return trainable_parameters, trainable_param_names


def folder_init(root, folder_name_list):
    for folder_name in folder_name_list:
        if not os.path.exists('../%s/%s/' % (root, folder_name)):
            os.makedirs('../%s/%s/' % (root, folder_name))


def convert_oht(targets, num_classes):
    batch_size = targets.size(0)
    target_oht = torch.FloatTensor(batch_size, num_classes)
    target_oht.zero_()
    target_oht.scatter_(1, targets, 1)

    return target_oht


def is_quantized(layer_name, model_name='ResNet18'):
    if 'ResNet' in model_name:
        if 'conv1' in layer_name or 'conv2' in layer_name or 'shortcut.0.' in layer_name or 'conv3' in layer_name or\
            'linear.weight' in layer_name or 'fc.weight' in layer_name:
            return True
        else:
            return False
    elif model_name == 'AlexNetBN':
        if 'features.0.weight' in layer_name or 'features.4.weight' in layer_name or 'features.8.weight' in layer_name or \
            'features.11.weight' in layer_name or 'features.14.weight' in layer_name or 'classifier.1.weight' in layer_name or \
            'classifier.4.weight' in layer_name or 'classifier.6.weight' in layer_name:
            return True
        else:
            return False
    elif model_name == 'AlexNetBN-short':
        if 'features.4.weight' in layer_name or 'features.8.weight' in layer_name or \
            'features.11.weight' in layer_name or 'features.14.weight' in layer_name or \
            'classifier.1.weight' in layer_name or 'classifier.4.weight' in layer_name :
            return True
        else:
            return False
    else:
        raise ('This model has not been implemented')


def initialize_model(net, pretrain_path, save_path):

    if save_path is not None and os.path.exists(save_path):
        print('Pretrain model found, loading from %s' %(save_path))
        pretrain_param = torch.load(save_path)
        net.load_state_dict(pretrain_param)
    elif os.path.exists(pretrain_path):
        print('Pretrain model unfound, creating.')
        pretrain_param = torch.load(pretrain_path)
        state_dict = net.state_dict()
        for layer_name in state_dict:
            if layer_name in pretrain_param and state_dict[layer_name].shape == pretrain_param[layer_name].shape:
                state_dict[layer_name] = pretrain_param[layer_name]
            else:
                print('Initial layer parameters: %s' % layer_name)
        if save_path is not None:
            torch.save(state_dict, open(save_path, 'wb'))
        net.load_state_dict(state_dict)
    else:
        print('Train a model from scratch')


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # from models_CIFAR10.resnet import ResNet18 as NetWork
    # from models_ImageNet.resnet import resnet18 as NetWork
    from models_ImageNet.alexnet_bn_layer_input import alexnet as NetWork
    # from models_ImageNet.alexnet_layer_input import alexnet as NetWork
    net = NetWork()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    # print (net.state_dict().keys())
    # layer_name_collection = generate_layer_name_collections(net)
    model_name = 'AlexNetBN-short'
    # print (layer_name_collection)
    # conv_name = layer_name_collection[5][0]
    # conv_name = 'module.conv1.weight'
    # conv_name = 'module.fc.weight'
    conv_name = 'module.classifier.4.weight'
    # print ('Process layer: %s' %conv_name)
    _, trainable_names = generate_trainable_parameters(net.named_parameters(), conv_name, model_name)
    for name in trainable_names:
        print (name)
