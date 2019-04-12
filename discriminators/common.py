"""
This file provides utility methods, mostly concerning data formatting and
functions to easily create neural nets
"""

import torch
from torch import nn
import numpy as np

def reformat_data(data):
    """
    Given some data, reformat it so that it corresponds to the specification
    below. Rewrite this function as needed (should probably never be more than
    a few lines) depending on how data is passed in from previous module

    The input data is to be formatted as follows: an (T x N x D) matrix where
        T: The number of timesteps per data
        N: The batch size
        D: Datum dimenstionality
    """
    # TODO Ensure that the data is being correctly formatted
    if torch.is_tensor(data):
        return data
    elif type(data) == np.ndarray:
        return torch.from_numpy(data)
    else:
        raise RuntimeError("Unknown data format")

    return data


def convnet_from_arch(timestep_dimensionality, in_channels, arch):
    """
    The architecture is a list of tuples of the form
        (conv_n_kernels, conv_kernel_size, pool_size)

    If pooling_size is None, then no pooling is added after the convolutional
    layer

    return: Tuple (layer, outsize) where

    layer: The actual nn.sequential object for use elsewhere in code outsize:
    The size of the dimensionality axis, after all convolutions are carried out
    """
    if len(arch) < 1:
        raise RuntimeError("Must have at least one convolutional layer")

    layers = []
    curr_channel_size = in_channels
    curr_dimensionality = timestep_dimensionality

    for (conv_n_kernels, conv_kernel_size, pool_size) in arch:
        # This padding ensures dimensionality is not reduced by the convolution
        conv_padding = int((conv_kernel_size - 1) / 2)

        layers.append(nn.Conv2d(curr_channel_size,
                                conv_n_kernels,
                                conv_padding).float())
        layers.append(nn.ReLU())
        curr_channel_size = conv_n_kernels

        if pool_size is not None:
            curr_dimensionality = int(curr_dimensionality / pool_size)
            layers.append(nn.MaxPool2d(pool_size))

    return nn.Sequential(*layers), curr_dimensionality

def fc_from_arch(input_dim, output_dim, hidden_list):
    """
    Create a multi-layer fully-connected neural network given the specifications

    :input_dim: An integer
    :output_dim: An integer
    :hidden_list: A list of integers
    """
    layer_list = [input_dim] + hidden_list + [output_dim]
    layers = []
    for index in range(len(layer_list[1:])):
        layers.append(nn.Linear(layer_list[index], layer_list[index + 1]))
        layers.append(nn.ReLU())

    return nn.Sequential(*layers)
