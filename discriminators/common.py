"""
This file provides utility methods, mostly concerning data formatting and
functions to easily create neural nets
"""

from __future__ import division

from torch import nn
import math
import numpy as np
import torch



def convnet_from_arch(mel_size, arch):
    """
    The architecture is a list of tuples of the form
        (conv_n_kernels, conv_kernel_size, pool_size)

    If pooling_size is None, then no pooling is added after the convolutional
    layer

    return: Tuple (layer, outsize) where

    layer: The actual nn.sequential object for use elsewhere in code outsize:
    The size of the dimensionality axis, after all convolutions are carried
    out
    """
    if len(arch) < 1:
        raise RuntimeError("Must have at least one convolutional layer")

    layers = []

    curr_channel_size = 1
    curr_mel_dim = mel_size

    for (conv_n_kernels, conv_size, pool_size) in arch:

        #############################
        # Convolutional Layer Stuff #
        #############################

        # First, we ensure that the conv_size is formatted correctly
        if type(conv_size) is tuple:
            assert (len(conv_size) == 2)
            assert (isinstance(conv_size[0], int))
            assert (isinstance(conv_size[1], int))
            t_conv_size, mel_conv_size = conv_size
        elif type(conv_size) is int:
            t_conv_size = conv_size
            mel_conv_size = conv_size
        else:
            raise RuntimeError("Unrecognized type for conv kernel size: " +
                               str(type(conv_size)))

        # Second, we choose the padding so as not to reduce the dimensionality
        # of the image via the convolution step
        mel_conv_padding = math.ceil((mel_conv_size - 1) / 2)
        t_conv_padding = math.ceil((t_conv_size - 1) / 2)
        conv_padding = (t_conv_padding, mel_conv_padding)

        # Finally, we actually add in the layer
        layers.append(
            nn.Conv2d(curr_channel_size,
                      conv_n_kernels,
                      kernel_size=conv_size,
                      padding=conv_padding).float())
        layers.append(nn.ReLU())
        curr_channel_size = conv_n_kernels

        new_mel_dim = math.ceil((curr_mel_dim + 2 * mel_conv_padding -
                                 (mel_conv_size - 1) - 1) + 1)
        assert ((new_mel_dim - curr_mel_dim)**2 <= 1)
        curr_mel_dim = new_mel_dim

        #######################
        # Pooling layer logic #
        #######################

        # Similarly to above, we ensure that the pooling spec is formatted
        # correctly

        if pool_size is None:
            continue
        elif type(pool_size) is tuple:
            assert (len(pool_size) == 2)
            assert (type(pool_size[0]) == int)
            assert (type(pool_size[1]) == int)
            t_pool_size, mel_pool_size = pool_size
        elif type(pool_size) is int:
            if pool_size <= 0:
                raise RuntimeError("Pool size must be positive")
            t_pool_size = pool_size
            mel_pool_size = pool_size
        else:
            raise RuntimeError("Unrecognized type for pool kernel size: " +
                               str(type(pool_size)))

        # If mel_pool_size is within one of mel_dim, set it to mel_dim
        # But if it's too big, then we have issues
        if mel_pool_size > curr_mel_dim + 1:
            raise RuntimeError("(mel_pool_size, curr_mel_dim) = "
                               + str((mel_pool_size, curr_mel_dim))
                               + "\nOff by greater than one, probably should "
                               + "make sure that clipping is correct here")
        elif (abs(mel_pool_size - curr_mel_dim) <= 1) \
             or (mel_pool_size == -1):
            mel_pool_size = curr_mel_dim

        pool_padding = 0
        layers.append(nn.MaxPool2d((t_pool_size, mel_pool_size)))

        curr_mel_dim = math.floor((curr_mel_dim + 2 * pool_padding -
                                   (mel_pool_size - 1) - 1) / mel_pool_size +
                                  1)

    return nn.Sequential(*layers), curr_mel_dim


def fc_from_arch(input_dim, output_dim, hidden_list):
    """
    Create a multi-layer fully-connected neural network given the
    specifications

    :input_dim: An integer
    :output_dim: An integer
    :hidden_list: A list of integers
    """
    layer_list = [input_dim] + hidden_list
    layers = []
    for index in range(len(layer_list[1:])):
        layers.append(nn.Linear(layer_list[index], layer_list[index + 1]))
        layers.append(nn.ReLU())

    layers.append(nn.Linear(layer_list[-1], output_dim))
    return nn.Sequential(*layers)


def train_dtor(dtor, optimizer, real_loader, fake_loader, num_batches, device):
    """
    Most of this code is stripped shamelelssly from
    https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

    But it's not quite the same, so I would really appreciate a sanity check

    :dtor: A model representing a discriminator. The forward() function should
        return a probability that the input is drawn from the real
        distribution
    :optimizer: An already-initalized Pytorch optimizer object.
        NOTE: The dtor params need to already be baked into the optimizer: I'm
              not sure if I could ensure this myself
              (see torch.optim.Optimization.add_param_group?)
    :real_loader: A dataloader object drawing examples from real distribution
    :fake_loader: A dataloader object drawing examples from fake distribution
    :num_batches: An integer or a tuple. If an integer, number of real and fake
        batches drawn will be equal. If tuple, should be (num_real, num_fake)

    According to Goodfellow's paper, the number of real and fake batches
    should be equal: I allow different options anyways

    Gradients are updated after every (real, fake) batch pair.
    """

    criterion = nn.BCELoss()

    if num_batches is None:
        assert(real_loader.batch_size == fake_loader.batch_size)
        lreal = len(real_loader)
        lfake = len(fake_loader)
        num_batches = int(min(lreal, lfake) / real_loader.batch_size)
    if isinstance(num_batches, int):
        num_real_batches = num_batches
        num_fake_batches = num_batches
    elif isinstance(num_batches, tuple):
        num_real_batches, num_fake_batches = num_batches
    else:
        raise TypeError("Invalid type for num_batches: " +
                        str(type(num_batches)))

    for batch_index in range(max(num_real_batches, num_fake_batches)):

        dtor.zero_grad()
        optimizer.zero_grad()

        cfgs = [(real_loader, num_real_batches,),
                (fake_loader, num_fake_batches,)]

        index = 0
        for loader, num_batches in cfgs:
            if batch_index < num_batches:
                data, lengths, labels = next(iter(loader))
                data = data.to(device)
                lengths = lengths.to(device)
                labels = labels.to(device)
                labels = labels.float()
                predictions = dtor(data, lengths).view(-1)
                err = criterion(predictions, labels)
                err.backward(retain_graph=True)
            index += 1

        optimizer.step()
