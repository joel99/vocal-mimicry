"""
This file provides discriminators... Who would have guessed?

The architecture calls for two discriminators

    (1) Takes two inputs and determines whether they are spoken by the same
    person
    (2) Takes one input and determines whether it is an actual voice or not

I also assume that all data is given as FLOAT, not DOUBLE

===========================================================================
D1 Architecture: Idk, prolly some combination of a siamese net + LSTM?

D1 Architecture:
    * TODO: Explain this a bit https://arxiv.org/pdf/1803.05427.pdf
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

def get_sameperson_discriminator(timestep_dimensionality, n_input_channels=1):
    """
    Return a network which takes two voice samples and returns the log
    probability that they are the same person

    This model takes in inputs of (N x C x H x W)
    """
    # TODO It's possible that it
    # might be straight up better/easier to use the style embedding as the
    # feature space for the network, actually...
    raise NotImplementedError("Umm, it's not impletemented!")


def get_isvoice_discriminator(timestep_dimensionality):
    """
    :timestep_dimensionality: The dimension of a single time-slice in the
    mel-cepstrum

    Here I think it makes a lot of sense to use a convolutional net
    """
    pass


class Voice_Discriminator(nn.Module):
    def __init__(self, conv_arch, fc_arch, slice_size, n_input_channels=1,):
        '''
        Create components of a CNN classifier and initialize their weights.

        Arguments:
            im_size (tuple): A tuple of ints with (channels, height, width)
            hidden_dim (int): Number of hidden activations to use
            kernel_size (int): Width and height of (square) convolution filters
            n_classes (int): Number of classes to score
        '''
        super(Voice_Discriminator, self).__init__()

        self.original_slice_size = slice_size
        self.n_input_channels = 1
        self.conv_arch = conv_arch

        self.conv_layer, \
            self.reduced_slice_size = convnet_from_arch(self.original_slice_size,
                                                        self.n_input_channels,
                                                        self.conv_arch)

        # The 1 being the single output it needs to give the log-prob of input
        # being a voice
        self.fc_layer = fc_from_arch(self.reduced_slice_size*self.conv_arch[-1][0],
                                                  1, fc_arch)


    def forward(self, input_):
        '''
        Run a set of inputs through the net to determine whether it is a voice

        Arguments:
            images (Variable): A tensor of size (N, D, T) where
                N is the batch size
                T is the number of timesteps
                D is the dimenstionality of a timestep
        Returns:
            A torch Variable of size (N,) specifying the score
            for each example.
        '''
        if len(input_.size()) != 3:
            raise RuntimeError("Something's messed up with the dimensions!")
        extended_input = input_[:,None,:]
        after_conv = self.conv_layer.forward(extended_input)
        # TODO It's possible that it's good to max across the pitch dimension
        # too, instead of just the time dimension
        after_max, _ = torch.max(after_conv, 3)
        flattened = after_max.reshape(input_.size(0), -1)

        return self.fc_layer(flattened)

if __name__ == "__main__":
    raise RuntimeError("Why in the world you running this as main?")
