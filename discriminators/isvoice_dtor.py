"""
Architecture
===================================================================

The architecture/parameters of the current architecture are pulled mostly from (1)
[page 3]. There are two key differences however.

    1. Said architecture is applied to speaker classification (which speaker is
       this) rather than isvoice discrimination (Is this a voice?). I have
       *assumed* that this architecture will suffice, as others (2, 3) have
       applied similar architectures (same basic idea but larger networks) to
       audio classification

    2. In (1), the given architecture was applied with fixed-time data. Since
       our data is possibly time-variable, I must implement some sort of global
       pooling across time to fix dimensionality for the fully connected
       layers. Currently, (TODO) I am using Max Pooling. However, (4, page 4)
       indicates that average pooling is also an option (though their use ase
       is slightly different). My gut says that max pooling is more justified,
       but we can test different things

(1) Speaker Verification Using CNNs
        https://arxiv.org/pdf/1803.05427.pdf
(2) https://github.com/aqibsaeed/Urban-Sound-Classification
(3) CNN ARCHITECTURES FOR LARGE-SCALE AUDIO CLASSIFICATION
        https://arxiv.org/pdf/1609.09430.pdf
(4) Neural Voice Cloning with Few Samples
        https://arxiv.org/pdf/1802.06006.pdf
"""

import torch
from torch import nn
import numpy as np

from common import fc_from_arch, convnet_from_arch


def get_isvoice_discriminator(timestep_dimensionality, n_input_channels=1):
    """
    Return a network which takes one voice sample and returns the log
    probability that said sample is actually a voice

    This model takes in inputs of (N x C x H x W)
    """
    conv_arch = [(7, 32, 2), (5, 64, None), (3, 128, None), (3, 256, 1)]
    return Isvoice_Discriminator(conv_arch=conv_arch,
                                 fc_arch=[1024, 256, 1024],
                                 slice_size=timestep_dimensionality)


class Isvoice_Discriminator(nn.Module):

    def __init__(self, conv_arch, fc_arch, slice_size, n_input_channels=1,):
        '''
        Create components of a CNN classifier and initialize their weights.

        Arguments:
            im_size (tuple): A tuple of ints with (channels, height, width)
            hidden_dim (int): Number of hidden activations to use
            kernel_size (int): Width and height of (square) convolution filters
            n_classes (int): Number of classes to score
        '''
        super(Isvoice_Discriminator, self).__init__()

        self.original_slice_size = slice_size
        self.n_input_channels = n_input_channels
        self.conv_arch = conv_arch

        self.conv_layer, \
            self.size_after_conv = convnet_from_arch(self.original_slice_size,
                                                     self.n_input_channels,
                                                     self.conv_arch)

        # The 1 being the single output it needs to give the log-prob of input
        # being a voice
        self.fc_layer = fc_from_arch(self.size_after_conv*self.conv_arch[-1][0],
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
        extended_input = input_[:, None, :]
        after_conv = self.conv_layer.forward(extended_input)

        # TODO It's possible that it's good to max across the pitch dimension
        # too, instead of just the time dimension

        # TODO Maybe use average pooling instead of max pooling?
        after_max, _ = torch.max(after_conv, 3)
        flattened = after_max.reshape(input_.size(0), -1)

        return torch.sigmoid(self.fc_layer(flattened))


if __name__ == "__main__":

    raise RuntimeError("Why in the world you running this as main?")
