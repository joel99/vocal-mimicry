"""
Architecture
===================================================================

The architecture/parameters of the current architecture are pulled mostly from
(1) [page 3]. There are two key differences however.

    1. Said architecture is applied to speaker classification (which speaker is
       this) rather than isvoice discrimination (Is this a voice?). I have
       *assumed* that this architecture will suffice, as others (2, 3) have
       applied similar architectures (same basic idea but larger networks) to
       audio classification

    2. In (1), the given architecture was applied with fixed-time data. Since
       our data is possibly time-variable, I must implement some sort of global
       pooling across time to fix dimensionality for the fully connected
       layers. Currently, I am using Max Pooling. However, (4, page 4)
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

from .common import fc_from_arch, convnet_from_arch


def get_isvoice_discriminator(mel_channels,):
    """
    Return a network which takes one voice sample and returns the log
    probability that said sample is actually a voice

    This model takes in inputs of (1 x T x M)
    """

    conv_arch = [(32, 7, 2), (64, 5, None), (128, 3, None), (256, 3, None)]
    fc_arch = [1024, 256, 1024]
    return Isvoice_Discriminator(conv_arch=conv_arch,
                                 fc_arch=fc_arch,
                                 slice_size=mel_channels)


class Isvoice_Discriminator(nn.Module):
    def __init__(
            self,
            conv_arch,
            fc_arch,
            slice_size,
    ):
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
        self.conv_arch = conv_arch

        self.conv_layer, \
            self.size_after_conv = convnet_from_arch(self.original_slice_size,
                                                     self.conv_arch)

        # The 1 being the single output it needs to give the log-prob of input
        # being a voice
        self.fc_layer = fc_from_arch(
            self.size_after_conv * self.conv_arch[-1][0], 1, fc_arch)

    def forward(self, x, lengths, pool_func=torch.max):
        '''
        Run a set of inputs through the net to determine whether it is a voice

        Arguments:
            x (Variable): A tensor of size (N, 1, T, M) where
                N is the batch size
                T is the number of timesteps
                D is the dimenstionality of a timestep
            lengths: Integer torch tensor of size (N,) of unpadded lengths
        Returns:
            A torch Tensor of size (N,) specifying the score for the given
            examples
        '''
        assert (len(x.size()) == 4)
        assert(x.size(1) == 1)

        after_conv = self.conv_layer.forward(x)

        after_max = pool_func(after_conv, dim=2)
        if type(after_max) == tuple:
            after_max = after_max[0]
        flattened = after_max.reshape(x.size(0), -1)

        return torch.sigmoid(self.fc_layer(flattened))


if __name__ == "__main__":

    raise RuntimeError("Why in the world you running this as main?")
