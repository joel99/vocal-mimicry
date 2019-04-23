"""
Architecture
====================================================================
The basic idea of this discriminator is to, given two voice samples,
determine whether they belong to the same person.

I opted for a Siamese-network approach to the problem, as described in (1):
which is a paper on exactly this topic. The architecture described in this
paper uses a neural-network dimensionality reduction on both inputs before
passing the reductions to some similarity function. After thinking about what
makes a good dimensionality reduction for voices such that they can be
compared, we came to the conclusion that it makes sense to use the style
embeddings themselves as the dimensionality-reduced data

The forward() function provides the probability that the two voices
are the same. I support different ways to calculate this probability, being

    1. Via norm of embedding difference
    2. Via cosine similarity
    3. Via learning the function via a (fully connected) neural network


References
----------------------------------
(1) Speaker Verification Using CNNs
        https://arxiv.org/pdf/1803.05427.pdf
"""
from __future__ import division

import torch
from torch import nn
import math

from .common import fc_from_arch


def get_identity_discriminator(style_size, identity_mode):
    """
    Return a network which takes two voice samples and returns the
    probability that they are the same person

    See documentation of forward() below for information on input size
    """
    return Identity_Discriminator(
        style_size,
        mode=identity_mode,
    )


class Identity_Discriminator(nn.Module):

    modes = ['norm', 'cos', 'nn']

    def __init__(
            self,
            style_size,
            mode='norm',
            fc_hidden_arch=None,
            cossim_degree=None,
    ):
        """
        :style_size: An integer, the size of the style embedding vector
        :distance_mode: One of 'norm', 'nn', 'cos'
        :fc_hidden_arch: The hidden layers to be used in the neural network if
            the difference function is to be learned. If distance_mode is not
            'nn' and this parameter is not None, or if distance mode is 'nn'
            and this parameter is None, then a runtime error will be thrown
        """

        super(Identity_Discriminator, self).__init__()

        self.style_size = style_size
        self.fc_hidden_arch = fc_hidden_arch
        self.cossim_degree = cossim_degree
        self.mode = mode

        if not (self.mode in self.modes):
            raise RuntimeError("Unrecognized mode: " + str(self.mode))

        if (self.mode == 'nn') and (fc_hidden_arch is None):
            raise RuntimeError("In NeuralNet mode but no arch provided")
        elif (self.mode != 'nn') and (fc_hidden_arch is not None):
            raise RuntimeError("Not in NeuralNet mode but arch provided")

        if (self.mode == 'cos') and (cossim_degree is None):
            raise RuntimeError("In Cos-Sim mode but no exponent provided")
        elif (self.mode != 'cos') and (cossim_degree is not None):
            raise RuntimeError("Not in Cos-Sim mode but exponent provided")

        if self.mode != 'nn':
            self.network = None
        else:
            self.network = fc_from_arch(2 * style_size, 1, self.fc_hidden_arch)

    def forward(self, x, lengths):
        """
        :x: should be a (N x 2 x S) tensor

        Returns a vector of shape (N,), with each entry being the probability
        that i1[n] and i2[n] were stylevectors for the same person
        """
        assert(len(x.size()) == 3)
        assert(x.size(1) == 2)

        i1 = x[:, 0, :]
        i2 = x[:, 1, :]

        if self.mode == 'norm':
            return 1 - (
                (2 / math.pi) * torch.atan(torch.norm(i1 - i2, p=2, dim=1)))
        elif self.mode == 'cos':
            return ((nn.functional.cosine_similarity(i1, i2, dim=1) + 1) /
                    2)**self.cossim_degree
        elif self.mode == 'nn':
            return torch.sigmoid(
                self.network.forward(torch.cat((i1, i2), dim=1)))


if __name__ == "__main__":

    raise RuntimeError("Why in the world you running this as main?")
