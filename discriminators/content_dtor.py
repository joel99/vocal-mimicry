"""
TODO What architecture should I use RIP
"""
from __future__ import division

import torch
from torch import nn
import math

from .common import fc_from_arch, convnet_from_arch

from .isvoice_dtor import Isvoice_Discriminator


def get_content_discriminator(mel_size):
    """
    :mel_size: The dimensionality of a single time-slice of a mel-cepstrogram

    Return a network which takes two voice samples and returns the log
    probability that they are the same person

    See documentation of forward() below for information on input size
    """
    # TODO Change all of the constants below!
    time_res = 2
    conv_arch = [(2, (time_res, mel_size), None), (2, (time_res, 1), None)]
    fc_hidden_arch = [500, 200, 50]
    return Content_Discriminator(conv_arch, fc_hidden_arch, mel_size)


class Content_Discriminator(Isvoice_Discriminator):
    def forward(self, x, lengths):
        """
        :i1: A (N, 2, 1, T, M) dimensional tensor (representing many mel-grams)

        :returns: The probability tensor (size n) that inputs[n] same content
        """

        assert (len(x.size()) == 5)
        assert (x.size(1) == 2)

        i1 = x[:, 0, :]
        i2 = x[:, 1, :]

        expanded_diff = ((i1 - i2)**2)

        return super(Content_Discriminator, self).forward(expanded_diff,
                                                          lengths,
                                                          pool_func=torch.mean)


if __name__ == "__main__":

    raise RuntimeError("Why in the world you running this as main?")
