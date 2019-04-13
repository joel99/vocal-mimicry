"""
TODO What architecture should I use RIP
"""
from __future__ import division

import torch
from torch import nn
import math

from common import fc_from_arch, convnet_from_arch

from isvoice_dtor import Isvoice_Discriminator


def get_content_discriminator(mel_size):
    """
    :mel_size: The dimensionality of a single time-slice of a mel-cepstrogram

    Return a network which takes two voice samples and returns the log
    probability that they are the same person

    See documentation of forward() below for information on input size
    """
    # TODO Change all of the constants below!
    time_res = 2
    conv_arch = [(2, (time_res, mel_size), None),
                 (2, (time_res, 1), None)]
    fc_hidden_arch = [500, 200, 50]
    return Content_Discriminator(conv_arch, fc_hidden_arch, mel_size)


class Content_Discriminator(Isvoice_Discriminator):

    def forward(self, input_):
        """
        :input_1: A (2, T, D) dimensional tensor (representing both mel-grams)

        NOTE: the user will have to ensure that the spectrograms are both
        clipped to the same timeframe somehow

        :returns: The probability that both inputs have the same content
        """

        i1 = input_[0]
        i2 = input_[1]
        expanded_diff = ((i1 - i2)**2)[None, :]

        return super(Content_Discriminator, self).forward(expanded_diff,
                                                          pool_func=torch.mean)


if __name__ == "__main__":

    raise RuntimeError("Why in the world you running this as main?")
