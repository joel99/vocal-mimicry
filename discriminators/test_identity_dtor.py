"""
Tests for everything in the identity_dtor file, whoulda thunk?
"""
from .identity_dtor import Identity_Discriminator
import numpy as np
import pytest
import torch


@pytest.fixture
def fc_hidden_dims():
    return [200, 500, 10, 300]


@pytest.fixture
def batch_size():
    return 10


@pytest.fixture
def style_size():
    return 50


@pytest.fixture
def ones_data(batch_size, style_size):
    data = np.ones((batch_size, 2, style_size))
    data = torch.from_numpy(data).float()
    return data


@pytest.fixture
def ones_data_lengths(ones_data,):
    return torch.ones((ones_data.size(0),)).int()


def test_siamese_voice_discriminator(style_size, fc_hidden_dims,
                                     ones_data, ones_data_lengths):

    siamese_discrim = Identity_Discriminator(style_size,
                                             mode='nn',
                                             fc_hidden_arch=fc_hidden_dims)
    siamese_discrim(ones_data, ones_data_lengths)


def test_cos_voice_discriminator(style_size,
                                 ones_data, ones_data_lengths):

    cosine_discrim = Identity_Discriminator(style_size,
                                            mode='cos',
                                            cossim_degree=3)
    p = cosine_discrim(ones_data, ones_data_lengths)
    assert (p[0] == 1)


def test_norm_voice_discriminator(style_size,
                                  ones_data, ones_data_lengths):

    norm_discrim = Identity_Discriminator(style_size, mode='norm')
    p = norm_discrim(ones_data, ones_data_lengths)

    assert (p[0] == 1)
