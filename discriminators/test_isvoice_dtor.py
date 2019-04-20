"""
Tests for everything in the identity_dtor file, whoulda thunk?
"""

from .isvoice_dtor import Isvoice_Discriminator
import numpy as np
import torch
import pytest


@pytest.fixture
def batch_size():
    return 42


@pytest.fixture
def mel_size():
    return 37


@pytest.fixture
def time_res():
    return 20


@pytest.fixture
def fc_hidden_list():
    return [500, 200, 50]


@pytest.fixture
def num_timesteps():
    return 301


@pytest.fixture
def example_data(batch_size, num_timesteps, mel_size):
    return torch.from_numpy(np.zeros(
        (batch_size, 1, num_timesteps, mel_size))).float()


@pytest.fixture
def data_lengths(example_data, ):
    return torch.ones((example_data.size(0), )).int()


def test_voice_discriminator(mel_size, example_data, data_lengths):

    conv_arch = [(7, 3, None), (13, 4, 2)]
    hidden_list = [200, 500, 10, 300]

    discrim = Isvoice_Discriminator(
        conv_arch,
        hidden_list,
        mel_size,
    )

    discrim(example_data, data_lengths)
