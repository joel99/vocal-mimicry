"""
Tests for everything in the identity_dtor file, whoulda thunk?
"""
from content_dtor import get_content_discriminator
import numpy as np
import pytest
import torch


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
def example_data(num_timesteps, mel_size):
    return torch.from_numpy(np.zeros((1, num_timesteps, mel_size))).float()


def test_content_discriminator(mel_size, time_res, fc_hidden_list,
                                example_data):

    discrim = get_content_discriminator(mel_size)
    discrim(example_data, example_data)
