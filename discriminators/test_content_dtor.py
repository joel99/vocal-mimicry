"""
Tests for everything in the identity_dtor file, whoulda thunk?
"""
import pytest
from common import *
from content_dtor import Content_Discriminator, get_content_discriminator

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
    return torch.from_numpy(np.zeros((2, num_timesteps, mel_size))).float()

def test_content_discriminator1(mel_size, time_res, fc_hidden_list,
                                example_data):

    discrim = get_content_discriminator(mel_size)
    c = discrim.forward(example_data)
