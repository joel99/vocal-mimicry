"""
Tests for everything in the identity_dtor file, whoulda thunk?
"""
import pytest

from common import *
from identity_dtor import Identity_Discriminator

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
    data = np.ones((2, batch_size, style_size))
    data = torch.from_numpy(data).float()
    return data

def test_siamese_voice_discriminator(style_size, fc_hidden_dims, ones_data):

    siamese_discrim = Identity_Discriminator(style_size, mode='nn',
                                             fc_hidden_arch=fc_hidden_dims)
    siamese_discrim.forward(ones_data)

def test_cos_voice_discriminator(style_size, ones_data):

    cosine_discrim = Identity_Discriminator(style_size, mode='cos',
                                            cossim_degree=3)
    p = cosine_discrim.forward(ones_data)
    assert(p[0] == 1)

def test_norm_voice_discriminator(style_size, ones_data):

    norm_discrim = Identity_Discriminator(style_size, mode='norm')
    p = norm_discrim.forward(ones_data)

    assert(p[0] == 1)
