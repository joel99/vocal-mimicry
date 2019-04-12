"""
Tests for everything in the identity_dtor file, whoulda thunk?
"""

from common import *
from isvoice_dtor import Isvoice_Discriminator

def test_voice_discriminator():

    batch_size = 10
    slice_dim = 50
    num_timesteps = 200
    conv_arch = [(7, 3, None), (13, 4, 2)]
    hidden_list = [200, 500, 10, 300]

    example_data = np.zeros((batch_size, slice_dim, num_timesteps))
    example_data = torch.from_numpy(example_data).float()

    discrim = Isvoice_Discriminator(conv_arch, hidden_list, slice_dim,)

    discrim.forward(example_data)
