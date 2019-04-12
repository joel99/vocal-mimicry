"""
Testing for everything in common file, whoulda thunk?
"""

from common import *

def test_convnet_from_arch():
    torch.set_default_tensor_type('torch.FloatTensor')
    batch_size = 10
    timestep_dimensionality = 32
    input_channels = 3
    num_timesteps = 200
    conv_arch = [(7, 3, None), (11, 4, 2)]

    layer, dim = convnet_from_arch(timestep_dimensionality,
                                   input_channels,
                                   conv_arch)
    layer.forward(torch.from_numpy(np.zeros((batch_size,
                                             input_channels,
                                             num_timesteps,
                                             timestep_dimensionality))).float())
    assert(dim == 16)

def test_fc_from_arch():
    torch.set_default_tensor_type('torch.FloatTensor')

    batch_size = 10
    input_dim = 32
    output_dim = 200
    hidden_list = [200, 500, 10, 300]

    layer = fc_from_arch(input_dim, output_dim, hidden_list)

    example_data = np.zeros((batch_size, input_dim))
    example_data = torch.from_numpy(example_data).float()

    output = layer(example_data)
    assert(output.size(0) == batch_size)
    assert(output.size(1) == output_dim)
