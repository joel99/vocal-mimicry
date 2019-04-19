"""
Testing for everything in common file, whoulda thunk?
"""
import pytest
from .common import *

@pytest.fixture
def batch_size():
    return 10

@pytest.fixture
def mel_size():
    return 17

@pytest.fixture
def num_timesteps():
    return 200

@pytest.fixture()
def time_res():
    return 20

@pytest.fixture
def example_conv_data(mel_size, num_timesteps, batch_size):
    return torch.from_numpy(np.zeros((batch_size, 1,
                                     num_timesteps, mel_size))).float()

def test_convnet_from_arch(batch_size, mel_size, num_timesteps,):
    torch.set_default_tensor_type('torch.FloatTensor')
    conv_arch = [(7, 3, None), (11, 4, 2)]

    layer, dim = convnet_from_arch(mel_size, conv_arch)
    layer.forward(torch.from_numpy(np.zeros((batch_size,
                                             1,
                                             num_timesteps,
                                             mel_size))).float())

def test_convnet_from_arch_filtersizetuples(batch_size, mel_size,
                                            num_timesteps, example_conv_data,
                                            time_res):

    torch.set_default_tensor_type('torch.FloatTensor')
    conv_arch = [(30, (time_res, mel_size), None), (10, (time_res, 1), None)]

    layer, dim = convnet_from_arch(mel_size, conv_arch)
    layer.forward(example_conv_data)

def test_1d_convs_preserve_dims(example_conv_data, mel_size):
    """
    Convolutions with kernel size (1,1) should never change dimensions
    """
    conv_arch = [(12, (1, 1), None), (13, 1, None)]
    discrim, mel_dim = convnet_from_arch(mel_size, conv_arch,)
    c = discrim.forward(example_conv_data)
    assert c.size() == (example_conv_data.size(0), conv_arch[-1][0],
                        example_conv_data.size(2), example_conv_data.size(3))

def test_raw_conv_preserves_dims(example_conv_data, mel_size,):
    """
    The conv paddings are set so that the size of data after a convolutional
    pass should never differ by more than one in either direction... check it
    out!
    """
    conv_archictectures = [
        [(1, (13, 42), None)],
        [(1, (12, 43), None)],
        [(1, (14, 42), None)],
        [(1, (15, 42), None)],
        [(1, (16, 42), None)],
        [(1, (17, 42), None)],
        [(1, (18, 42), None)],
        [(1, (19, 42), None)],
        [(1, (20, 42), None)],
        [(1, (32, 42), None)],
    ]
    for arch in conv_archictectures:
        discrim, mel_dim = convnet_from_arch(mel_size, arch)
        c = discrim.forward(example_conv_data)

        assert((c.size(2) - example_conv_data.size(2))**2 <= 1)
        assert((c.size(3) - example_conv_data.size(3))**2 <= 1)

def test_melcep_conv_clipping_within_one(example_conv_data, mel_size):
    """
    When the melcep dim pooling window is within one of the actual melcep dim,
    the function is supposed to just automatically clip (the idea being to
    be able to reduce out a dimension without annoying off-by-one errors)

    This makes sure that clipping happens in no other cases
    """
    conv_arch = [(1, (1, 1), (1, mel_size + 1))]
    discrim, mel_dim = convnet_from_arch(mel_size, conv_arch)


    conv_arch = [(1, (1, 1), (1, mel_size - 1))]
    discrim, mel_dim = convnet_from_arch(mel_size, conv_arch)


    conv_arch = [(1, (1, 1), (1, mel_size))]
    discrim, mel_dim = convnet_from_arch(mel_size, conv_arch)

    try:
        conv_arch = [(1, (1, 1), (1, mel_size + 2))]
        discrim, mel_dim = convnet_from_arch(mel_size, conv_arch)
        assert(False)
    except RuntimeError:
        pass

def test_melcep_pool_out_dimension(example_conv_data, mel_size):
    """
    One is allowed to pass -1 for the mel_pool_size in order to immediately
    max-pool out the mel dimension. Check that this works correctly
    (this feature is important for the content dtor)
    """
    conv_arch = [(1, (1, 1), (1, -1))]
    discrim, mel_dim = convnet_from_arch(mel_size, conv_arch)
    c = discrim.forward(example_conv_data)

    assert(c.size(0) == example_conv_data.size(0))
    assert(c.size(1) == conv_arch[-1][0])
    assert(c.size(2) == example_conv_data.size(2))
    assert(c.size(3) == 1)

def test_pooling_affects_dimension_nicely():
    """
    Basically, make sure that pooling always reduces the dimension in
    a way we're OK with
    """
    # TODO Actually implement
    pass

########################################################################

@pytest.fixture
def input_dim():
    return 10

@pytest.fixture
def output_dim():
    return 3

@pytest.fixture
def fc_hidden_list():
    return [20, 30, 10]


def test_fc_from_arch(input_dim, output_dim, fc_hidden_list,
                      batch_size):
    torch.set_default_tensor_type('torch.FloatTensor')

    layer = fc_from_arch(input_dim, output_dim, fc_hidden_list)

    example_data = np.zeros((batch_size, input_dim))
    example_data = torch.from_numpy(example_data).float()

    output = layer(example_data)
    assert(output.size(0) == batch_size)
    assert(output.size(1) == output_dim)
