"""
Where everything comes together
"""
import torch

from discriminators.isvoice_dtor import get_isvoice_discriminator
from discriminators.content_dtor import get_content_discriminator
from discriminitaors.identity_dtor import get_identity_dtor


def get_transformer(style_size, mel_size):
    """
    Returns a neural network which takes a stylevector (1 x S) tensor
    and audiosample (1 x T x M) and returns another (1 x T x M) tensor
    representing the output mel

    Neural net should extend pytorch.module so it can be easily checkpointed?

    where S is the dimensionality of style vector and M is the number of
    mel-spectrogram channels
    """
    raise NotImplementedError()


def get_embedder():
    """
    Net should extend pytorch.module so it can be easily checkpointed?

    returns (network, style_size)

    where network which takes an mel-spectrogram (1 x T x M)
    and returns a stylevector (1 x S) tensor

    and style_size is a plain old integer
    """
    raise NotImplementedError()


class ProjectModel(torch.nn.Module):
    def __init__(self, mel_size):
        """
        :style_size: The size of the stylevector produced by embedder
        :mel_size: The number of frequency channels in the mel-cepstrogram
        """
        super().__init__()

        self.embedder, self.style_size = get_embedder()
        self.mel_size = mel_size

        self.isvoice_dtor = get_isvoice_discriminator(self.mel_size)
        self.content_dtor = get_content_discriminator(self.mel_size)
        self.identity_dtor = get_identity_dtor(self.style_size)
        self.transformer = get_transformer(self.style_size, self.mel_size)

    def forward(self, target_style, source_mel):
        """
        :target_style: A (1 x S) tensor
        :input_audio: A (1 x T x M) tensor

        returns the tuple (out_audio, isvoice_prob,
        samecontent_prob, targetperson_prob)

        where the probabilities are scalar Tensors
        """

        source_style = self.embedder(source_mel)
        transformed_mel = self.transformer(input_audio, target_style)
        transformed_style = self.embedder(transformed_mel)

        isvoice_prob = self.isvoice_dtor(transformed_mel)
        content_prob = self.content_dtor(source_mel, transformed_mel)
        identity_prob = self.identity_dtor(target_style, transformed_style)

        return transformed_mel, isvoice_prob, content_prob, identity_prob
