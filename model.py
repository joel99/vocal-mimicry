"""
Where everything comes together
"""
import torch

from discriminators.isvoice_dtor import get_isvoice_discriminator
from discriminators.content_dtor import get_content_discriminator
from discriminators.identity_dtor import get_identity_discriminator


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


def get_embedder(path='embedder/data/best_model', cuda = None):
    """
    Returns a embedding model captures the sytle of a speakers voice

    returns (network, style_size)

    where network which takes a transformation of a speakers utterances (Batch Size x 1 x Features x Frames)
    """

    from embedding import embeddings

    if path != None:
        embedding_size, num_classes = embeddings.parse_params(path)
        return embeddings.load_embedder(path, embedding_size, num_classes, cuda)
    else:
        print("No Model Found, initializing random weights")
        return embeddings.load_embedder()


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
        self.identity_dtor = get_identity_discriminator(self.style_size)
        self.transformer = get_transformer(self.style_size, self.mel_size)

    def forward(self, target_style, source_mel):
        """
        :target_style: A (1 x S) tensor
        :input_audio: A (1 x T x M) tensor

        returns the tuple (out_audio, isvoice_prob,
        samecontent_prob, targetperson_prob)

        where the probabilities are scalar Tensors
        """

        transformed_mel = self.transformer(source_mel, target_style)
        transformed_style = self.embedder(transformed_mel)

        isvoice_prob = self.isvoice_dtor(transformed_mel)
        content_prob = self.content_dtor(torch.stack((source_mel,
                                                      transformed_mel),
                                                     dim=1),)
        identity_prob = self.identity_dtor(torch.stack((target_style,
                                                        transformed_style),
                                                       dim=1))

        return transformed_mel, isvoice_prob, content_prob, identity_prob
