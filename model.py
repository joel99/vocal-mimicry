"""
Where everything comes together
"""
import torch

from discriminators.isvoice_dtor import get_isvoice_discriminator
from discriminators.content_dtor import get_content_discriminator
from discriminators.identity_dtor import get_identity_discriminator

from transformer.mem_transformer import MemTransformer

from embedding import embeddings

import warnings


def get_transformer(config):
    """
    Returns a neural network which takes a stylevector (N x S) tensor
    and audiosample (N x 1 x T x M) and returns another (N x 1 x T x M) tensor
    representing the output mel. (Where audiosample i has been transformed to
    have style stylevector[i])

    Neural net should extend pytorch.module so it can be easily checkpointed?

    where S is the dimensionality of style vector and M is the number of
    mel-spectrogram channels
    """
    return MemTransformer(config)

def get_embedder_and_size(mel_size, path=None, cuda=None):
    """
    Returns a embedding model captures the sytle of a speakers voice

    returns (Batch Size, style_size)

    where network which takes a transformation of a speakers utterances (Batch Size x 1 x Frames x Features)
    """

    embedder = None
    embedding_size = 512

    warnings.warn("Bypassing loading embedder!")

    if path != None:
        embedding_size, num_classes, num_features, num_frames = embeddings.parse_params(path)
        embedder = embeddings.load_embedder(
            checkpoint_path=path,
            embedding_size=embedding_size,
            require_audio_path=False,
            permute=True
        )
    else:
        print("No Model Found, initializing random weights")
        embedder = embeddings.load_embedder(
            embedding_size=embedding_size,
            require_audio_path=False,
            permute=True
        )

    return (embedder, embedding_size)


class ProjectModel(torch.nn.Module):
    def __init__(self, config, mel_size, identity_mode):
        """
        :style_size: The size of the stylevector produced by embedder
        :mel_size: The number of frequency channels in the mel-cepstrogram
        """
        super().__init__()

        self.mel_size = mel_size
        self.embedder, self.style_size = get_embedder_and_size(self.mel_size)

        config["d_model"] = self.mel_size
        config["d_style"] = self.style_size

        self.isvoice_dtor = get_isvoice_discriminator(self.mel_size)
        # self.content_dtor = get_content_discriminator(self.mel_size)
        self.identity_dtor = get_identity_discriminator(self.style_size,
                                                        identity_mode=identity_mode)
        self.transformer = get_transformer(config)

    def forward(self, source_mel, target_style):
        """
        :target_style: An (N x S) tensor
        :input_audio: A (N x 1 x T x M) tensor

        returns the tuple (out_audio, isvoice_prob,
        samecontent_prob, targetperson_prob)

        where the probabilities are scalar Tensors
        """

        transformed_mel = self.transformer(source_mel, target_style)
        transformed_style = self.embedder(transformed_mel)

        isvoice_prob = self.isvoice_dtor(transformed_mel, None)
        # content_prob = self.content_dtor(torch.stack((source_mel,
        #                                               transformed_mel),
        #                                              dim=1), None)
        identity_prob = self.identity_dtor(torch.stack((target_style,
                                                        transformed_style),
                                                       dim=1), None)

        return transformed_mel, isvoice_prob, identity_prob
