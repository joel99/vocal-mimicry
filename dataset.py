from functools import reduce
from os.path import join
from os import listdir

import torch
from torch.utils.data import Dataset
import numpy as np

from transformer.mem_transformer import MemTransformer


class SoundDataset(Dataset):
    def __init__(self, source_dir):
        super().__init__()
        # TODO - installation script on nonexistent or empty source
        self.source_dir = source_dir
        self.filenames = listdir(
            source_dir
        )  # Note this is deterministic, but unintuitive (sort todo?)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        full_path = join(self.source_dir, self.filenames[index])
        return torch.from_numpy(np.load(full_path))


######################################################################


def coords_from_index(index, dimensions):
    prods = [None] * len(dimensions)
    p = 1
    for dim_index, dim in enumerate(dimensions[::-1]):
        p = p * dim
        prods[len(dimensions) - dim_index - 1] = p

    assert (index < prods[0])

    i = index
    coords = [None] * len(dimensions)
    for dim_index in range(len(dimensions) - 1):
        coords[dim_index] = int(i / prods[dim_index + 1])
        i = i % prods[dim_index + 1]
    coords[len(coords) - 1] = i

    return tuple(coords)


def stylevec_from_person(style_pid):
    """
    Given a person's ID (a number), return the stylevector (probably the
    average of the stylevectors for all their audio samples?)

    returns: The style vector as a torch tensor
    """
    # TODO [Alex] implement pls
    raise NotImplementedError()


def generate_voice(model, original_mel, stylevec):
    """
    Return a mel-spectrogram of the transformed voice

    Both original_mel and stylevec are torch tensors
    """
    return model.forward(original_mel, stylevec)


class VCTK_Wrapper:

    MAX_NUM_PEOPLE = 150
    MAX_NUM_SAMPLES = 300
    VCTK_MEL_ROOT = "data/mel/"
    # For whatever reason, the ID of the first person is actually 225
    STARTING_ID = 225

    def __init__(self, embedder, num_people, num_samples):

        assert (num_people <= self.MAX_NUM_PEOPLE)
        assert (num_samples <= self.MAX_NUM_SAMPLES)
        self.num_samples = num_samples
        self.num_people = num_people

        self.embedder = embedder

        self.person_stylevecs = [None] * num_people
        self._calculate_person_stylevecs()

    def mel_from_ids(self, person_id, sample_id):

        assert (person_id <= self.num_people)
        assert (sample_id <= self.num_samples)

        actual_id = self.STARTING_ID + person_id
        np_mel = np.load(self.VCTK_MEL_ROOT + "/p" + str(actual_id) + "/p" +
                         str(actual_id) + "_" + "{:03d}".format(sample_id) +
                         ".npy")
        return torch.from_numpy(np_mel)[None, :]

    def _calculate_person_stylevecs():
        for pid in range(self.num_people):
            sample_stylevecs = [None] * self.num_samples
            for sid in range(self.num_samples):
                sample_stylevecs[sid] = torch.from_numpy(self.embedder(self.mel_from_ids(pid, sid))[0])
            sample_stylevecs = np.array(sample_stylevecs)
            self.person_stylevecs[pid] = np.mean(sample_stylevecs, axis=0)

        self.person_stylevecs = torch.from_numpy(self.person_stylevecs)

    def person_stylevec(self, pid):
        return self.person_stylevecs[pid]


class ParallelAudioDataset(Dataset):
    def __init__(self, wrapper, dims):
        super().__init__()
        self.wrapper = wrapper
        self.dims = dims
        self.length = reduce(lambda x, y: x * y, self.dims, 1)

    def __len__(self):
        return self.length


class Isvoice_Dataset_Real(ParallelAudioDataset):
    """
    A class for training the isvoice discriminator.

    It just loads every voice in the VCTK dataset, and returns with a "true"
    label
    """

    def __init__(
            self,
            wrapper,
    ):
        dims = (wrapper.num_people, wrapper.num_samples)
        super().__init__(self, dims)

    def __getitem__(self, index):
        person_id, sample_id = coords_from_index(index, self.dims)
        return self.wrapper.mel_from_ids(person_id, sample_id), 1


class Isvoice_Dataset_Fake(ParallelAudioDataset):
    """
    A class for training the isvoice discriminator with negative (generated)
    examples
    """

    def __init__(
            self,
            wrapper,
            embedder,
            transformer,
    ):
        """
        There are (people * samples) original "real" files, and (people)
        possible transformations of each of files.
        """
        dims = (wrapper.num_people, wrapper.num_people, wrapper.num_samples)
        super().__init__(self, dims)
        self.embedder = embedder
        self.transformer = transformer

    def __getitem__(self, index):
        """
        # TODO Actually integrate this with the generator
        # TODO Work on some sort of caching if there are speed/memory issues,
            as I imagine there will be...
        """
        style_pid, \
            source_pid, source_sid = coords_from_index(index, self.dims)
        source_audio = self.wrapper.mel_from_ids(source_pid, source_sid)

        stylevec = self.wrapper.person_stylevec(style_pid)
        fake_sample = self.transformer(source_audio, stylevec)
        return fake_sample, 0


class Identity_Dataset_Real(ParallelAudioDataset):
    """
    For training the identity discriminator: provides both positive and
    negative samples

    Of course this assumes that we're training the identity discriminator in
    the first place: if we use cos/norm distance, there's no need for this
    discriminator at all

    TODO: I don't think there's any need to augment the dataset with
    transformed voices? I could be really wrong on this though.
    """

    def __init__(
            self,
            wrapper,
            embedder,
    ):
        """
        There are (people * samples) original "real" files, and (people)
        possible transformations of each of files.
        """
        dims = (wrapper.num_people, wrapper.num_samples, wrapper.num_samples)
        super().__init__(self, dims)
        self.embedder = embedder

    def __getitem__(self, index):

        pid, sid1, sid2 = coords_from_index(index, self.dims)

        a1_mel = self.wrapper.mel_from_ids(pid, sid1)
        a2_mel = self.wrapper.mel_from_ids(pid, sid2)

        s1_stylevec = embedder(a1_mel)
        s2_stylevec = embedder(a2_mel)

        return torch.from_numpy(np.array([s1_stylevec, s2_stylevec])), 1


class Identity_Dataset_Fake(ParallelAudioDataset):
    def __init__(
            self,
            wrapper,
            embedder,
            transformer,
    ):
        """
        There are (people * samples) original "real" files, and (people)
        possible transformations of each of files.
        """
        dims = (
            wrapper.num_people,
            wrapper.num_samples,
            wrapper.num_people - 1,
        )
        super().__init__(self, dims)
        self.embedder = embedder
        self.transformer = transformer

    def __getitem__(self, index):
        source_pid, source_sid, style_pid = coords_from_index(index, self.dims)
        if style_pid >= source_pid:
            style_pid += 1

        source_mel = self.wrapper.mel_from_ids(source_pid, source_sid)

        stylevec = self.wrapper.person_stylevec(source_pid)
        transformed_mel = self.transformer(source_mel, stylevec)
        transformed_stylevec = self.embedder(transformed_mel)

        return torch.from_numpy(np.array([stylevec, transformed_stylevec])), 0


class Content_Dataset_Real(ParallelAudioDataset):
    """
    TODO Current the "real" and "generated" dataset for content actually
    aren't trained on generated data, just positive/negative samples from
    the real dataset (ie, we're essentially just pretraining the discriminator)

    Is this OK? I feel like it's not
    """

    def __init__(
            self,
            wrapper,
    ):
        dims = (wrapper.num_samples, wrapper.num_people, wrapper.num_people)
        super().__init__(self, dims)

    def __getitem__(self, index):
        sid, p1id, p2id = coords_from_index(index, self.dims)
        mel1 = self.wrapper.mel_from_ids(p1id, sid)
        mel2 = self.wrapper.mel_from_ids(p2id, sid)

        # [COMPATIBILITY] The following four lines assume that the first
        # dimension of the mel-cepstrums is the one for time
        max_time = min(mel1.size(0), mel2.size(0))
        mel1 = torch.Tensor.numpy(mel1[:max_time])
        mel2 = torch.Tensor.numpy(mel2[:max_time])

        return torch.from_numpy(np.array([mel1, mel2])), 1


class Content_Dataset_Fake(ParallelAudioDataset):
    """
    TODO See todo item for Content_Dataset_Real
    """

    def __init__(
            self,
            wrapper,
    ):
        dims = (wrapper.num_people, wrapper.num_samples, wrapper.num_people,
                wrapper.num_samples - 1)
        super().__init__(self, dims)

    def __getitem__(self, index):
        p1id, s1id, p2id, s2id = coords_from_index(index, self.dims)

        # Make sure the same sentence is never chosen for both
        if s2id >= s1id:
            s2id += 1

        mel1 = self.wrapper.mel_from_ids(p1id, s1id)
        mel2 = self.wrapper.mel_from_ids(p2id, s2id)

        # [COMPATIBILITY] The following four lines assume that the first
        # dimension of the mel-cepstrums is the one for time
        max_time = min(mel1.size(0), mel2.size(0))
        mel1 = torch.Tensor.numpy(mel1[:max_time])
        mel2 = torch.Tensor.numpy(mel2[:max_time])

        return torch.from_numpy(np.array([mel1, mel2])), 0
