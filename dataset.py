from functools import reduce
from os import listdir
from os.path import join
from torch.utils.data import Dataset
import numpy as np
import torch


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

def torch_allisfinite(x):
    return torch.sum(torch_isnotfinite(x).view(-1)) == 0

def torch_isnotfinite(x):
    """
    Quick pytorch test that there are no nan's or infs.

    note: torch now has torch.isnan
    url: https://gist.github.com/wassname/df8bc03e60f81ff081e1895aabe1f519
    """
    not_inf = ((x + 1) != x)
    not_nan = (x == x)
    return 1 - (not_inf & not_nan)

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


def pad_tensor_list(tensor_list, pad_dim, pad_element=0):

    dim_permutation = list(range(len(tensor_list[0].size())))
    dim_permutation[0] = pad_dim
    dim_permutation[pad_dim] = 0

    tensor_list = [t.permute(dim_permutation) for t in tensor_list]

    for d in tensor_list:
        if not (d.size()[1:] == tensor_list[0].size()[1:]):
            raise RuntimeError("Probably forgot to call with correct pad_dim")

    lengths = [d.size(0) for d in tensor_list]
    max_length = max(lengths)
    padded_data = torch.full([len(tensor_list), max_length] +
                             list(tensor_list[0].size()[1:]), pad_element)

    for index, d in enumerate(tensor_list):
        padded_data[index][:lengths[index]] = d

    ret = padded_data.permute([0] + [a + 1 for a in dim_permutation])

    return ret, \
        torch.tensor(lengths)


def collate_pad_tensors(sample_list, pad_dim=0, pad_element=0):
    """
    From the pytorch documentation, the collate function "merges" a list of
    samples to form a mini-batch

    :sample_list: A python list of (data, label) tuples
    :pad_dimension: The dimension which is unequal in the samples
        (there can only be one)
    :pad_element: The element to pad with
    """

    data_list = [d[0] for d in sample_list]
    label_list = [d[1] for d in sample_list]

    padded_data, lengths = pad_tensor_list(data_list, pad_dim, pad_element)

    return padded_data, lengths, torch.tensor(label_list)


class VCTK_Wrapper:

    MAX_NUM_PEOPLE = 107
    MAX_NUM_SAMPLES = 172
    # For whatever reason, the ID of the first person is actually 225
    STARTING_ID = 225

    def __init__(self, embedder, num_people, num_samples,
                 mel_root, device):

        assert (num_people <= self.MAX_NUM_PEOPLE)
        assert (num_samples <= self.MAX_NUM_SAMPLES)
        self.num_samples = num_samples
        self.num_people = num_people
        self.mel_root = mel_root
        self.device = device

        self.embedder = embedder

        self.person_stylevecs = [None] * num_people
        self._calculate_person_stylevecs()

    def mel_from_ids(self, person_id, sample_id):

        assert (person_id <= self.num_people)
        assert (sample_id <= self.num_samples)

        actual_id = self.STARTING_ID + person_id
        #mel = np.load(self.mel_root + "p" + str(actual_id) + "/p" + str(actual_id) + "_" + "{:03d}".format(sample_id + 1) + ".npy").T
        mel = torch.load(self.mel_root + "p" + str(actual_id) + "/p" +
                         str(actual_id) + "_" + "{:03d}".format(sample_id + 1) +
                         ".pt").t()
        # TODO Maybe harcoding this isn't the greatest idea?
        if isinstance(mel, np.ndarray):
            mel = mel.astype(np.float32)
            mel = torch.from_numpy(mel)
        assert isinstance(mel, torch.Tensor)
        ret = (mel[None, :]).float().to(self.device)
        if not torch_allisfinite(mel):
            raise RuntimeError("Encountered non-finite data")
        return ret

    def _calculate_person_stylevecs(self, ):
        for pid in range(self.num_people):
            sample_stylevecs = [None] * self.num_samples
            for sid in range(self.num_samples):
                mel = self.mel_from_ids(pid, sid)[None, :]
                sample_stylevecs[sid] = self.embedder(mel)

            self.person_stylevecs[pid] = torch.mean(torch.stack(sample_stylevecs),
                                                    dim=0)

        self.person_stylevecs = torch.stack(self.person_stylevecs)

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

class Generator_Dataset(ParallelAudioDataset):
    """
    Dataset for training the generator.

    Loads every voice in VCTK, with random style vectors.
    """

    def __init__(self, wrapper):
        dims = (wrapper.num_people, wrapper.num_samples)
        super().__init__(wrapper, dims)

    def __getitem__(self, index):
        person_id, sample_id = coords_from_index(index, self.dims)
        style = self.wrapper.person_stylevec(np.random.randint(0, self.wrapper.num_people))
        mel = self.wrapper.mel_from_ids(person_id, sample_id)

        return mel, style

    def collate_fn(datalist):
        source_mels = [d[0] for d in datalist]
        target_styles = [d[1] for d in datalist]

        collated_mels, lengths = pad_tensor_list(source_mels, pad_dim=1)
        target_styles = torch.stack(target_styles)

        return collated_mels, lengths, target_styles

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
        super().__init__(wrapper, dims)

    def __getitem__(self, index):
        person_id, sample_id = coords_from_index(index, self.dims)
        ret = self.wrapper.mel_from_ids(person_id, sample_id)
        return ret, 1


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
        super().__init__(wrapper, dims)
        self.embedder = embedder
        self.transformer = transformer

    def __getitem__(self, index):
        """
        # TODO Work on some sort of caching if there are speed/memory issues,
            as I imagine there will be...
        """
        style_pid, \
            source_pid, source_sid = coords_from_index(index, self.dims)
        source_audio = self.wrapper.mel_from_ids(source_pid, source_sid)[None,:]
        stylevec = self.wrapper.person_stylevec(style_pid)

        if not torch_allisfinite(source_audio):
            raise RuntimeError("Source audio isn't finite!")
        if not torch_allisfinite(stylevec):
            raise RuntimeError("Style vector isn't finite")

        fake_sample = self.transformer(source_audio, stylevec)
        ret = fake_sample[0]

        if not torch_allisfinite(fake_sample):
            raise RuntimeError("Transformed audio isn't finite!")

        return ret, 0


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
        super().__init__(wrapper, dims)
        self.embedder = embedder

    def __getitem__(self, index):

        pid, sid1, sid2 = coords_from_index(index, self.dims)

        a1_mel = self.wrapper.mel_from_ids(pid, sid1)[None, :]
        a2_mel = self.wrapper.mel_from_ids(pid, sid2)[None, :]

        s1_stylevec = self.embedder(a1_mel)[0]
        s2_stylevec = self.embedder(a2_mel)[0]

        ret = np.array([s1_stylevec, s2_stylevec])
        return torch.from_numpy(), 1


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
        super().__init__(wrapper, dims)
        self.embedder = embedder
        self.transformer = transformer

    def __getitem__(self, index):
        source_pid, source_sid, style_pid = coords_from_index(index, self.dims)
        if style_pid >= source_pid:
            style_pid += 1

        source_mel = self.wrapper.mel_from_ids(source_pid,
                                               source_sid)[None, :]

        stylevec = self.wrapper.person_stylevec(source_pid)
        transformed_mel = self.transformer(source_mel, stylevec)
        transformed_stylevec = self.embedder(transformed_mel)[0]

        ret = torch.from_numpy(np.array([stylevec, transformed_stylevec]))

        return ret, 0


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
        super().__init__(wrapper, dims)

    def __getitem__(self, index):
        sid, p1id, p2id = coords_from_index(index, self.dims)
        mel1 = self.wrapper.mel_from_ids(p1id, sid)
        mel2 = self.wrapper.mel_from_ids(p2id, sid)

        # [COMPATIBILITY] The following line
        return pad_tensor_list([mel1, mel2], )[0], 1


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
        super().__init__(wrapper, dims)

    def __getitem__(self, index):
        p1id, s1id, p2id, s2id = coords_from_index(index, self.dims)

        # Make sure the same sentence is never chosen for both
        if s2id >= s1id:
            s2id += 1

        mel1 = self.wrapper.mel_from_ids(p1id, s1id)
        mel2 = self.wrapper.mel_from_ids(p2id, s2id)

        return pad_tensor_list([mel1, mel2])[0], 0
