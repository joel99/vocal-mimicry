import argparse
from os import listdir, makedirs, walk
from os.path import isfile, join

from librosa.filters import mel as librosa_mel_fn
from librosa.feature import melspectrogram
from librosa.core import load, ifgram
from tqdm import tqdm
import numpy as np
import torch
from taco.stft import STFT

from functools import partial
"""
TODO: 
- Consider pulling preprocessing from https://github.com/keithito/tacotron/blob/master/util/audio.py
- Batch file loading and add workers
- Add Audio config (fs)
- Implement listener preprocessing: https://towardsdatascience.com/human-like-machine-hearing-with-ai-2-3-f9fab903b20a
- Investigate Waveglow
- Investigate own VAE [lstm encoder -> wavenet decoder]
"""


fixed_mel = partial(melspectrogram, n_mels=80)

def dynamic_range_compression(x, C=1, clip_val=1e-5):
    """
    PARAMS
    ------
    C: compression factor
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)

# Match tacotron specs
def taco_mel(mel_basis, stft_fn, y, sr):
    max_wav_value=1.0 # different dataset
    audio = torch.FloatTensor(y.astype(np.float32))
    audio_norm = audio / max_wav_value
    audio_norm = audio_norm.unsqueeze(0)
    audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
    magnitudes, _ = stft_fn.transform(audio_norm)
    magnitudes = magnitudes.data
    mel_output = torch.matmul(mel_basis, magnitudes)
    mel_output = dynamic_range_compression(mel_output)
    return torch.squeeze(mel_output, 0) # we're returning tensor

def preprocess_wrap(embedder, name, in_dir, parser_args):
    for root, _, files in tqdm(walk(in_dir)):
        out_root = root[root.find('wav48') + 6:]
        out_dir = join(parser_args.out_dir, out_root)
        makedirs(out_dir, exist_ok=True)
        for name in files:
            ext_pt = name.rfind('.')
            name_stem = name[:ext_pt]
            name_ext = name[ext_pt+1:]
            if name_ext != 'wav':
                continue
            audio, _ = load(join(root, name), sr=22050, duration=10.0)
            gram = embedder(y=audio, sr=parser_args.fs)
            # fn = '{}.npy'.format(name_stem)
            # np.save(join(out_dir, fn), gram, allow_pickle=False)
            # Also save as torch, for waveglow
            torch.save(gram, join(out_dir, '{}.pt'.format(name_stem)))

preprocess_mel = partial(preprocess_wrap, fixed_mel, 'mel')

def if_wrap(y, sr):
    return ifgram(y,sr)[0]
preprocess_if = partial(preprocess_wrap, if_wrap, 'if')

def preprocess_listener(audio_paths, args):
    raise NotImplementedError

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in-dir", default="data/raw/",
        help="Path to raw audio."
    )

    valid_methods = ["mel", "if", "taco"]
    parser.add_argument(
        "--method", default=valid_methods[2],
        help="Preprocessing method to use.",
        choices=valid_methods
    )

    parser.add_argument(
        "--out-dir", default="data/",
        help="Output base directory"
    )

    args = parser.parse_args()

    args.fs = 22050 # 48000
    
    if args.method not in valid_methods:
        raise ValueError("Expected method to be one of {}".format(valid_methods))

    args.out_dir = join(args.out_dir, args.method)
    makedirs(args.out_dir, exist_ok=True)

    if args.method==valid_methods[0]:
        preprocess_mel(args.in_dir, args)
    if args.method==valid_methods[1]:
        preprocess_if(args._in_dir, args)
    if args.method==valid_methods[2]:
        mel_basis = librosa_mel_fn(
            22050, 1024, 80, 0, 8000)
        mel_basis = torch.from_numpy(mel_basis).float()
        stft = STFT(1024, 256, 1024)
        embed_taco = partial(taco_mel, mel_basis, stft)
        preprocess_taco = partial(preprocess_wrap, embed_taco, 'taco')
        preprocess_taco(args.in_dir, args)

if __name__ == "__main__":
    main()
        
