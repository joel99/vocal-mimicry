import argparse
from os import listdir, makedirs, walk
from os.path import isfile, join

from librosa.feature import melspectrogram
from librosa.core import load, ifgram
from tqdm import tqdm
import numpy as np

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
def preprocess_wrap(embedder, name, in_dir, args):
    for root, _, files in tqdm(walk(in_dir)):
        out_root = root[root.find('wav48') + 6:]
        out_dir = join(args.out_dir, out_root)
        makedirs(out_dir, exist_ok=True)
        for name in files:
            ext_pt = name.rfind('.')
            name_stem = name[:ext_pt]
            name_ext = name[ext_pt+1:]
            if name_ext != 'wav':
                continue
            audio, _ = load(join(root, name), sr=None, duration=10.0)
            gram = embedder(y=audio, sr=args.fs).astype(np.float16)
            fn = '{}.npy'.format(name_stem)
            np.save(join(out_dir, fn), gram, allow_pickle=False)
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

    valid_methods = ["mel", "if", "listener"]
    parser.add_argument(
        "--method", default=valid_methods[0],
        help="Preprocessing method to use (np, mel, listener)."
    )

    parser.add_argument(
        "--out-dir", default="data/",
        help="Output base directory"
    )

    args = parser.parse_args()

    args.fs = 48000
    
    if args.method not in valid_methods:
        raise ValueError("Expected method to be one of {}".format(valid_methods))

    args.out_dir = join(args.out_dir, args.method)
    makedirs(args.out_dir, exist_ok=True)

    if args.method==valid_methods[0]:
        preprocess_mel(args.in_dir, args)
    if args.method==valid_methods[1]:
        preprocess_if(args._in_dir, args)
    if args.method==valid_methods[2]:
        preprocess_mel(args.in_dir, args)

if __name__ == "__main__":
    main()