import argparse
from os import listdir, makedirs
from os.path import isfile, join

from librosa.feature import melspectrogram
from librosa.core import load
from tqdm import tqdm
import numpy as np

"""
TODO: 
- Consider pulling preprocessing from https://github.com/keithito/tacotron/blob/master/util/audio.py
- Batch file loading and add workers
- Add Audio config (fs)
- Implement listener preprocessing: https://towardsdatascience.com/human-like-machine-hearing-with-ai-2-3-f9fab903b20a
- Investigate Waveglow
- Investigate own VAE [lstm encoder -> wavenet decoder]
"""

def preprocess_mel(audio_paths, args):
    for index, path in tqdm(enumerate(audio_paths)):
        audio, _ = load(join(args.in_dir, path), sr=None)
        fn = 'mel_%05d.npy' % index
        mel = melspectrogram(y=audio, sr=args.fs).astype(np.float16)
        np.save(join(args.out_dir, fn), mel, allow_pickle=False)

def preprocess_listener(audio_paths, args):
    raise NotImplementedError

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in-dir", default="data/raw/",
        help="Path to raw audio."
    )

    valid_methods = ["mel", "listener"]
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

    audio_paths = [f for f in listdir(args.in_dir) if isfile(join(args.in_dir, f))]
    print(audio_paths[:3])

    if args.method==valid_methods[0]:
        preprocess_mel(audio_paths, args)
    if args.method==valid_methods[1]:
        preprocess_listener(audio_paths, args)

if __name__ == "__main__":
    main()