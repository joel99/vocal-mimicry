from tqdm import tqdm
import torch
from torch import nn, optim
from argparse import ArgumentParser
from pathlib import Path
import json
import yaml
import os

from utils.checkpointing import load_checkpoint
from model import ProjectModel

parser = ArgumentParser()
parser.add_argument("config_json", type=Path)
parser.add_argument("transformer_path", type=Path)
parser.add_argument("isvoice_dtor_path", type=Path)
parser.add_argument("style_mel", type=Path)
parser.add_argument("content_mel", type=Path)
parser.add_argument("--output-path", type=Path, default="outputs")
parser.add_argument("--use-gpu", action="store_true", default=False)


def generate(model, args, device):
    model.eval()
    criterion = nn.BCELoss()
    data = torch.load(args.content_mel)
    data = data.permute(1, 0)
    data = data.unsqueeze(0).unsqueeze(0)
    style_mel = torch.load(args.style_mel)
    style_mel = style_mel.permute(1, 0)
    style_mel = style_mel.unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        data, style_mel = data.to(device), style_mel.to(device)
        style = model.embedder(style_mel)
        pred = model.transformer(data, style)

    return pred

if __name__ == '__main__':
    args = parser.parse_args()

    with open(args.config_json) as json_file:
        file_args = json.load(json_file)

    with open("configs/basic.yml") as yaml_file:
        config = yaml.full_load(yaml_file)

    os.makedirs(args.output_path, exist_ok=True)

    device = torch.device("cuda") if args.use_gpu else torch.device("cpu")

    model = ProjectModel(config=config["transformer"],
                         embedder_path=file_args["embedder_path"],
                         mel_size=file_args["mel_size"],
                         style_size=file_args["style_size"],
                         identity_mode=file_args["identity_mode"],
                         cuda=args.use_gpu)

    model = model.to(device)
    tform_md, tform_od = load_checkpoint(args.transformer_path)
    model.transformer.load_state_dict(tform_md)

    mel = generate(model, args, device).to(torch.device("cpu")).squeeze().permute(1, 0)
    torch.save(mel, Path(args.output_path, args.content_mel.stem + "_" + args.style_mel.stem + ".pt"))
