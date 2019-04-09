import argparse
import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from utils.checkpointing import CheckpointManager, load_checkpoint
from dataset import SoundDataset


parser = argparse.ArgumentParser()
parser.add_argument(
    "--config-yml", default="configs/basic.yml",
    help="Path to a config file listing model and optimization parameters."
)

parser.add_argument_group("Evaluation related arguments")
parser.add_argument(
    "--load-pthpath", default="checkpoints/checkpoint_xx.pth",
    help="Path to .pth file of pretrained checkpoint."
)

parser.add_argument_group("Arguments independent of experiment reproducibility")
parser.add_argument(
    "--gpu-ids", nargs="+", type=int, default=-1,
    help="List of ids of GPUs to use."
)

# for reproducibility - refer https://pytorch.org/docs/stable/notes/randomness.html
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# ================================================================================================
#   INPUT ARGUMENTS AND CONFIG
# ================================================================================================

args = parser.parse_args()

# keys: {"dataset", "model", "solver"}
config = yaml.load(open(args.config_yml))
if type(args.gpu_ids) == int: 
    args.gpu_ids = [args.gpu_ids]
device = torch.device("cuda", args.gpu_ids[0]) if args.gpu_ids[0] >= 0 else torch.device("cpu")

# print config and args
print(yaml.dump(config, default_flow_style=False))
for arg in vars(args):
    print("{:<20}: {}".format(arg, getattr(args, arg)))


# ================================================================================================
#   SETUP DATASET, DATALOADER, MODEL
# ================================================================================================

dataset = SoundDataset(config["dataset"]["source_dir"])
dataloader = DataLoader(
    dataset, batch_size=config["solver"]["batch_size"], num_workers=args.cpu_workers
)

model = None
if -1 not in args.gpu_ids:
    model = nn.DataParallel(model, args.gpu_ids)

model_state_dict, _ = load_checkpoint(args.load_pthpath)
if isinstance(model, nn.DataParallel):
    model.module.load_state_dict(model_state_dict)
else:
    model.load_state_dict(model_state_dict)
print("Loaded model from {}".format(args.load_pthpath))

# ================================================================================================
#   EVALUATION LOOP
# ================================================================================================

# Note that since our evaluation is qualitative, we may consider just generating audio
model.eval()

for i, batch in enumerate(tqdm(dataloader)):
    for key in batch:
        batch[key] = batch[key].to(device)
    with torch.no_grad():
        output = model(batch)
    # Do something with the prediction here
    pass