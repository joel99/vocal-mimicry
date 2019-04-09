import argparse
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from utils.checkpointing import CheckpointManager, load_checkpoint
from dataset import SoundDataset
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config-yml", default="configs/basic.yml",
    help="Path to a config file listing model and solver parameters."
)

parser.add_argument_group("Arguments independent of experiment reproducibility")
parser.add_argument(
    "--gpu-ids", nargs="+", type=int, default=0,
    help="List of ids of GPUs to use."
)
parser.add_argument(
    "--cpu-workers", type=int, default=4,
    help="Number of CPU workers for dataloader."
)

parser.add_argument_group("Checkpointing related arguments")
parser.add_argument(
    "--save-dirpath", default="checkpoints/",
    help="Path of directory to create checkpoint directory and save checkpoints."
)
parser.add_argument(
    "--load-pthpath", default="",
    help="To continue training, path to .pth file of saved checkpoint."
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

if isinstance(args.gpu_ids, int): 
    args.gpu_ids = [args.gpu_ids]
device = torch.device("cuda", args.gpu_ids[0]) if args.gpu_ids[0] >= 0 else torch.device("cpu")

# print config and args
print(yaml.dump(config, default_flow_style=False))
for arg in vars(args):
    print("{:<20}: {}".format(arg, getattr(args, arg)))

# ================================================================================================
#   TRAINING SETUP
# ================================================================================================

model = None

if -1 not in args.gpu_ids:
    model = nn.DataParallel(model, args.gpu_ids)

criterion = None
optimizer = optim.Adam(model.parameters())
scheduler = None

dataset = SoundDataset(config["dataset"]["source_dir"])
dataloader = DataLoader(
    dataset, batch_size=config["solver"]["batch_size"], num_workers=args.cpu_workers
)

checkpoint_manager = CheckpointManager(model, optimizer, args.save_dirpath, config=config)

# if loading from checkpoint, adjust start epoch and load parameters
if args.load_pthpath == "":
    start_epoch = 0
else:
    # "path/to/checkpoint_xx.pth" -> xx
    start_epoch = int(args.load_pthpath.split("_")[-1][:-4])

    model_state_dict, optimizer_state_dict = load_checkpoint(args.load_pthpath)
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(model_state_dict)
    else:
        model.load_state_dict(model_state_dict)
    optimizer.load_state_dict(optimizer_state_dict)

# ================================================================================================
#   TRAINING LOOP
# ================================================================================================
train_begin = datetime.now()
for epoch in range(config["solver"]["num_epochs"]):
    # batch loop here
    for i, batch in enumerate(dataloader):
        for key in batch:
            batch[key] = batch[key].to(device)

        optimizer.zero_grad()
        # Optimization here
        # TODO
    checkpoint_manager.step()
    # Validation here, if done

