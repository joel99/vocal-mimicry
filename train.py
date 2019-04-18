import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import yaml
import click

from utils.checkpointing import CheckpointManager, load_checkpoint
from dataset import SoundDataset
from datetime import datetime
from model import ProjectModel
from discriminators.common import train_dtor
from transformer.new_train import train as train_gen

from dataset import VCTK_Wrapper, \
    Isvoice_Dataset_Real, Isvoice_Dataset_Fake, \
    Identity_Dataset_Real, Identity_Dataset_Fake, Generator_Dataset

@click.command()
@click.option('--verbose', default=0,)
@click.option('--cpu-workers', type=int, default=4,
              help="Number of CPU workers for dataloader")
@click.option('--torch-seed', type=int, default=0,
              help="Seed for for torch and torch_cudnn")
@click.argument('--gpu-ids', default=(0,), nargs=-1,
                help="The GPU IDs to use. If -1 appears anywhere, then use CPU")
@click.argument('--mel-size', required=True,
                help="The number of channels in the mel-gram. Placeholder")
@click.argument('--num-epochs', required=True,
                help="The number of epochs to train for")

# Checkpoint-related arguments #
@click.option('--epoch-save-interval',
              default=5,
              help="After every [x] epochs save w/ checkpoint manager")
@click.option('--save-dir', type=str, default="checkpoints/",
              help="Relative path of save directory, include the trailing /")
@click.option("--load-file", type=str, default=None,
              help="Checkpoint to load initial model from")

# Model-related arguments #
@click.option('--isvoice-mode', default='norm', help='One of [norm, cos, nn]')

def train(epoch_save_interval, isvoice_mode, verbose, cpu_workers, save_dir,
          load_file, torch_seed, gpu_ids, mel_size,
          num_epochs,):

    ############################
    # Reproducibility Settings #
    ############################
    # Refer to https://pytorch.org/docs/stable/notes/randomness.html
    torch.manual_seed(torch_seed)
    torch.cuda.manual_seed_all(torch_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    #############################
    # Setting up Pytorch device #
    #############################
    use_cpu = -1 in gpu_ids
    device = torch.device("cpu" if use_cpu else "cuda")

    ######################################################
    # Initialize the model and load checkpoint if needed #
    ######################################################
    model = ProjectModel(mel_size)
    optimizer = torch.optim.Adam(model.parameters())

    # Paralellize the model if we're told to use multiple GPUs
    if not use_cpu and (len(gpu_ids) > 1) :
        model = nn.DataParallel(model, gpu_ids)

    # Load the checkpoint, if it is specified
    if load_file is None:
        start_epoch = 0
    else:
        # "path/to/checkpoint_xx.pth" -> xx
        start_epoch = int(load_file.split("_")[-1][:-4])
        model_state_dict, optimizer_state_dict = load_checkpoint(load_file)
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(model_state_dict)
        else:
            model.load_state_dict(model_state_dict)
        optimizer.load_state_dict(optimizer_state_dict)

    checkpoint_manager = CheckpointManager(model, optimizer,
                                           save_dir, epoch_save_interval,
                                           start_epoch + 1)

    #######################################################
    # The actual training loop gaaah what a rollercoaster #
    #######################################################
    dset_wrapper = VCTK_Wrapper(model.embedder,
                                VCTK_Wrapper.MAX_NUM_PEOPLE,
                                VCTK_Wrapper.MAX_NUM_SAMPLES)
    dset_isvoice_real = Isvoice_Dataset_Real(dset_wrapper,
                                             embedder, transformer)
    dset_isvoice_fake = Isvoice_Dataset_Fake(dset_wrapper,
                                             embedder, transformer)
    dset_identity_real = Identity_Dataset_Real(dset_wrapper,
                                               embedder)
    dset_identity_fake = Identity_Dataset_Fake(dset_wrapper,
                                               embedder, transformer)
    dset_generator_train = Generator_Dataset(dset_wrapper)

    rf_d = model.isvoice_dtor
    rf_d_opt = optim.Adam(rf_d.parameters())
    gen = model.transformer
    gen_opt = optim.Adam(gen.parameters())

    train_start_time = datetime.now()
    print("Started Training at {}".format(train_start_time))
    for epoch in range(num_epochs):
        epoch_start_time = datetime.now()
        ################
        # (D1) Update Real vs Fake Discriminator
        ################
        train_dtor(rf_d, rf_d_opt, dset_isvoice_real, dset_isvoice_fake)

        ################
        # (G) Update Generator
        ################
        val_loss = train_gen(model, dset_generator_train, dset_generator_train)

        checkpoint_manager.step()

# train_begin = datetime.now()
# for epoch in range(config["solver"]["num_epochs"]):
#     # batch loop here
#     for i, batch in enumerate(dataloader):
#         for key in batch:
#             batch[key] = batch[key].to(device)

#         optimizer.zero_grad()
#         # Optimization here
#         # TODO
#     checkpoint_manager.step()
#     # Validation here, if done

if __name__ == "__main__":
    train()
