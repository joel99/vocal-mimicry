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
    Isvoice_Dataset_Real, Isvoice_Dataset_Fake, collate_pad_tensors,
    Identity_Dataset_Real, Identity_Dataset_Fake, Generator_Dataset
from torch.utils.data import DataLoader

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

@click.argument('--lr-dtor-isvoice', default=0.001)
@click.argument('--lr-tform', default=0.001)

@click.argument('--num-batches-dtor-isvoice', required=True)
@click.argument('--batch-size-dtor-isvoice', required=True)

@click.argument('--num-batches-tform', required=True)
@click.argument('--batch-size-tform', required=True)

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
          num_epochs,
          lr_tform, lr_dtor_isvoice,
          num_batches_dtor_isvoice, batch_size_dtor_isvoice,
          num_batches_tform, batch_size_tform):

    ############################
    # Setting up the constants #
    ############################

    SAVE_DTOR_ISVOICE = load_file + "/" + "isvoice-dtor"
    SAVE_TRANSFORMER = load_file + "/" + "transformer"

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

    ###############################################
    # Initialize the model and related optimizers #
    ###############################################
    model = ProjectModel(mel_size)
    tform_optimizer = torch.optim.Adam(model.transformer.parameters(),
                                       lr=lr_tform)
    tform_checkpointer = CheckpointManager(model.transformer,
                                           tform_optimizer,
                                           SAVE_TRANSFORMER,
                                           epoch_save_interval,
                                           start_epoch + 1)

    dtor_isvoice_optimizer = torch.optim.Adam(model.isvoice_dtor.parameters(),
                                              lr=lr_dtor_isvoice)
    dtor_isvoice_checkpointer = CheckpointManager(model.isvoice_dtor,
                                                  dtor_isvoice_optimizer,
                                                  SAVE_DTOR_ISVOICE,
                                                  epoch_save_interval,
                                                  start_epoch + 1)

    ###############################################

    # Load the checkpoint, if it is specified
    if load_file is None:
        start_epoch = 0
    else:
        # "path/to/checkpoint_xx.pth" -> xx
        start_epoch = int(load_file.split("_")[-1][:-4])

        tform_md, tform_od = load_checkpoint(SAVE_TRANSFORMER)
        model.transformer.load_state_dict(tform_md)
        tform_optimizer.load_state_dict(tform_od)

        dtor_isvoice_md, dtor_isvoice_od = load_checkpoint(SAVE_DTOR_ISVOICE)
        model.dtor_isvoice.load_state_dict(dtor_isvoice_md)
        tform_optimizer.load_state_dict(dtor_isvoice_od)

    ##########################
    # Declaring the datasets #
    ##########################

    dset_wrapper = VCTK_Wrapper(model.embedder,
                                VCTK_Wrapper.MAX_NUM_PEOPLE,
                                VCTK_Wrapper.MAX_NUM_SAMPLES)
    dset_isvoice_real = Isvoice_Dataset_Real(dset_wrapper,
                                             embedder, transformer)
    dset_isvoice_fake = Isvoice_Dataset_Fake(dset_wrapper,
                                             embedder, transformer)
    # We're enforcing identity via a resnet connection for now, so unused
    # dset_identity_real = Identity_Dataset_Real(dset_wrapper,
    #                                            embedder)
    # dset_identity_fake = Identity_Dataset_Fake(dset_wrapper,
    #                                            embedder, transformer)
    dload_isvoice_real = DataLoader(dset_isvoice_real,
                                    batch_size=batch_size_dtor_isvoice,
                                    collate_fn=collate_pad_tensors)
    dload_isvoice_fake = DataLoader(dset_isvoice_fake,
                                    batch_size=batch_size_dtor_isvoice,
                                    collate_fn=collate_pad_tensors)

    #######################################################
    # The actual training loop gaaah what a rollercoaster #
    #######################################################
    train_start_time = datetime.now()
    print("Started Training at {}".format(train_start_time))
    for epoch in range(num_epochs):
        epoch_start_time = datetime.now()
        ###############
        # (D1) Train Real vs Fake Discriminator
        ###############
        train_dtor(model.isvoice_dtor, dtor_isvoice_optimizer,
                   dload_isvoice_real, dload_isvoice_fake,
                   batches_dtor_isvoice)
        dtor_isvoice_checkpointer.step()

        # Train generators here
        ################
        # (G) Update Generator
        ################
        val_loss = train_gen(model, tform_optimizer, dset_generator_train)
        tform_checkpointer.step()

if __name__ == "__main__":
    train()
