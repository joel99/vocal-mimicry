import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import yaml
import argparse

from utils.checkpointing import CheckpointManager, load_checkpoint
from dataset import SoundDataset
from datetime import datetime
from model import ProjectModel
from discriminators.common import train_dtor
from transformer.new_train import train as train_gen

from dataset import VCTK_Wrapper, \
    Isvoice_Dataset_Real, Isvoice_Dataset_Fake, collate_pad_tensors,\
    Identity_Dataset_Real, Identity_Dataset_Fake, Generator_Dataset
from torch.utils.data import DataLoader


def train():

    #################################################
    # Argparse stuff click was a bad idea after all #
    #################################################

    parser = argparse.ArgumentParser()

    parser.add_argument('--verbose', default=0, type=int,
                        help='[DUMMY] Does nothing currently')
    parser.add_argument('--cpu-workers', type=int, default=1,
                        help="Number of CPU workers for dataloader")
    parser.add_argument('--torch-seed', type=int, default=0,
                        help="Seed for for torch and torch_cudnn")
    parser.add_argument('--gpu-ids', default=-1,
                help="The GPU ID to use. If -1, use CPU")
    parser.add_argument('--mel-size', required=True,
                help="[DUMMY] The number of channels in the mel-gram")
    parser.add_argument('--num-epochs', required=True,
                help="The number of epochs to train for")
    parser.add_argument('--dset-num-people', type=int, required=True,
                        help="If using VCTK, an integer under 150")
    parser.add_argument('--dset-num-samples', type=int, required=True,
                        help="If using VCTK, an integer under 300")

    parser.add_argument('--lr-dtor-isvoice', type=float, default=0.001)
    parser.add_argument('--lr-tform', type=float, default=0.001)

    parser.add_argument('--num-batches-dtor-isvoice', type=int, required=True)
    parser.add_argument('--batch-size-dtor-isvoice', type=int, required=True)

    parser.add_argument('--num-batches-tform', type=int, required=True)
    parser.add_argument('--batch-size-tform', type=int, required=True)

    # Checkpoint-related arguments #
    parser.add_argument('--epoch-save-interval', type=int, default=5,
                        help="After every [x] epochs save w/ checkpoint manager")
    parser.add_argument('--save-dir', type=str, default="checkpoints/",
                        help="Relative path of save directory, include the trailing /")
    parser.add_argument("--load-file", type=str, default="",
                        help="Checkpoint prefix to load initial model from")

    # Model-related arguments #
    parser.add_argument('--isvoice-mode', default='norm',
                        help='One of [norm, cos, nn]')

    args = parser.parse_args()

    ############################
    # Setting up the constants #
    ############################

    SAVE_DTOR_ISVOICE = args.load_file + "/" + "isvoice-dtor"
    SAVE_TRANSFORMER = args.load_file + "/" + "transformer"

    ############################
    # Reproducibility Settings #
    ############################
    # Refer to https://pytorch.org/docs/stable/notes/randomness.html
    torch.manual_seed(args.torch_seed)
    torch.cuda.manual_seed_all(args.torch_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    #############################
    # Setting up Pytorch device #
    #############################
    use_cpu = -1 == args.gpu_ids
    device = torch.device("cpu" if use_cpu else "cuda")

    ###############################################
    # Initialize the model and related optimizers #
    ###############################################
    model = ProjectModel(args.mel_size)
    tform_optimizer = torch.optim.Adam(model.transformer.parameters(),
                                       lr=args.lr_tform)
    tform_checkpointer = CheckpointManager(model.transformer,
                                           tform_optimizer,
                                           SAVE_TRANSFORMER,
                                           args.epoch_save_interval,
                                           start_epoch + 1)

    dtor_isvoice_optimizer = torch.optim.Adam(model.isvoice_dtor.parameters(),
                                              lr=args.lr_dtor_isvoice)
    dtor_isvoice_checkpointer = CheckpointManager(model.isvoice_dtor,
                                                  dtor_isvoice_optimizer,
                                                  SAVE_DTOR_ISVOICE,
                                                  args.epoch_save_interval,
                                                  start_epoch + 1)

    ###############################################

    # Load the checkpoint, if it is specified
    if args.load_file is "":
        start_epoch = 0
    else:
        # "path/to/checkpoint_xx.pth" -> xx
        start_epoch = int(args.load_file.split("_")[-1][:-4])

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
                                args.dset_num_people,
                                args.dset_num_samples)
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
