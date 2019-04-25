import gc

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import yaml
import json
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

FOLDER_DTOR_IV = "isvoice-dtor/"
FOLDER_TRANSFORMER = "transformer/"

def train():

    #################################################
    # Argparse stuff click was a bad idea after all #
    #################################################

    parser = argparse.ArgumentParser()

    parser.add_argument('--config-json', type=str,
                        help="The json file specifying the args below")

    parser.add_argument('--embedder-path', type=str,
                        help="Path to the embedder checkpoint."
                        + " Example: 'embedder/data/best_model'")

    group_chk = parser.add_argument_group('checkpointing')
    group_chk.add_argument('--epoch-save-interval', type=int,
                        help="After every [x] epochs save w/"
                           + "checkpoint manager")
    group_chk.add_argument('--save-dir', type=str,
                        help="Relative path of save directory, "
                           + "include the trailing /")
    group_chk.add_argument("--load-dir", type=str,
                        help="Checkpoint prefix directory to "
                           + "load initial model from")

    group_system = parser.add_argument_group('system')
    group_system.add_argument('--cpu-workers', type=int,
                        help="Number of CPU workers for dataloader")
    group_system.add_argument('--torch-seed', type=int,
                        help="Seed for for torch and torch_cudnn")
    group_system.add_argument('--gpu-ids',
                help="The GPU ID to use. If -1, use CPU")

    group_data = parser.add_argument_group('data')
    group_data.add_argument('--mel-size', type=int,
                            help="Number of channels in the mel-gram")
    group_data.add_argument('--style-size', type=int,
                            help="Dimensionality of style vector")
    group_data.add_argument('--dset-num-people', type=int,
                            help="If using VCTK, an integer under 150")
    group_data.add_argument('--dset-num-samples', type=int,
                            help="If using VCTK, an integer under 300")
    group_data.add_argument('--mel-root', default='data/taco/', type=str,
                            help='Path to the directory (include last /) '
                            + 'where the person mel folders are')

    group_training = parser.add_argument_group('training')
    group_training.add_argument('--num-epochs', type=int,
                help="The number of epochs to train for")
    group_training.add_argument('--lr-dtor-isvoice', type=float, )
    group_training.add_argument('--lr-tform', type=float, )

    group_training.add_argument('--num-batches-dtor-isvoice', type=int,)
    group_training.add_argument('--batch-size-dtor-isvoice', type=int,)

    group_training.add_argument('--num-batches-tform', type=int,)
    group_training.add_argument('--batch-size-tform', type=int,)


    group_model = parser.add_argument_group('model')
    group_model.add_argument('--identity-mode',
                             help='One of [norm, cos, nn]')

    args = parser.parse_args()
    if args.config_json is not None:
        with open(args.config_json) as json_file:
            file_args = json.load(json_file)
        cli_dict = vars(args)
        for key in cli_dict:
            if cli_dict[key] is not None:
                file_args[key] = cli_dict[key]
        args.__dict__ = file_args

    print("CLI args are: ", args)
    with open("configs/basic.yml") as f:
        config = yaml.full_load(f)

    ############################
    # Setting up the constants #
    ############################

    if args.save_dir is not None and args.save_dir[-1] != "/":
        args.save_dir += "/"
    if args.load_dir is not None and args.load_dir[-1] != "/":
        args.load_dir += "/"

    SAVE_DTOR_ISVOICE = args.save_dir + FOLDER_DTOR_IV
    SAVE_TRANSFORMER = args.save_dir + FOLDER_TRANSFORMER

    ############################
    # Reproducibility Settings #
    ############################
    # Refer to https://pytorch.org/docs/stable/notes/randomness.html
    torch.manual_seed(args.torch_seed)
    torch.cuda.manual_seed_all(args.torch_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # TODO Enable?
    # torch.set_default_tensor_type(torch.cuda.FloatTensor)

    #############################
    # Setting up Pytorch device #
    #############################
    use_cpu = -1 == args.gpu_ids
    device = torch.device("cpu" if use_cpu else "cuda")

    ###############################################
    # Initialize the model and related optimizers #
    ###############################################

    if args.load_dir is None:
        start_epoch = 0
    else:
        start_epoch = int(args.load_dir.split("_")[-1][:-4])

    model = ProjectModel(config=config["transformer"],
                         embedder_path=args.embedder_path,
                         mel_size=args.mel_size,
                         style_size=args.style_size,
                         identity_mode=args.identity_mode,
                         cuda=(not use_cpu))
    model = model.to(device)
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
    if args.load_dir is not None:
        tform_md, tform_od = load_checkpoint(SAVE_TRANSFORMER)
        model.transformer.load_state_dict(tform_md)
        tform_optimizer.load_state_dict(tform_od)

        dtor_isvoice_md, dtor_isvoice_od = load_checkpoint(SAVE_DTOR_ISVOICE)
        model.dtor_isvoice.load_state_dict(dtor_isvoice_md)
        tform_optimizer.load_state_dict(dtor_isvoice_od)

    ##########################
    # Declaring the datasets #
    ##########################

    dset_wrapper = VCTK_Wrapper(
        model.embedder,
        args.dset_num_people,
        args.dset_num_samples,
        args.mel_root,
        device,
    )

    if args.mel_size != dset_wrapper.mel_from_ids(0, 0).size()[-1]:
        raise RuntimeError("mel size arg is different from that in file")

    dset_isvoice_real = Isvoice_Dataset_Real(dset_wrapper,)
    dset_isvoice_fake = Isvoice_Dataset_Fake(dset_wrapper,
                                             model.embedder,
                                             model.transformer)
    dset_generator_train  = Generator_Dataset(dset_wrapper,)
    # We're enforcing identity via a resnet connection for now, so unused
    # dset_identity_real = Identity_Dataset_Real(dset_wrapper,
    #                                            embedder)
    # dset_identity_fake = Identity_Dataset_Fake(dset_wrapper,
    #                                            embedder, transformer)

    collate_along_timeaxis = lambda x: collate_pad_tensors(x, pad_dim=1)
    dload_isvoice_real = DataLoader(dset_isvoice_real,
                                    batch_size=args.batch_size_dtor_isvoice,
                                    collate_fn=collate_along_timeaxis)
    dload_isvoice_fake = DataLoader(dset_isvoice_fake,
                                    batch_size=args.batch_size_dtor_isvoice,
                                    collate_fn=collate_along_timeaxis)
    dload_generator = DataLoader(dset_generator_train,
                                 batch_size=args.batch_size_tform,
                                 collate_fn=Generator_Dataset.collate_fn)

    #######################################################
    # The actual training loop gaaah what a rollercoaster #
    #######################################################
    train_start_time = datetime.now()
    print("Started Training at {}".format(train_start_time))
    for epoch in range(args.num_epochs):
        epoch_start_time = datetime.now()
        ###############
        # (D1) Train Real vs Fake Discriminator
        ###############
        train_dtor(model.isvoice_dtor, dtor_isvoice_optimizer,
                   dload_isvoice_real, dload_isvoice_fake,
                   args.num_batches_dtor_isvoice, device)
        dtor_isvoice_checkpointer.step()
        gc.collect()

        # Train generators here
        ################
        # (G) Update Generator
        ################
        val_loss = train_gen(model, tform_optimizer, dload_generator, device,
                             num_batches=args.num_batches_tform)
        tform_checkpointer.step()
        gc.collect()

if __name__ == "__main__":
    train()
