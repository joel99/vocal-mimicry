from tqdm import tqdm
import torch
from torch import nn, optim


def evaluate(model, val_dset, config=None):
    model.eval()
    criterion = nn.BCELoss()

    total_loss = 0
    for data, style in tqdm(val_dset, desc="Generator evaluation."):
        data, style = data.to(model.device), style.to(model.device)
        pred = model(data, style)
        ones_v = torch.ones(pred.size()).float()
        mel, is_voice, content, identity = pred
        loss = criterion(is_voice, ones_v) \
            + criterion(content, ones_v) + criterion(identity, ones_v)
        total_loss += loss
    print("Validation Loss: {}".format(total_loss / len(val_dset)))
    return total_loss / len(val_dset)


def train(model, optimizer, train_dset, config=None):
    # Turn on training mode which enables dropout.
    model.train()
    criterion = nn.BCELoss()

    pbar = tqdm(train_dset, desc="Generator training", unit="batch")
    for data, style in pbar:
        data, style = data.to(model.device), style.to(model.device)
        model.zero_grad()
        pred = model(data, style)
        mel, is_voice, content, identity = pred
        ones_v = torch.ones(is_voice.size()).float()
        loss = criterion(is_voice, ones_v) + \
            criterion(content, ones_v) + criterion(identity, ones_v)
        loss.backward()
        optimizer.step()
        pbar.set_postfix({"loss": loss})

    return evaluate(model, train_dset, config=config)
