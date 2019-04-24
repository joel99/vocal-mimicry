from tqdm import tqdm
import torch
from torch import nn, optim


def evaluate(model, val_dset, config=None):
    model.eval()
    criterion = nn.BCELoss()

    with torch.no_grad():
        total_loss = 0
        for data, lengths, style in tqdm(val_dset, desc="Generator evaluation."):
            # data, style = data.to(model.device), style.to(model.device)
            pred = model(data, style)
            mel, is_voice, identity = pred
            ones_v = torch.ones(is_voice.size()).float()
            loss = criterion(is_voice, ones_v) + criterion(identity, ones_v)
            total_loss += loss
    print("Validation Loss: {}".format(total_loss.item() / len(val_dset)))
    return total_loss / len(val_dset)


def train(model, optimizer, train_dset, config=None,
          num_batches=None):
    # Turn on training mode which enables dropout.
    model.train()
    criterion = nn.BCELoss()

    batch_count = 0
    pbar = tqdm(train_dset, desc="Generator training", unit="batch")
    for data, lengths, style in pbar:
        if (num_batches is None) or (batch_count <= num_batches):
            model.zero_grad()
            pred = model(data, style)
            mel, is_voice, identity = pred
            ones_v = torch.ones(is_voice.size()).float()
            loss = criterion(is_voice, ones_v) + criterion(identity, ones_v)
            loss.backward(retain_graph=True)
            optimizer.step()
            pbar.set_postfix({"loss": loss.item()})
            batch_count += 1
        else:
            break

    return evaluate(model, train_dset, config=config)
