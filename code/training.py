import torch.nn.functional as F
import torch
import torchvision
import numpy as np
import torch.nn as nn
import tqdm
import wandb

from sklearn.metrics import f1_score
from matplotlib import pyplot as plt
from torchvision import utils


def train(client, net, num_classes, loaders, optimizer, criterion, round, LOG_WANDB=True, epochs=100, dev=torch.device('cuda')):
    try:
        net = net.to(dev)
        #print(net)
        # Initialize history
        history_loss = {"train": [], "test": []}
        history_accuracy = {"train": [], "test": []}
        history_f1 = {"train": [], "test": []}
        # Process each epoch
        for epoch in range(epochs):
            # Initialize epoch variables
            sum_loss = {"train": 0, "test": 0}
            sum_accuracy = {"train": 0, "test": 0}
            sum_f1 = {"train": 0, "test": 0}
            # Process each split
            for split in ["train", "test"]:
                # Process each batch
                for (input, labels) in loaders[split]:
                    # Move to CUDA
                    input = input.to(dev)
                    labels = labels.to(dev)
                    # Reset gradients
                    optimizer.zero_grad()
                    # Compute output
                    pred = net(input)
                    if num_classes == 2:
                        pred = pred.double()
                    labels = labels.type(torch.LongTensor)
                    labels = labels.to(dev)
                    if num_classes == 2:
                        labels = labels.unsqueeze(1)
                        labels = labels.float()
                    loss = criterion(pred, labels)
                    # Update loss
                    sum_loss[split] += loss.item()
                    # Check parameter update
                    if split == "train":
                        # Compute gradients
                        loss.backward()
                        # Optimize
                        optimizer.step()
                    # Compute accuracy
                    if num_classes == 2:
                        pred_labels = (pred >= 0).long() # Binarize predictions to 0 and 1
                        batch_accuracy = (pred_labels == labels).sum().item()/input.size(0)
                    else:
                        _,pred_labels = pred.max(1)
                        batch_accuracy = (pred_labels == labels).sum().item()/input.size(0)
                    # Update accuracy
                    sum_accuracy[split] += batch_accuracy
                    if num_classes == 2:
                        sum_f1[split] += f1_score(labels.cpu().numpy(), pred_labels.cpu().numpy(), average='binary', zero_division=0)
                    else:
                        sum_f1[split] += f1_score(labels.cpu().numpy(), pred_labels.cpu().numpy(), average='macro', zero_division=0)

            # Compute epoch loss/accuracy
            epoch_loss = {split: sum_loss[split]/len(loaders[split]) for split in ["train", "test"]}
            epoch_accuracy = {split: sum_accuracy[split]/len(loaders[split]) for split in ["train", "test"]}
            epoch_f1 = {split: sum_f1[split]/len(loaders[split]) for split in ["train", "test"]}
            if LOG_WANDB:
                    wandb.run.summary["step"] = epoch
                    wandb.log({f'Client {client} Training loss': epoch_loss["train"], f'Client {client} Training accuracy': epoch_accuracy["train"],
                               f'Client {client} Test loss': epoch_loss["test"], f'Client {client} Test accuracy': epoch_accuracy["test"],
                               f'Client {client} Training F1': epoch_f1["train"], f'Client {client} Test F1': epoch_f1["test"]})
            # Update history
            for split in ["train", "test"]:
                history_loss[split].append(epoch_loss[split])
                history_accuracy[split].append(epoch_accuracy[split])
                history_f1[split].append(epoch_f1[split])
            # Print info
            print(f"Client {client+1}:",
                  f"Epoch {epoch+1}:",
                  f"TrL={epoch_loss['train']:.4f},",
                  f"TrA={epoch_accuracy['train']:.4f},",
                  f"TeL={epoch_loss['test']:.4f},",
                  f"TeA={epoch_accuracy['test']:.4f},",
                  f"Trf1={epoch_f1['train']:.4f},",
                  f"Tef1={epoch_f1['test']:.4f},")
    except KeyboardInterrupt:
        print("Interrupted")


def denoising(client, net, train_loader, optimizer, criterion, round, epochs=100, dev=torch.device('cuda')):
    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        for images in tqdm.tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}'):
            images = images.to(dev)
            net = net.to(dev)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, images)  # MSE loss between noisy input and original image
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Client {client+1}, epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}")

def init(layer):
    if isinstance(layer, nn.Linear):
        nn.init.constant_(layer.weight.data, 0)
        nn.init.constant_(layer.bias.data, 0)