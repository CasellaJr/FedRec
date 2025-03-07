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


def train(client, net, num_classes, loaders, optimizer, scheduler, criterion, round, LOG_WANDB=True, epochs=200, testing=False, dev=torch.device('cuda')):
    try:
        net = net.to(dev)
        # Initialize history
        history_loss = {split: [] for split in ["train", "val", "test"] if split in loaders}
        history_accuracy = {split: [] for split in ["train", "val", "test"] if split in loaders}
        history_f1 = {split: [] for split in ["train", "val", "test"] if split in loaders}
        
        # Process each epoch
        for epoch in range(epochs):
            # Initialize epoch variables
            sum_loss = {split: 0 for split in loaders}
            sum_accuracy = {split: 0 for split in loaders}
            sum_f1 = {split: 0 for split in loaders}
            
            # Process each split
            for split in loaders.keys():
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
                    labels = labels.type(torch.LongTensor).to(dev)
                    if num_classes == 2:
                        labels = labels.unsqueeze(1).float()
                    loss = criterion(pred, labels)
                    # Update loss
                    sum_loss[split] += loss.item()
                    
                    if split == "train":
                        # Compute gradients
                        loss.backward()
                        optimizer.step()
                        
                    # Compute accuracy
                    if num_classes == 2:
                        pred_labels = (pred >= 0).long()
                        batch_accuracy = (pred_labels == labels).sum().item() / input.size(0)
                    else:
                        _, pred_labels = pred.max(1)
                        batch_accuracy = (pred_labels == labels).sum().item() / input.size(0)
                    
                    sum_accuracy[split] += batch_accuracy
                    
                    if num_classes == 2:
                        sum_f1[split] += f1_score(labels.cpu().numpy(), pred_labels.cpu().numpy(), average='binary', zero_division=0)
                    else:
                        sum_f1[split] += f1_score(labels.cpu().numpy(), pred_labels.cpu().numpy(), average='macro', zero_division=0)
                
                if split == "train":
                    scheduler.step()

            # Compute epoch metrics only for available splits
            epoch_loss = {split: sum_loss[split] / len(loaders[split]) for split in loaders}
            epoch_accuracy = {split: sum_accuracy[split] / len(loaders[split]) for split in loaders}
            epoch_f1 = {split: sum_f1[split] / len(loaders[split]) for split in loaders}

            # Logging su wandb solo se lo split è disponibile
            if LOG_WANDB:
                wandb.run.summary["step"] = epoch
                '''
                for split in loaders:
                    wandb.log({f'Client {client} {split.capitalize()} Loss': epoch_loss[split],
                               f'Client {client} {split.capitalize()} Accuracy': epoch_accuracy[split],
                               f'Client {client} {split.capitalize()} F1': epoch_f1[split]})
                '''
                wandb.log({f'Client {client} {split.capitalize()} Accuracy': epoch_accuracy["test"],
                        f'Client {client} {split.capitalize()} F1': epoch_f1["test"]})

            # Update history solo se lo split è disponibile
            for split in loaders:
                history_loss[split].append(epoch_loss[split])
                history_accuracy[split].append(epoch_accuracy[split])
                history_f1[split].append(epoch_f1[split])

            # Print info
            print(f"Client {client+1} Epoch {epoch+1}:", end=" ")
            for split in loaders:
                print(f"{split.capitalize()} Loss={epoch_loss[split]:.4f}, {split.capitalize()} Acc={epoch_accuracy[split]:.4f}, {split.capitalize()} F1={epoch_f1[split]:.4f}", end=", ")
            print()

        if "test" in loaders and testing:
            return torch.tensor(epoch_accuracy["test"]), torch.tensor(epoch_f1["test"])
        else:
            return None


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
            print(outputs.shape, images.shape)
            loss = criterion(outputs, images)  # MSE loss between noisy input and original image
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()
            running_loss += loss.item() * images.size(0)
            #return #interrompi dopo una batch
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Client {client+1}, epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}")

def init(layer):
    if isinstance(layer, nn.Linear):
        nn.init.constant_(layer.weight.data, 0)
        nn.init.constant_(layer.bias.data, 0)