import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import numpy as np
import os

from torchvision import datasets, models, transforms
from PIL import Image

transforms_cifar10_train = transforms.Compose([
    transforms.Resize(32),   #must same as here
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(), # data augmentation
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]) # normalization
])
transforms_cifar10_test = transforms.Compose([
    transforms.Resize(32),   # must same as here
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])

transforms_cifar100_train = transforms.Compose([
    transforms.Resize(32),   #must same as here
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(), # data augmentation
    transforms.ToTensor(),
    transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]) # normalization
])
transforms_cifar100_test = transforms.Compose([
    transforms.Resize(32),   # must same as here
    transforms.ToTensor(),
    transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])
])

def import_data(dataset_name):

    root_dir = './'

    if dataset_name == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root=root_dir, train=True, download=True, transform=transforms_cifar10_train)
        valset = torchvision.datasets.CIFAR10(root=root_dir, train=True, download=True, transform=transforms_cifar10_test)
        testset = torchvision.datasets.CIFAR10(root=root_dir, train=False, download=True, transform=transforms_cifar10_test)
        num_classes = len(trainset.classes)
    elif dataset_name == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(root=root_dir, train=True, download=True, transform=transforms_cifar100_train)
        valset = torchvision.datasets.CIFAR100(root=root_dir, train=True, download=True, transform=transforms_cifar100_test)
        testset = torchvision.datasets.CIFAR100(root=root_dir, train=False, download=True, transform=transforms_cifar100_test)
        num_classes = len(trainset.classes)

    print(f"Num. classes: {num_classes}")
    print(f"Classes:\n {trainset.classes}")
    print(f"Num. train samples: {len(trainset)}")
    print(f"Num. test samples: {len(testset)}")

    return num_classes, trainset, valset, testset

class UnlabeledDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_name, transform_train=None):
        self.transform = transform_train

        if dataset_name.lower() == 'cifar10':
            self.dataset = datasets.CIFAR10(root='./', train=True, download=True, transform=self.transform)
        else:
            raise ValueError(f"Dataset {dataset_name} non supportato.")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, _ = self.dataset[idx]
        return image

def import_unlabeled_data(dataset_name):

    # Carica il dataset senza etichette
    train_data = UnlabeledDataset(dataset_name, transforms_cifar10_train)

    return train_data

def split_data_uniform(features, target, num_clients, rng):
    # Controlla se il numero di clienti è valido
    if num_clients <= 0:
        raise ValueError("Number of clients must be greater than 0.")
    unique_labels = np.unique(target)

    X_clients = [[] for _ in range(num_clients)]
    Y_clients = [[] for _ in range(num_clients)]
    
    shuffled_indices = rng.permutation(len(features))

    # One sample of each class to each client
    for class_label in unique_labels:
        class_indices = np.where(target == class_label)[0]
        rng.shuffle(class_indices)
        num_samples = len(class_indices)

        # WHEN NUM_SAMPLES_PER_CLASS < NUM_CLIENTS, WE REPEAT THOSE SAMPLES UNTIL THEY CAN BE SPLITTED AMONG CLIENTS. THIS HAPPENS ONLY FOR A FEW DATASETS, SUCH AS ECG5000, DIATOMSIZEREDUCTION, FACEFOUR. COMMENTED LINES ABOVE THIS DOES NOT HAPPEN.
        if num_samples < num_clients:
            class_indices = np.resize(class_indices, num_clients) 
        for i in range(num_clients):
            X_clients[i].append(features[class_indices[i % num_samples]])
            Y_clients[i].append(target[class_indices[i % num_samples]])

    # Remaining samples in uniform distribution
    for index in shuffled_indices:
        x_sample, y_sample = features[index], target[index]
        min_samples_client = min(range(num_clients), key=lambda k: len(Y_clients[k]))

        X_clients[min_samples_client].append(x_sample)
        Y_clients[min_samples_client].append(y_sample)

    # Convert lists in numpy arrays
    X_clients = [np.array(client) for client in X_clients]
    Y_clients = [np.array(client) for client in Y_clients]

    return X_clients, Y_clients