import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import numpy as np
import os

from collections import defaultdict
from numpy.random import dirichlet, choice
from typing import List, Tuple
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

'''
def split_data_dirichlet(features: np.ndarray,
                         target: np.ndarray,
                         num_clients: int,
                         rng: np.random.Generator = np.random.default_rng(),
                         beta: float = 0.5) -> Tuple[List[np.ndarray], List[np.ndarray]]:

    assert beta > 0, "beta must be > 0"
    if isinstance(target, list):
        target = np.array(target)
    if isinstance(features, list):
        features = np.array(features)
    
    labels = np.unique(target)
    pk = {c: rng.dirichlet(beta * np.ones(num_clients), size=1)[0] for c in labels}
    assignment = np.zeros(target.shape[0], dtype=int)
    
    X_clients = [[] for _ in range(num_clients)]
    Y_clients = [[] for _ in range(num_clients)]
    
    for c in labels:
        ids = np.where(target == c)[0]
        assignment[ids] = rng.choice(num_clients, size=len(ids), p=pk[c])
        
    features_clients = [features[assignment==i] for i in range(num_clients)]
    target_clients = [target[assignment==i] for i in range(num_clients)]
    
    return features_clients, target_clients
'''

def split_data_dirichlet(features, targets, num_clients, alpha=0.5):
    """
    Divide i dati tra i client secondo una distribuzione di Dirichlet sulle classi.
    
    :param features: numpy array delle feature
    :param targets: numpy array dei target
    :param num_clients: numero di client
    :param alpha: parametro della distribuzione di Dirichlet (maggiore -> più uniformità)
    :return: tuple con features e targets assegnati ai client
    """
    unique_classes = np.unique(targets)
    data_indices = {c: np.where(targets == c)[0] for c in unique_classes}
    
    # Genera le distribuzioni Dirichlet per ciascuna classe
    class_proportions = {c: np.random.dirichlet(alpha * np.ones(num_clients)) for c in unique_classes}
    
    # Assegna gli indici ai client secondo le proporzioni generate
    client_data_indices = defaultdict(list)
    
    for c, indices in data_indices.items():
        np.random.shuffle(indices)  # Mescola gli indici della classe
        proportions = (class_proportions[c] * len(indices)).astype(int)
        
        # Ensure the proportions sum to the correct number of samples
        proportions[-1] = len(indices) - np.sum(proportions[:-1])
        
        start = 0
        for client_id, count in enumerate(proportions):
            if count > 0:  # Only assign indices if count is positive
                client_data_indices[client_id].extend(indices[start:start+count])
                start += count
    
    # Crea il dataset per ciascun client
    features_split = []
    targets_split = []
    for client_id in range(num_clients):
        idx = np.array(client_data_indices[client_id], dtype=int)  # Ensure idx is an integer array
        if len(idx) > 0:  # Check if idx is not empty
            features_split.append(features[idx])
            targets_split.append(targets[idx].reshape(-1).tolist())
        else:
            features_split.append(np.array([]))  # Append empty array if idx is empty
            targets_split.append([])  # Append empty list if idx is empty
    
    return features_split, targets_split

def split_data(features: np.ndarray, target: np.ndarray, num_clients: int, rng: np.random.Generator, noniid: bool = False) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Wrapper function to select between uniform and Dirichlet-based data splitting.
    """
    if noniid:
        return split_data_dirichlet(features, target, num_clients, alpha=0.5)
    else:
        return split_data_uniform(features, target, num_clients, rng)
