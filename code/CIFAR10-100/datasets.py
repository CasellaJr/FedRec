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

def split_features_uniform(features, n_clients, rng):
    # Similar to split_data_uniform but only for features
    n_samples = len(features)
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    
    features_shuffled = features[indices]
    features_split = np.array_split(features_shuffled, n_clients)
    
    return features_split

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


def split_data_dirichlet(train_features, train_targets, val_features, val_targets, test_features, test_targets, num_clients, alpha=0.5, seed=None):
    """
    Suddivide i dati tra i client secondo una distribuzione di Dirichlet, mantenendo la stessa distribuzione tra train, val e test.
    
    :param train_features: array con le immagini di train
    :param train_targets: array con le etichette di train
    :param val_features: array con le immagini di validation
    :param val_targets: array con le etichette di validation
    :param test_features: array con le immagini di test
    :param test_targets: array con le etichette di test
    :param num_clients: numero di client nella federazione
    :param alpha: parametro della distribuzione di Dirichlet (controlla la non-IIDness)
    :param seed: seed per la riproducibilità
    :return: dizionario con i dati di ogni client per train, val e test
    """
    if seed is not None:
        np.random.seed(seed)
    
    num_classes = len(set(train_targets))
    client_data = {i: {'train': {'features': [], 'targets': []},
                        'val': {'features': [], 'targets': []},
                        'test': {'features': [], 'targets': []}} for i in range(num_clients)}
    
    # Raggruppa gli indici per classe per ciascun set
    def get_class_indices(features, targets):
        return {i: np.where(np.array(targets) == i)[0] for i in range(num_classes)}
    
    train_class_indices = get_class_indices(train_features, train_targets)
    val_class_indices = get_class_indices(val_features, val_targets)
    test_class_indices = get_class_indices(test_features, test_targets)
    
    # Genera una volta le proporzioni per ogni classe
    class_proportions = {c: np.random.dirichlet(alpha * np.ones(num_clients)) for c in range(num_classes)}
    
    # Funzione per assegnare dati ai client
    def assign_data(features, targets, class_indices, split_name):
        for c in range(num_classes):
            indices = class_indices[c]
            np.random.shuffle(indices)
            proportions = (np.cumsum(class_proportions[c]) * len(indices)).astype(int)[:-1]
            client_splits = np.split(indices, proportions)
            
            for i, split in enumerate(client_splits):
                client_data[i][split_name]['features'].extend(features[idx] for idx in split)
                client_data[i][split_name]['targets'].extend(targets[idx] for idx in split)
    
    # Assegna i dati mantenendo la coerenza
    assign_data(train_features, train_targets, train_class_indices, 'train')
    assign_data(val_features, val_targets, val_class_indices, 'val')
    assign_data(test_features, test_targets, test_class_indices, 'test')
    
    # Converti liste in array numpy
    for i in range(num_clients):
        for split in ['train', 'val', 'test']:
            client_data[i][split]['features'] = np.array(client_data[i][split]['features'])
            client_data[i][split]['targets'] = np.array(client_data[i][split]['targets'])
    
    return ([client_data[i]['train']['features'] for i in range(num_clients)],
            [client_data[i]['train']['targets'] for i in range(num_clients)],
            [client_data[i]['val']['features'] for i in range(num_clients)],
            [client_data[i]['val']['targets'] for i in range(num_clients)],
            [client_data[i]['test']['features'] for i in range(num_clients)],
            [client_data[i]['test']['targets'] for i in range(num_clients)])
