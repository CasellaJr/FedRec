import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import itertools
import random
import time
import argparse
import tarfile
import torch
import torchvision
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler
import wandb

from torchvision import utils
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from typing import List


# ignore all warnings
from warnings import simplefilter
simplefilter(action='ignore')

from training import *
from model import *
from eval import eval_model
from server_aggregation import *
from datasets import *

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--rounds', type=int, default=200, help="Federated training rounds")
parser.add_argument('-e', '--epochs-per-round', type=int, default=1, help="Number of local steps of gradient descent before performing weighted aggregation.")
parser.add_argument('-c', '--clients', type=int, default=2, help="Number of federation clients.")
parser.add_argument('--denoising', action='store_true', help="Exploit unlabeled datasets in a federated scenario by doing image denoising")
parser.add_argument('--personalization', action='store_true', help="Leave that each client trains its own classifier rather than averaging")
parser.add_argument('-w', '--weighted', action='store_true', help="Weighted average")
parser.add_argument('-n', '--noniid', action='store_true', help="Allocate proportion of samples of classes according to a Dirichlet distribution")
parser.add_argument('-a', '--alpha', type=float, default=0.5, help="Parameter of the Dirichlet distribution")
parser.add_argument('--debug', action='store_true', help="Disable WandB metrics tracking")
args = parser.parse_args()

my_seeds = [0, 1, 2, 3, 4]
n_clients = args.clients
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if args.debug:
    wandb.init(mode="disabled")
else:
    wandb.init()

# All the runs
run_names = []
runs = wandb.Api().runs("mlgroup/SSFL")
for run in runs:
    run_names.append(run.name)
print("RUN NAMES DONE: ", run_names)
wandb.finish()

for myseed in my_seeds:
    print("SEED: ", myseed)
    torch.manual_seed(myseed)
    np.random.seed(myseed)
    random.seed(myseed)
    generator=torch.Generator()
    generator.manual_seed(myseed)
    rng = np.random.RandomState(myseed)
    
    if args.debug:
        wandb.init(mode="disabled")
    else:
        entity = "mlgroup"
        project = "SSFL"
        if args.denoising and args.noniid and args.weighted:
            run_name = f"WFedRec_{myseed}_clients{args.clients}_Dirichlet_{args.alpha}"
        if args.denoising and args.noniid and args.weighted==False:
            run_name = f"FedRec_{myseed}_clients{args.clients}_Dirichlet_{args.alpha}"
        elif args.denoising and args.weighted and args.noniid==False:
            run_name = f"WFedRec_{myseed}_clients{args.clients}_EpR{args.epochs_per_round}"
        elif args.denoising and args.personalization:
            run_name = f"PersFedRec_{myseed}_clients{args.clients}_EpR{args.epochs_per_round}"
        elif args.denoising and args.noniid==False and args.weighted==False:
            run_name = f"FedRec_{myseed}_clients{args.clients}_EpR{args.epochs_per_round}"
        elif args.personalization:
            run_name = f"PersBaseline_{myseed}_clients{args.clients}_EpR{args.epochs_per_round}"
        elif args.denoising==False and args.noniid==False and args.weighted==False and args.personalization==False:
            run_name = f"Baseline_{myseed}_clients{args.clients}_EpR{args.epochs_per_round}"
        tags = ["FL", "SSFL"]
        if run_name in run_names:
            print(f"Experiment {run_name} already executed.")
            continue
        else:
            wandb.init(project=project, entity=entity, group=f"{run_name}", name=run_name, tags=tags)

    class Client:
        def __init__(self, train_set, val_set, test_set, labeled):
            self.train_data = train_set
            self.val_data = val_set
            self.test_data = test_set
            self.labeled = labeled
            self.model = None
            self.model_without_fc = None
            self.optimizer = None
            self.scheduler = None
            self.train_loader = None
            self.test_loader = None
            self.importance = True
            
        def create_model(self):
            if num_classes == 2:
                self.model = MyModel(num_classes-1)
                self.model.initialize_weights()
            else:
                self.model = MyModel(num_classes)
                self.model.initialize_weights()

        def create_model_without_fc(self):
            self.model_without_fc = MyModelWithoutFC(self.model.model)
                
        def create_optimizer(self):
            self.optimizer = optim.SGD(self.model.parameters(), lr = 0.1, momentum=0.9, weight_decay=0.0005)

        def create_scheduler(self):
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones = [60,120,160], gamma=0.2)
                           

        def create_loaders(self):
            c_train = self.train_data
            c_val = self.val_data
            c_test = self.test_data

            num_train = len(c_train)
            indices = list(range(num_train))
            valid_size = 0.1
            split = int(np.floor(valid_size * num_train))
            shuffle = True
            if shuffle:
                np.random.seed(0)
                np.random.shuffle(indices)

            train_idx, valid_idx = indices[split:], indices[:split]
            train_sampler = SubsetRandomSampler(train_idx)
            valid_sampler = SubsetRandomSampler(valid_idx)

            train_loader = DataLoader(c_train, batch_size=128, num_workers=2, sampler=train_sampler, drop_last=True, shuffle=False, generator=generator)
            val_loader = DataLoader(c_val, batch_size=128, num_workers=2, sampler=valid_sampler, drop_last=False, shuffle=False, generator=generator)
            test_loader = DataLoader(c_test, batch_size=128, num_workers=2, drop_last=False, shuffle=False, generator=generator)
                
            self.train_loader = train_loader
            self.val_loader = val_loader
            self.test_loader = test_loader 

        def fit(self, num_client, round, testing):
            # Extract features and then train the linear model
            self.epochs_per_round = 1 if testing else args.epochs_per_round
            self.create_loaders()
            # Define dictionary of loaders
            if testing == False:
                loaders = {"train": self.train_loader,
                            "val": self.val_loader,
                            "test": self.test_loader}         
            else:
                loaders = {"test": self.test_loader} 
            LOG_WANDB = not args.debug
            evaluation = train(num_client, self.model, num_classes, loaders, self.optimizer, self.scheduler, criterion, round, LOG_WANDB, epochs=self.epochs_per_round, 
                testing=True if testing else False, dev=dev)
            return evaluation

        def local_eval(self, num_client, local_loader):
            self.model.to(dev)
            self.model.eval() # Set model to eval mode
            true_preds, num_preds = 0., 0.
            with torch.no_grad(): # Deactivate gradients for the following code
                for data_inputs, data_labels in local_loader:
                    # Determine prediction of model on dev set
                    data_inputs = data_inputs.to(dev)
                    data_labels = data_labels.to(dev)
                    preds = self.model(data_inputs)
                    if num_classes == 2:
                        preds = preds.double()
                    data_labels = data_labels.type(torch.LongTensor)
                    data_labels = data_labels.to(dev)
                    if num_classes == 2:
                        data_labels = data_labels.unsqueeze(1)
                        data_labels = data_labels.float()
                    # Compute accuracy
                    if num_classes == 2:
                        pred_labels = (preds >= 0).long() # Binarize predictions to 0 and 1
                    else:
                        _,pred_labels = preds.max(1)
                    true_preds += (pred_labels == data_labels).sum()
                    num_preds += data_labels.shape[0]
            acc = true_preds / num_preds
            acc = 100.0*acc
            return acc

    class UnlabeledClient:
        def __init__(self, train_set, labeled):
            self.train_data = train_set
            self.labeled = labeled
            self.model = None
            self.model_without_fc = None
            self.optimizer = None
            self.scheduler = None
            self.train_loader = None
            self.importance = False
            
        def create_model(self):
            self.model = DenoisingResNet()

        def create_model_without_fc(self):
            self.model_without_fc = MyModelWithoutFC(self.model.model)

        def create_optimizer(self):
            self.optimizer = optim.SGD(self.model.parameters(), lr = 0.1, momentum=0.9, weight_decay=0.0005)

        def create_scheduler(self):
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones = [60,120,160], gamma=0.2)
                           
        def create_loaders(self):
            # Transform with Rocket
            c_train = self.train_data
            train_loader = DataLoader(c_train, batch_size=64, num_workers=2, drop_last=True, shuffle=False, generator=generator)
            self.train_loader = train_loader

        def fit(self, num_client, round, testing):
            # Extract features and then train the linear model
            self.create_loaders()           
            denoising(num_client, self.model, self.train_loader, self.optimizer, criterion_unlabeled, round, epochs=args.epochs_per_round, dev=dev)

    class Server:
        def __init__(self, clients: List[Client]):
            self.clients = clients

    def plot_label_distribution(client_targets, title):
        num_clients = len(client_targets)
        num_classes = len(np.unique(np.concatenate(client_targets)))

        class_counts = np.zeros((num_clients, num_classes))
        
        for i, targets in enumerate(client_targets):
            unique, counts = np.unique(targets, return_counts=True)
            class_counts[i, unique] = counts

        plt.figure(figsize=(10, 5))
        plt.imshow(class_counts, aspect="auto", cmap="Blues")
        plt.colorbar(label="Sample Count")
        plt.xlabel("Class")
        plt.ylabel("Client")
        plt.title(title)
        
        plt.savefig(title + ".png")  # Salva l'immagine
        print(f"Grafico salvato come {title}")


    def create_clients(train_dataset, val_dataset, test_dataset, n_clients, rng):
        # Preleviamo i dati dai dataset, assumendo che siano già array NumPy
        # Non è necessario usare .cpu() o .numpy() se sono già array NumPy
        train_features, train_target = train_dataset.data, train_dataset.targets
        val_features, val_target = val_dataset.data, val_dataset.targets
        test_features, test_target = test_dataset.data, test_dataset.targets

        # Utilizziamo la funzione split_data_uniform per ottenere i dati suddivisi tra i clienti
        
        if args.noniid:
            X_train_clients, Y_train_clients, X_val_clients, Y_val_clients, X_test_clients, Y_test_clients = split_data_dirichlet(
                train_features, train_target, 
                val_features, val_target, 
                test_features, test_target, 
                num_clients=n_clients, alpha=args.alpha, seed=myseed
            )
            plot_label_distribution(Y_train_clients, f"Label Distribution Among {n_clients} Clients (Dirichlet, alpha={args.alpha})")
        else:
            X_train_clients, Y_train_clients = split_data_uniform(train_features, train_target, n_clients, rng)
            X_val_clients, Y_val_clients = split_data_uniform(val_features, val_target, n_clients, rng)
            X_test_clients, Y_test_clients = split_data_uniform(test_features, test_target, n_clients, rng)
            plot_label_distribution(Y_train_clients, f"Label Distribution Among {n_clients} Clients (Uniform)")

        clients = []

        for i in range(n_clients):
            X_train_tensor = torch.from_numpy(X_train_clients[i]).permute(0, 3, 1, 2).float()
            Y_train_tensor = torch.tensor(Y_train_clients[i])
            client_train_data = torch.utils.data.TensorDataset(X_train_tensor, Y_train_tensor)

            X_val_tensor = torch.from_numpy(X_val_clients[i]).permute(0, 3, 1, 2).float()
            Y_val_tensor = torch.tensor(Y_val_clients[i])
            client_val_data = torch.utils.data.TensorDataset(X_val_tensor, Y_val_tensor)

            X_test_tensor = torch.from_numpy(X_test_clients[i]).permute(0, 3, 1, 2).float()
            Y_test_tensor = torch.tensor(Y_test_clients[i])
            client_test_data = torch.utils.data.TensorDataset(X_test_tensor, Y_test_tensor)

            client = Client(client_train_data, client_val_data, client_test_data, labeled="yes")
            clients.append(client)

        return clients
    
    def create_unlabeled_clients(train_dataset, n_clients, rng):
        # Get the data from the dataset (only features, no labels)
        train_features = train_dataset.dataset.data  # Access the underlying CIFAR data
        
        # Since it's unlabeled, we don't have Y values
        # We'll split only the features uniformly
        X_train_clients = split_features_uniform(train_features, n_clients, rng)
        
        unlabeled_clients = []
        
        for i in range(n_clients):
            X_train_tensor = torch.from_numpy(X_train_clients[i]).permute(0, 3, 1, 2).float()
            client = UnlabeledClient(X_train_tensor, labeled='no')
            unlabeled_clients.append(client)
        
        return unlabeled_clients

    #dev
    torch.cuda.is_available()
    dev = torch.device('cuda')

    #CREATION OF NUM_CLIENT CLIENTS AND OF A SERVER
    #import data
    cifar100_num_classes, cifar100_train_data, cifar100_val_data, cifar100_test_data = import_data("cifar100")
    num_classes = cifar100_num_classes

    if num_classes == 2:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    #aggregated model
    if num_classes == 2:
        aggregated_model = MyModel(num_classes-1)
        aggregated_model_withoutFC = MyModelWithoutFC(aggregated_model.model)
    else:
        aggregated_model = MyModel(num_classes)
        aggregated_model_withoutFC = MyModelWithoutFC(aggregated_model.model)

    #client_cifar100 = Client(cifar100_train_data, cifar100_val_data, cifar100_test_data, labeled='yes')
    # Use the create_clients function to create the clients
    cifar100_clients = create_clients(cifar100_train_data, cifar100_val_data, cifar100_test_data, n_clients, rng)

    if args.denoising:
        cifar10_train_data = import_unlabeled_data("cifar10")
        criterion_unlabeled = nn.MSELoss()
        client_cifar10 = create_unlabeled_clients(cifar10_train_data, n_clients, rng)
        
        #client_cifar10 = UnlabeledClient(cifar10_train_data, labeled='no')
        #clients = [client_cifar100, client_cifar10]
        clients = [cifar100_clients, client_cifar10]
        clients = list(itertools.chain(*clients))
        unlabeled_clients = client_cifar10
        
    else:
        #clients = [client_cifar100]
        clients = [cifar100_clients]
        clients = list(itertools.chain(*clients))

    #labeled_clients = [client_cifar100]
    labeled_clients = [cifar100_clients]
    labeled_clients = list(itertools.chain(*labeled_clients))
    n_labeled = len(labeled_clients)
    server = Server(clients)

    rounds = args.rounds
    PATHAGG = "state_dict_aggregated_model.pt"
    PATHAGG_NOFC = "state_dict_aggregated_model_withoutFC.pt"
    locally_training_aggregated_accuracy_f1 = []
    locally_tested_aggregated_accuracy_f1 = []

    for round in range(rounds):
        print("ROUND: ", round)
        for num_client, client in enumerate(clients):
            if round == 0:
                client.create_model()
            client.create_optimizer()
            client.create_scheduler()
            # Stampa i pesi solo se 'fc' è presente nel modello
            print("INIZIALE:", list(client.model.parameters())[0][0][0][0])
            
            if hasattr(client.model.model, 'fc'):  # Verifica se esiste il livello 'fc'
                print("INIZIALE Pesi del classificatore:", list(client.model.model.fc.parameters())[0])
            else:
                print("INIZIALE Pesi del classificatore: Questo modello non ha un classificatore.")
            
            client.fit(num_client, round, False)
            
            print("FINALE:", list(client.model.parameters())[0][0][0][0])
            
            if hasattr(client.model.model, 'fc'):  # Verifica se esiste il livello 'fc'
                print("FINALE Pesi del classificatore:", list(client.model.model.fc.parameters())[0])
            else:
                print("FINALE Pesi del classificatore: Questo modello non ha un classificatore.")

        models = []
        models_without_fc = []

        
        if args.weighted:
            models = [(client.model, client.importance) for client in clients]
        else:
            for client in clients:
                models.append(client.model)

        if args.denoising:
            if args.weighted:
                weighted_fed_avg(aggregated_model, *models, num_labeled=n_labeled, exclude_fc=False)
            else:
                fed_avg(aggregated_model, *models, num_labeled=n_labeled, exclude_fc=False)
            modelagg_sd = torch.save(aggregated_model.state_dict(), PATHAGG)
        else:
            if args.weighted:
                weighted_fed_avg(aggregated_model, *models, num_labeled=n_labeled, exclude_fc=False)
            else:
                fed_avg(aggregated_model, *models, num_labeled=n_labeled, exclude_fc=False)
            modelagg_sd = torch.save(aggregated_model.state_dict(), PATHAGG)

        for client in clients:
            if args.personalization:
                copy_weights(client.model, aggregated_model_withoutFC, exclude_fc=True)
            else:
                copy_weights(client.model, aggregated_model, exclude_fc=False)



        for num_client, client in enumerate(labeled_clients):
            test_local_eval = client.fit(num_client, round, True)
            locally_tested_aggregated_accuracy_f1.append(test_local_eval)
            #print("Client", num_client + 1, "accuracy", test_local_eval[0], "f1", test_local_eval[1])
            wandb.log({f'Client {num_client+1} accuracy': test_local_eval[0],
                f'Client {num_client+1} f1': test_local_eval[1],
                })
        if args.denoising:
            for client in unlabeled_clients:
                copy_weights(client.model, aggregated_model_withoutFC)
                #client.model.load_state_dict(torch.load(PATHAGG_NOFC))
        
        #testing the aggregated model
        aggregated_accuracy = torch.stack([item[0] for item in locally_tested_aggregated_accuracy_f1]).mean().item()
        aggregated_f1 = torch.stack([torch.tensor(item[1]) for item in locally_tested_aggregated_accuracy_f1]).mean().item()
        print("Aggregator accuracy: ", aggregated_accuracy)
        print("Aggregator f1: ", aggregated_f1)

        wandb.log({f'Aggregated model accuracy': aggregated_accuracy,
            f'Aggregated model f1': aggregated_f1,
            })
        locally_training_aggregated_accuracy_f1 = []
        locally_tested_aggregated_accuracy_f1 = []
            

    wandb.finish()