import os
import sys
import numpy as np
import pandas as pd
import math
import random
import time
import argparse
import tarfile
import torch
import torchvision
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
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
from server_aggregation import fed_avg
from datasets import *

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--rounds', type=int, default=100, help="Federated training rounds")
parser.add_argument('-e', '--epochs-per-round', type=int, default=1, help="Number of local steps of gradient descent before performing weighted aggregation.")
parser.add_argument('--benchmarking', action='store_true', help="Use the hyperparameters that Alessio found be the best for S.G.L.P. dataset")
parser.add_argument('--debug', action='store_true', help="Disable WandB metrics tracking")
args = parser.parse_args()

my_seeds = [0, 1, 2, 3, 4]
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
    
    if args.debug:
        wandb.init(mode="disabled")
    else:
        entity = "mlgroup"
        project = "SSFL"
        if args.benchmarking:
            run_name = f"SSFL_{myseed}_EpR{args.epochs_per_round}_BENCHMARKING"
        else:
            run_name = f"SSFL_{myseed}_EpR{args.epochs_per_round}"
        tags = ["FL", "SSL", "SSFL"]
        if run_name in run_names:
            print(f"Experiment {run_name} already executed.")
            continue
        else:
            wandb.init(project=project, entity=entity, group=f"{run_name}", name=run_name, tags=tags)

    class Client:
        def __init__(self, train_set, test_set):
            self.train_data = train_set
            self.test_data = test_set
            self.model = None
            self.optimizer = None
            self.train_loader = None
            self.test_loader = None
            
        def create_model(self):
            if num_classes == 2:
                self.model = MyModel(num_classes-1)
                self.model.initialize_weights()
            else:
                self.model = MLP(num_classes)
                self.model.initialize_weights()
                
        def create_optimizer(self):
            if args.benchmarking:
                self.optimizer = optim.SGD(self.model.parameters(), lr=1e-4, momentum=0.8, weight_decay=1e-5)
            else:
                self.optimizer = optim.Adam(self.model.parameters(), lr=0.001) 
                           

        def create_loaders(self):
            # Transform with Rocket
            c_train = self.train_data
            c_test = self.test_data

            train_loader = DataLoader(c_train, batch_size=8, num_workers=2, drop_last=True, shuffle=False, generator=generator)
            test_loader = DataLoader(c_test, batch_size=8, num_workers=2, drop_last=True, shuffle=False, generator=generator)
                
            self.train_loader = train_loader
            self.test_loader = test_loader 

        def fit_linear_model(self, num_client, round):
            # Extract features and then train the linear model
            self.create_loaders()
            # Define dictionary of loaders
            loaders = {"train": self.train_loader,
                        "test": self.test_loader}             
            if args.debug:             
                train(num_client, self.model, num_classes, loaders, self.optimizer, criterion, round, False, epochs=args.epochs_per_round, dev=dev)
            else:
                train(num_client, self.model, num_classes, loaders, self.optimizer, criterion, round, True, epochs=args.epochs_per_round, dev=dev)

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

    class Server:
        def __init__(self, clients: List[Client]):
            self.clients = clients

    #dev
    torch.cuda.is_available()
    dev = torch.device('cuda')

    #CREATION OF NUM_CLIENT CLIENTS AND OF A SERVER
    #import data
    sglp_num_classes, sglp_train_data, sglp_test_data = import_data("ceilometer_dataset1.1")
    cg_num_classes, cg_train_data, cg_test_data = import_data("CapoGranitola")
    if sglp_num_classes == cg_num_classes:
        num_classes = sglp_num_classes

    if num_classes == 2:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    #aggregated model
    if num_classes == 2:
        aggregated_model = MyModel(num_classes-1)
    else:
        aggregated_model = MyModel(num_classes)

    client_sglp = Client(sglp_train_data, sglp_test_data)
    client_cg = Client(cg_train_data, cg_test_data)
    clients = [client_sglp, client_cg]
    server = Server(clients)

    rounds = args.rounds
    PATHAGG = "state_dict_aggregated_model.pt"
    locally_training_aggregated_accuracy_f1 = []
    locally_tested_aggregated_accuracy_f1 = []

    for round in range(rounds):
        print("ROUND: ", round)
        for num_client, client in enumerate(clients):
            if round == 0:
                client.create_model()
            client.create_optimizer()
            client.fit_linear_model(num_client, round)

        models = []
        for client in clients:
            models.append(client.model)
        fed_avg(aggregated_model, *models) #average of parameters
        modelaggsd = torch.save(aggregated_model.state_dict(), PATHAGG)
        #loading the aggregated model parameters in the local models
        for client in clients:
            locally_training_aggregated_accuracy_f1.append(eval_model(num_classes, aggregated_model, client.train_loader))
            locally_tested_aggregated_accuracy_f1.append(eval_model(num_classes, aggregated_model, client.test_loader))
            client.model.load_state_dict(torch.load(PATHAGG))
        #testing the aggregated model
        tr_aggregated_accuracy = torch.stack([item[0] for item in locally_training_aggregated_accuracy_f1]).mean().item()
        aggregated_accuracy = torch.stack([item[0] for item in locally_tested_aggregated_accuracy_f1]).mean().item()
        aggregated_accuracy_client0 = locally_tested_aggregated_accuracy_f1[0][0].cpu().numpy()
        aggregated_accuracy_client1 = locally_tested_aggregated_accuracy_f1[1][0].cpu().numpy()
        tr_aggregated_f1 = torch.stack([item[1] for item in locally_training_aggregated_accuracy_f1]).mean().item()
        aggregated_f1 = torch.stack([item[1] for item in locally_tested_aggregated_accuracy_f1]).mean().item()
        aggregated_f1_client0 = locally_tested_aggregated_accuracy_f1[0][1].cpu().numpy()
        aggregated_f1_client1 = locally_tested_aggregated_accuracy_f1[1][1].cpu().numpy()
        print("Tr Aggregator accuracy: ", tr_aggregated_accuracy)
        print("Aggregator accuracy: ", aggregated_accuracy)
        print("Aggregator accuracy on SGLP: ", aggregated_accuracy_client0)
        print("Aggregator accuracy on CapoGranitola: ", aggregated_accuracy_client1)
        print("Tr Aggregator f1: ", tr_aggregated_f1)
        print("Aggregator f1: ", aggregated_f1)
        print("Aggregator f1 on SGLP: ", aggregated_f1_client0)
        print("Aggregator f1 on CapoGranitola: ", aggregated_f1_client1)
        wandb.log({f'Aggregated model tr. accuracy': tr_aggregated_accuracy, 
            f'Aggregated model accuracy': aggregated_accuracy,
            f'Aggregated model accuracy SGLP': aggregated_accuracy_client0,
            f'Aggregated model accuracy CapoGranitola': aggregated_accuracy_client1, 
            f'Aggregated model tr. f1': tr_aggregated_f1, 
            f'Aggregated model f1': aggregated_f1,
            f'Aggregated model f1 SGLP': aggregated_f1_client0,
            f'Aggregated model f1 CapoGranitola': aggregated_f1_client1})
        locally_training_aggregated_accuracy_f1 = []
        locally_tested_aggregated_accuracy_f1 = []
            

    wandb.finish()