import torch
import torch.nn as nn
import copy

def fed_avg(aggregated_net, *nets):
    '''Init layer parameters by averaging weights of multiple neural networks.'''
    if len(nets) < 1:
        raise ValueError("At least one neural network should be provided.")

    sum_of_weights = None

    # Sum weight by weight
    for net in nets:
        # Fir net, initialize sum of weights
        if sum_of_weights is None:
            sum_of_weights = [param.data.clone() for param in net.parameters()]
        else:
            # Add weights to the variable
            for i, param in enumerate(net.parameters()):
                sum_of_weights[i] += param.data

    # Averaging weights
    num_nets = len(nets)
    avg_weights = [weights / num_nets for weights in sum_of_weights]

    # Aggregated network equal to the averaged weights
    with torch.no_grad():
        for p_out, avg_p in zip(aggregated_net.parameters(), avg_weights):
            p_out.data = nn.Parameter(avg_p)

def copy_weights(model1, model2):
    # Ensure both models are in the same device (e.g., CPU or GPU)
    device = next(model1.parameters()).device
    
    # Freeze the fully connected layer of model1
    if hasattr(model1.model, 'fc'):
        model1.model.fc.weight.requires_grad = False
        model1.model.fc.bias.requires_grad = False
    
    # Iterate through the layers of both models
    for (name1, param1), (name2, param2) in zip(model1.model.named_parameters(), model2.model.named_parameters()):
        # Skip the fully connected layer weights of model1
        if 'fc' in name1:
            continue
        
        # Copy the weights from model2 to model1
        param1.data.copy_(param2.data)

    if hasattr(model1.model, 'fc'):
        model1.model.fc.weight.requires_grad = False
        model1.model.fc.bias.requires_grad = False
    
    # Ensure model1 is back in its original device
    model1.to(device)