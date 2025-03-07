import torch
import torch.nn as nn
import copy
      
def fed_avg(aggregated_net, *nets, num_labeled, exclude_fc=False):
    device = next(aggregated_net.parameters()).device  # Prende il device dal modello aggregato
    print(f"Aggregated model is on device: {device}")

    if len(nets) < 1:
        raise ValueError('At least one neural network should be provided.')

    for model in nets:
        model.to(device)

    # Determina il numero di reti da usare per la media
    num_nets = len(nets)

    # Inizializza il dizionario con tensori sul device corretto
    sum_of_weights = {name: torch.zeros_like(param.data, device=device) for name, param in aggregated_net.named_parameters()}

    # Itera su tutti i client e accumula i pesi
    for net in nets:
        local_params = dict(net.named_parameters())
        for name, param in local_params.items():
            if exclude_fc and 'fc' in name:
                continue  # Salta il classificatore se richiesto
            if name in sum_of_weights:
                sum_of_weights[name] += param.data.to(device)

    # Calcola la media
    for name in sum_of_weights:
        if 'fc' in name:
            sum_of_weights[name] /= num_labeled
        else:
            sum_of_weights[name] /= num_nets

    # Copia i pesi aggregati nel modello finale
    with torch.no_grad():
        for name, param in aggregated_net.named_parameters():
            if name in sum_of_weights:
                param.data.copy_(sum_of_weights[name])

    print(f"Aggregated model is on device: {next(aggregated_net.parameters()).device}")

    # Check finale dei pesi
    if hasattr(aggregated_net, 'fc'):
        print("FINALE Pesi del classificatore:", aggregated_net.fc.weight)
    else:
        print("Questo modello non ha un classificatore.")





def copy_weights(model1, model2, exclude_fc=False):
    # Ensure both models are in the same device (e.g., CPU or GPU)
    device = next(model1.parameters()).device

    # Iterate through the layers of both models
    for (name1, param1), (name2, param2) in zip(model1.model.named_parameters(), model2.model.named_parameters()):
        if exclude_fc and 'fc' in name1:
            continue  # Salta il classificatore solo se il flag è True
        
        # Copia i pesi
        param1.data.copy_(param2.data)

    # Assicura che il modello torni sul device originale
    model1.to(device)

'''
def fed_avg(aggregated_net, *nets):
    #Init layer parameters by averaging weights of multiple neural networks.
    if len(nets) < 1:
        raise ValueError('At least one neural network should be provided.')

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
            p_out.data.copy_(avg_p)
'''