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


def weighted_fed_avg(aggregated_net, *nets, num_labeled, exclude_fc=False):
    device = next(aggregated_net.parameters()).device  # Prende il device dal modello aggregato
    print(f"Aggregated model is on device: {device}")

    if len(nets) < 1:
        raise ValueError('At least one neural network should be provided.')
    
    models, importances = zip(*nets)  # Divide la tupla in due liste

    for model in models:
        model.to(device)

    # Determina il numero di reti da usare per la media
    num_nets = len(nets)

    important_nets = [model for model, important in zip(models, importances) if important]
    normal_nets = [model for model, important in zip(models, importances) if not important]
    
    num_important = len(important_nets)
    num_normal = len(normal_nets)
    
    # Definisce i pesi automaticamente
    if num_important > 0:
        weight_important = 0.7 / num_important  # 70% del peso totale
        weight_normal = 0.3 / num_normal if num_normal > 0 else 0  # 30% del peso totale
    else:
        weight_important = 0
        weight_normal = 1.0 / num_normal  # Se non ci sono reti importanti, usa solo quelle normali
    
    sum_of_weights = {name: torch.zeros_like(param.data, device=device) for name, param in aggregated_net.named_parameters()}
    
    for net in important_nets:
        local_params = dict(net.named_parameters())
        for name, param in local_params.items():
            if exclude_fc and 'fc' in name:
                continue  # Salta il classificatore se richiesto
            if name in sum_of_weights:
                sum_of_weights[name] += weight_important * param.data.to(device)
    
    for net in normal_nets:
        local_params = dict(net.named_parameters())
        for name, param in local_params.items():
            if exclude_fc and 'fc' in name:
                continue  # Salta il classificatore se richiesto
            if name in sum_of_weights:
                sum_of_weights[name] += weight_normal * param.data.to(device)
    
    for name in sum_of_weights:
        if 'fc' in name:
            sum_of_weights[name] /= num_labeled
    
    with torch.no_grad():
        for name, param in aggregated_net.named_parameters():
            if name in sum_of_weights:
                param.data.copy_(sum_of_weights[name])
    
    print("Aggregazione completata con pesi automatici basati sull'importanza.")

    # Check finale dei pesi
    if hasattr(aggregated_net, 'fc'):
        print("FINALE Pesi del classificatore:", aggregated_net.fc.weight)


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
