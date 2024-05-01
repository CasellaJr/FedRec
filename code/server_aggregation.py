import torch
import torch.nn as nn

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