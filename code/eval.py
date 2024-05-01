import torch
import torch.nn as nn
import random
from sklearn.metrics import f1_score

#dev
torch.cuda.is_available()
dev = torch.device('cuda')

def eval_model(num_classes, model, data_loader):
    model = model.to(dev)
    model.eval() # Set model to eval mode
    true_preds, num_preds, sum_f1 = 0., 0., 0.
    y_true = []
    y_pred = []

    with torch.no_grad(): # Deactivate gradients for the following code
        for data_inputs, data_labels in data_loader:
            # Determine prediction of model on dev set
            data_inputs = data_inputs.to(dev)
            data_labels = data_labels.to(dev)
            preds = model(data_inputs)
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
            
            y_true.extend(data_labels.cpu().numpy())
            y_pred.extend(pred_labels.cpu().numpy())
            
            
        if num_classes == 2:
            sum_f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
        else:
            sum_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

            
    acc = true_preds / num_preds
    acc = 100.0*acc
    sum_f1 = torch.tensor(sum_f1*100)
    return acc, sum_f1
    
    
   