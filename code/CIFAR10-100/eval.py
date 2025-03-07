import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

# Dev
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def eval_model(num_classes, model, data_loader, dev=torch.device('cuda')):
    model = model.to(dev)
    model.eval()  # Set model to eval mode
    y_true = []
    y_pred = []

    with torch.no_grad():
        for data_inputs, data_labels in data_loader:
            data_inputs = data_inputs.to(dev)
            data_labels = data_labels.to(dev).long()
            preds = model(data_inputs)

            # Binarize predictions if binary classification
            if num_classes == 2:
                preds = preds.double()
                data_labels = data_labels.unsqueeze(1).float()
                pred_labels = (preds >= 0).long()
            else:
                _, pred_labels = preds.max(1)


            # Accumulate labels and predictions
            y_true.extend(data_labels.cpu().numpy())
            y_pred.extend(pred_labels.cpu().numpy())

    # Compute final metrics
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='binary' if num_classes == 2 else 'macro', zero_division=0)

    return torch.tensor(acc), torch.tensor(f1)