import torch.nn as nn
import torch
from torchvision import models

class MyModel(nn.Module):
    def __init__(self, num_classes):
        super(MyModel, self).__init__()
        self.model = models.resnet18() 
        self.num_features = self.model.fc.in_features     #extract fc layers features
        self.model.fc = nn.Linear(self.num_features, num_classes)

    def forward(self, x):
        # Forward pass through the ResNet model
        x = self.model(x)
        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(self.model.fc.weight)