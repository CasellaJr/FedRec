import torch.nn as nn
import torch
import copy
from torchvision import models
from collections import OrderedDict

class MyModel(nn.Module):
    def __init__(self, num_classes):
        super(MyModel, self).__init__()
        self.num_classes = num_classes
        self.model = models.resnet18(pretrained=False, num_classes=self.num_classes) 
        self.num_features = self.model.fc.in_features  #extract fc layers features
        #self.model.fc = nn.Linear(self.num_features, num_classes)

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

class DenoisingResNet(nn.Module):
    def __init__(self):
        super(DenoisingResNet, self).__init__()
        # Load pre-trained ResNet-18 model
        resnet = models.resnet18(pretrained=False, num_classes = 100)
        num_features = resnet.fc.in_features     #extract fc layers features
        # Remove the classification layer
        self.model = nn.Sequential(OrderedDict([*(list(resnet.named_children())[:-1])]))
        # Define the decoder layers for upsampling
        self.decoder = nn.Sequential(
            nn.Upsample(size=(56, 56), mode='bilinear'),  # Upsample to match input size (56x56)
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(size=(112, 112), mode='bilinear'),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(size=(224, 224), mode='bilinear'),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Tanh(),  # Output range between -1 and 1
            nn.AdaptiveAvgPool2d((32, 32))
        )

    def forward(self, x):
        # Encode input image
        x = self.model(x)
        # Decode to reconstruct the image
        x = self.decoder(x)
        return x

class MyModelWithoutFC(nn.Module):
    def __init__(self, model):
        super(MyModelWithoutFC, self).__init__()
        self.model = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
            model.avgpool
        )

    def forward(self, x):
        x = self.model(x)
        return x