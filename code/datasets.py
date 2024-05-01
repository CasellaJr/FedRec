import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import numpy as np

from torchvision import datasets, models, transforms

'''
ceilometer_dataset
    ----> train
          ---->true
          ---->false
    ----> test
          ---->true
          ---->false

70% train, 30% test
'''

# Image resize must be (224 * 224) because Resnet accepts input image size of (224 * 224)
transforms_train = transforms.Compose([
    transforms.Resize((224, 224)),   #must same as here
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(), # data augmentation
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # normalization
])
transforms_test = transforms.Compose([
    transforms.Resize((224, 224)),   # must same as here
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def import_data(dataset_name):

    root_dir = "../" + dataset_name + "/"

    # remember to modify placeholders!
    train_dir = root_dir + "train/"
    test_dir = root_dir +"test/"
    train_classa_dir = root_dir + "train/true/"
    train_classb_dir = root_dir + "train/false/"
    test_classa_dir = root_dir + "test/true/"
    test_classb_dir = root_dir + "test/false/"

    train_data = datasets.ImageFolder(train_dir, transforms_train)
    test_data = datasets.ImageFolder(test_dir, transforms_test)

    print('Train dataset size:', len(train_data))
    print('Test dataset size:', len(test_data))
    class_names = test_data.classes
    print('Class names:', class_names)

    num_classes = 2

    return num_classes, train_data, test_data