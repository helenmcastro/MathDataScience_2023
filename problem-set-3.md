# Flower Image Classification Report

This report documents the development of an image classification model using the AlexNet architecture fine-tuned on the Oxford-102 Flower dataset. Below, we present a step-by-step walkthrough of the process including data preprocessing, model modification, training, and evaluation.

## Data Preprocessing

The first step in our pipeline involves preparing our dataset for training. We accomplish this through a series of transformations and loading mechanisms provided by PyTorch's `torchvision` module.

```python
import os
import pandas as pd
from torchvision import datasets, transforms

# Directory and transforms
data_dir = '/content/flower_data/'

# Normalize mean and standard deviation for images
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Transformations for image preprocessing
data_transform = transforms.Compose([
   transforms.Resize(256),
   transforms.CenterCrop(224),
   transforms.ToTensor(),
   transforms.Normalize(mean=mean, std=std)
])

# Load the dataset using ImageFolder
dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transform)
```

The dataset is then loaded into a `DataLoader`, which allows us to iterate over the dataset in batches.

```python
from torch.utils.data import DataLoader

# Load the dataset into a DataLoader for batching
dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
```

### Plotting Function
A plotting function is defined to visualize the images from the dataset after transformations.

```python
import matplotlib.pyplot as plt

def plot(x, title=None):
    # ...
    # Code to plot the image
    # ...

# Example of plotting a single image
images, labels = next(iter(dataloader))
plot(images[0], dataset_labels[0])
```

## Model Setup

The pre-trained AlexNet model is loaded, and the final fully connected layer is modified to output the number of classes present in the Flower dataset (102 classes).

```python
import torch.nn as nn
from torchvision import models

# Load the pre-trained AlexNet model
alexnet = models.alexnet(pretrained=True)

# Modify the classifier for 102 classes
num_classes = 102
in_features = alexnet.classifier[6].in_features
alexnet.classifier[6] = nn.Linear(in_features, num_classes)
```

## Model Training

The training process involves setting up a criterion, an optimizer, and running the training loop.

```python
import torch.optim as optim
from tqdm.notebook import tqdm

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(alexnet.parameters(), lr=0.001, momentum=0.9)

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    # ...
    # Training steps
    # ...
```

During training, we track the accuracy and loss, printing them out at each epoch.

## Results

The results below indicate the performance of the model after the specified number of epochs.

```plaintext
Phase: train, Epoch: 1, Loss: 2.3028, Acc: 0.4412
Phase: valid, Epoch: 1, Loss: 1.1078, Acc: 0.6790
...
Phase: train, Epoch: 5, Loss: 0.8321, Acc: 0.7523
Phase: valid, Epoch: 5, Loss: 0.6254, Acc: 0.8102
```

## Image Classification Examples

Post-training, we use the trained model to predict a few examples from the test set.

```python
# Classify the images with AlexNet
for i in range(5):
    image = inputs[i:i+1].to(device)
    scores, class_idx = alexnet(image).max(1)
    print('Predicted class:', dataset_labels[class_idx.item()])
    plot(image[0], dataset_labels[labels[i].item()])
```

## Conclusion

This report has outlined the process of fine-tuning a pre-trained AlexNet model on the Oxford-102 Flower dataset. The steps taken from data preprocessing, model adjustments, to training and validation are aimed at achieving a high-accuracy classification model for flower images.

### Saving the Model

Finally, the trained model weights are saved for future use.

```python
torch.save(alexnet.state_dict(), 'flower_model.pth')
```
