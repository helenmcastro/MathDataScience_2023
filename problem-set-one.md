# Problem Set 1

## Code Link
Notebook:
https://colab.research.google.com/drive/1rL06nGb7KzOHRIUOcY48LRvAr-wq4IUb?usp=sharing


## Importing Libaries 
```python

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import datasets
from skimage.util import montage
import wandb as wb

# Define a function to plot images
def plot(x):
    if type(x) == torch.Tensor:
        x = x.cpu().detach().numpy()

    fig, ax = plt.subplots()
    im = ax.imshow(x, cmap='gray')
    ax.axis('off')
    fig.set_size_inches(7, 7)
    plt.show()
```
In this initial code block, I import the required libraries which include NumPy, Matplotlob, Torch, and Torchvision. I define a function called plot for displaying images. The plot function takes an image x as input, converts it to a NumPy array if it's a PyTorch tensor, and then displays the image using Matplotlib.

## Data Preparation 

```python
# Define a function to move data to the GPU
def GPU(data):
    return torch.tensor(data, requires_grad=True, dtype=torch.float, device=torch.device('cuda'))

# Define a function to move data to the GPU without requiring gradients
def GPU_data(data):
    return torch.tensor(data, requires_grad=False, dtype=torch.float, device=torch.device('cuda'))



# Load the MNIST dataset
train_set = datasets.MNIST('./data', train=True, download=True)
test_set = datasets.MNIST('./data', train=False, download=True)

# Extract data and labels from the dataset
X = train_set.data.numpy()
X_test = test_set.data.numpy()
Y = train_set.targets.numpy()
Y_test = test_set.targets.numpy()

# Normalize and reshape the data
X = X[:, None, :, :] / 255
X_test = X_test[:, None, :, :] / 255
```
In this code block, I define two functions, GPU and GPU_data, for moving data to the GPU with and without requiring gradients. I also load the MNIST dataset using torchvision and extract the data and labels, followed by normalizing and reshaping the data.

## Visualization 
```python
# Define a function to create and display a montage of images
def montage_plot(x):
    x = np.pad(x, pad_width=((0, 0), (1, 1), (1, 1)), mode='constant', constant_values=0)
    plot(montage(x))

# Display a montage of the first 25 images from the dataset
montage_plot(X[0:25, 0, :, :])
```
![Montage](https://github.com/helenmcastro/MathDataScience_2023/blob/main/montage-plot.png?raw=true) 
Here, I define the montage_plot function to create and display a montage of images using the montage function from skimage. Then, I use this function to display a montage of the first 25 images from the dataset.

## Reshaping
```python
# Reshape the input data
X = X.reshape(X.shape[0], 784)
X = X.T
X = GPU_data(X)
Y = GPU_data(Y)
X_test = GPU_data(X_test)
Y_test = GPU_data(Y_test)

# Check the shape of the reshaped data
X.shape
```
In these lines of code, the input data `X` is reshaped to have 784 features per sample.`X` is then transposed to make it suitable for matrix operations. The reshaped data is moved to the GPU using the GPU_data function. The shapes of the data are checked using `X.shape`.

## Subset of data
```python
# Select a subset of data (64 samples)
x = X[:, 0:64]

# Initialize a random linear model
M = GPU(np.random.rand(10, 784))

# Check the shapes of the model and input data
M.shape, x.shape

```
Here, a subset of the data `x` is selected, containing the first 64 samples.
A random linear model `M` with 10 output units and 784 input features is initialized.
The shapes of the model and input data are checked.

```python
# Perform matrix multiplication for classification
y = M @ x

# Find the predicted labels
y = torch.argmax(y, 0)

# Calculate accuracy
accuracy = torch.sum((y == Y[0:64])) / 64
accuracy
```
Matrix multiplication is performed between the model `M` and the input data `x`.
The predicted class labels are obtained using `torch.argmax`.
Accuracy is calculated by comparing the predicted labels to the ground truth labels `Y`.

##  Random models
```python
batch_size=64

# Initialize variables for the best model and score
M_Best = 0
Score_Best = 0

# Perform random matrix generation and classification
for i in range(100000):
    M_new = GPU(np.random.rand(10, 784))

    y = M_new @ x

    y = torch.argmax(y, 0)

    Score = (torch.sum((y == GPU_data(Y[0:batch_size]))) / batch_size).item()
    if Score > Score_Best:

        Score_Best = Score
        M_Best = M_new

        print(i,Score_Best)
0 0.109375
1 0.125
4 0.140625
5 0.171875
13 0.203125
20 0.21875
84 0.25
472 0.265625
2923 0.34375
75396 0.359375
```
This section initializes variables for tracking the best model `M_Best` and its corresponding score `Score_Best`.
A loop runs for 100,000 iterations, where random linear models `M_new` are generated. These models are used to make predictions on the subset of data `x`.
The predicted class labels are calculated and compared to the ground truth labels `Y`. The score, which represents accuracy, is calculated and compared to the previous best score. If it's better, the best score and best model are updated, and the progress is printed.

## Small steps added

```python
# Initialize variables for the best model and best accuracy
M_best = 0
accuracy_best = 0

# Loop for training
for i in range(100000):

    # Define the step size for updating the model
    step = 0.0000000001

    # Generate a random model perturbation
    m_random = GPU_data(np.random.randn(10, 784))

    # Update the model using the random perturbation
    m = M_best + step * m_random

    # Make predictions with the updated model
    y = m @ X

    # Find the predicted labels
    y = torch.argmax(y, axis=0)

    # Calculate accuracy on the entire dataset
    acc = ((y == Y)).sum() / len(Y)

    # Check if the current accuracy is better than the best accuracy
    if acc > accuracy_best:
        # Print and update the best accuracy
        print(acc.item())
        M_best = m
        accuracy_best = acc
```
I reached an accuracy of 81.6%.
In this code, I am training a model using random perturbations to improve accuracy.
The loop runs for 100,000 iterations, attempting to find a better model. M_best and accuracy_best are initialized to 0, indicating the best model and its accuracy have not been found yet. The loop iterates 100,000 times. Step is a small step size used to update the model. I am adding a random perturbation `m_random` to the current best model `M_best` to get a new candidate model `m`. The new candidate model `m` is used to make predictions on the entire dataset `X`. The predicted class labels are obtained using `torch.argmax`. Accuracy `acc` is calculated by comparing the predicted labels to the ground truth labels `Y`. If the current accuracy is better than the best accuracy found so far `accuracy_best`, it's printed, and `M_best` and `accuracy_best` are updated with the new model and accuracy.

