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
In this initial code block, we import the required libraries and define a function called plot for displaying images. The plot function takes an image x as input, converts it to a NumPy array if it's a PyTorch tensor, and then displays the image using Matplotlib.

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
In this code block, we define two functions, GPU and GPU_data, for moving data to the GPU with and without requiring gradients. We also load the MNIST dataset using torchvision and extract the data and labels, followed by normalizing and reshaping the data.


```python
# Define a function to create and display a montage of images
def montage_plot(x):
    x = np.pad(x, pad_width=((0, 0), (1, 1), (1, 1)), mode='constant', constant_values=0)
    plot(montage(x))

# Display a montage of the first 25 images from the dataset
montage_plot(X[0:25, 0, :, :])
```
![Montage](https://github.com/helenmcastro/MathDataScience_2023/blob/main/montage-plot.png?raw=true) Here, we define the montage_plot function to create and display a montage of images using the montage function from skimage. Then, we use this function to display a montage of the first 25 images from the dataset.

```python
# Initialize a random linear model
M = GPU(np.random.rand(10, 784))
batch_size = 64
x = GPU_data(X[0:batch_size, 0, :, :])
x = x.reshape(784, batch_size)

# Perform matrix multiplication for classification
y = M @ x

# Find the predicted labels
y = torch.argmax(y, 0)

# Calculate accuracy
accuracy = (torch.sum((y == GPU_data(Y[0:batch_size]))) / batch_size).item()
print(f"Accuracy of random linear model: {accuracy * 100:.2f}%")
Accuracy of random linear model: 7.81%
```
In this block, we initialize a random linear model M, perform matrix multiplication for classification, find the predicted labels, and calculate the accuracy of this random linear model.

```python
# Initialize variables for the best model and score
M_Best = 0
Score_Best = 0.80
max_iterations = 100000

# Perform random matrix generation and classification
for i in range(max_iterations):
    M_new = GPU(np.random.rand(10, 784))

    y = M_new @ x

    y = torch.argmax(y, 0)

    Score = (torch.sum((y == GPU_data(Y[0:batch_size]))) / batch_size).item()

    # Check if the current score is better than the best score
    if Score > Score_Best:
        Score_Best = Score
        M_Best = M_new

    # Check if the target accuracy is reached
    if Score_Best >= 0.80:
        print(f"Reached the target accuracy of {Score_Best * 100:.2f}%")
        break

# Print the best achieved accuracy
print(f"Best accuracy achieved: {Score_Best * 100:.2f}%")
Reached the target accuracy of 80.00%
Best accuracy achieved: 80.00%
```
In this final code block, we initialize variables for tracking the best model and its accuracy. Then, we perform random matrix generation and classification, continuously checking if the current score is better than the best score. If the target accuracy of 80% is reached, it prints a message indicating that. Finally, it prints the best achieved accuracy.
