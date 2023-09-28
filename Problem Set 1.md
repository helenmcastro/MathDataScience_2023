
```import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import datasets
from skimage.util import montage
!pip install wandb
import wandb as wb 
from skimage.io import imread



def plot(x):
    if type(x) == torch.Tensor:
        x = x.cpu().detach().numpy()

    fig, ax = plt.subplots()
    im = ax.imshow(x, cmap='gray')
    ax.axis('off')
    fig.set_size_inches(7, 7)
    plt.show()

def GPU(data):
    return torch.tensor(data, requires_grad=True, dtype=torch.float, device=torch.device('cuda'))

def GPU_data(data):
    return torch.tensor(data, requires_grad=False, dtype=torch.float, device=torch.device('cuda'))




train_set = datasets.MNIST('./data', train=True, download=True)
test_set = datasets.MNIST('./data', train=False, download=True)

X = train_set.data.numpy()
X_test = test_set.data.numpy()
Y = train_set.targets.numpy()
Y_test = test_set.targets.numpy()

X = X[:, None, :, :] / 255
X_test = X_test[:, None, :, :] / 255

def montage_plot(x):
    x = np.pad(x, pad_width=((0, 0), (1, 1), (1, 1)), mode='constant', constant_values=0)
    plot(montage(x))

montage_plot(X[0:25, 0, :, :])


M = GPU(np.random.rand(10, 784))
batch_size = 64
x = GPU_data(X[0:batch_size, 0, :, :])
x = x.reshape(784, batch_size)
y = M @ x

y = torch.argmax(y, 0)
accuracy = (torch.sum((y == GPU_data(Y[0:batch_size]))) / batch_size).item()
print(f"Accuracy of random linear model: {accuracy * 100:.2f}%")

M_Best = 0
Score_Best = 0.80  
max_iterations = 100000

for i in range(max_iterations):
    M_new = GPU(np.random.rand(10, 784))

    y = M_new @ x

    y = torch.argmax(y, 0)

    Score = (torch.sum((y == GPU_data(Y[0:batch_size]))) / batch_size).item()

    if Score > Score_Best:
        Score_Best = Score
        M_Best = M_new

    if Score_Best >= 0.80:
        print(f"Reached the target accuracy of {Score_Best * 100:.2f}%")
        break

print(f"Best accuracy achieved: {Score_Best * 100:.2f}%")```
