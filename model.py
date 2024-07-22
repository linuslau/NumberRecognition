import torch  # Import PyTorch
from torch import nn  # Import the neural network module from PyTorch

# Define the neural network class, inheriting from nn.Module
class Network(nn.Module):
    def __init__(self):
        super().__init__()  # Call the initializer of the parent class nn.Module
        self.layer1 = nn.Linear(784, 256)  # Define the first linear layer (input size 784, output size 256)
        self.layer2 = nn.Linear(256, 10)  # Define the second linear layer (input size 256, output size 10)

    def forward(self, x):
        x = x.view(-1, 28*28)  # Flatten the input tensor to a 1D tensor of size 28*28
        x = self.layer1(x)  # Pass the input through the first linear layer
        x = torch.relu(x)  # Apply the ReLU activation function
        return self.layer2(x)  # Pass the result through the second linear layer and return it
