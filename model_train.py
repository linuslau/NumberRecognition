import torch
from torch import nn
from torch import optim
from model import Network
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader

if __name__ == '__main__':
    # Define the image transformations: convert to grayscale and then to tensor
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])

    # Load the training dataset from the specified directory and apply transformations
    train_dataset = datasets.ImageFolder(root='./mnist_train', transform=transform)
    # Load the test dataset from the specified directory and apply transformations
    test_dataset = datasets.ImageFolder(root='./mnist_test', transform=transform)
    # Print the length of the training dataset
    print("train_dataset length: ", len(train_dataset))
    # Print the length of the test dataset
    print("test_dataset length: ", len(test_dataset))

    # Create a DataLoader for the training dataset with batch size of 64 and shuffling enabled
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    # Print the number of batches in the training DataLoader
    print("train_loader length: ", len(train_loader))

    # Iterate over the first few batches of the training DataLoader
    for batch_idx, (data, label) in enumerate(train_loader):
        # Uncomment the following lines to break after 3 batches
        # if batch_idx == 3:
        #     break
        # Print the batch index
        print("batch_idx: ", batch_idx)
        # Print the shape of the data tensor
        print("data.shape: ", data.shape)
        # Print the shape of the label tensor
        print("label.shape: ", label.shape)
        # Print the labels
        print(label)

    # Initialize the neural network model
    model = Network()
    # Initialize the Adam optimizer with the model's parameters
    optimizer = optim.Adam(model.parameters())
    # Define the loss function as cross-entropy loss
    criterion = nn.CrossEntropyLoss()

    # Train the model for 10 epochs
    for epoch in range(10):
        # Iterate over the batches in the training DataLoader
        for batch_idx, (data, label) in enumerate(train_loader):
            # Forward pass: compute the model output
            output = model(data)
            # Compute the loss
            loss = criterion(output, label)
            # Backward pass: compute the gradients
            loss.backward()
            # Update the model parameters
            optimizer.step()
            # Zero the gradients for the next iteration
            optimizer.zero_grad()
            # Print the loss every 100 batches
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch + 1}/10 "
                      f"| Batch {batch_idx}/{len(train_loader)} "
                      f"| Loss: {loss.item():.4f}")

    # Save the trained model's state dictionary to a file
    torch.save(model.state_dict(), 'mnist.pth')
