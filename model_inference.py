from model import Network  # Import the custom neural network model class
from torchvision import transforms  # Import torchvision transformations
from torchvision import datasets  # Import torchvision datasets
import torch  # Import PyTorch

if __name__ == '__main__':
    # Define the image transformations: convert to grayscale and then to tensor
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])

    # Load the test dataset from the specified directory and apply transformations
    test_dataset = datasets.ImageFolder(root='./mnist_test', transform=transform)
    # Print the length of the test dataset
    print("test_dataset length: ", len(test_dataset))

    # Initialize the neural network model
    model = Network()
    # Load the model's state dictionary from the saved file
    model.load_state_dict(torch.load('mnist.pth'))

    right = 0  # Initialize a counter for correctly classified images

    # Iterate over the test dataset
    for i, (x, y) in enumerate(test_dataset):
        output = model(x.unsqueeze(0))  # Forward pass: add batch dimension and compute the model output
        predict = output.argmax(1).item()  # Get the index of the highest score as the predicted label
        if predict == y:
            right += 1  # Increment the counter if the prediction is correct
        else:
            img_path = test_dataset.samples[i][0]  # Get the path of the misclassified image
            # Print details of the misclassified case
            print(f"wrong case: predict = {predict} actual = {y} img_path = {img_path}")

    sample_num = len(test_dataset)  # Get the total number of samples in the test dataset
    acc = right * 1.0 / sample_num  # Calculate the accuracy as the ratio of correct predictions
    # Print the test accuracy
    print("test accuracy = %d / %d = %.31f" % (right, sample_num, acc))
