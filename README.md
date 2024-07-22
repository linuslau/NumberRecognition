# NumberRecognition

- [NumberRecognition](#numberrecognition)
  - [Quick start](#quick-start)
  - [MNIST Digit Recognition with PyTorch](#mnist-digit-recognition-with-pytorch)
    - [Overview](#overview)
    - [Contents](#contents)
      - [parse\_train\_images\_labels.py](#parse_train_images_labelspy)
      - [parse\_t10k\_images\_labels.py](#parse_t10k_images_labelspy)
      - [model\_train.py](#model_trainpy)
      - [model\_inference.py](#model_inferencepy)
      - [How to Use](#how-to-use)
      - [Dependencies](#dependencies)
      - [License](#license)
      - [Acknowledgements](#acknowledgements)

## Quick start
```
pip install -r requirements.txt
python parse_train_images_labels.py
python parse_t10k_images_labels.py
python model_training.py
python model_test.py
```


## MNIST Digit Recognition with PyTorch
This repository contains a complete implementation of a neural network for recognizing handwritten digits from the MNIST dataset using PyTorch. The project includes both training and inference scripts, along with utilities for preparing the dataset.

### Overview
The project consists of the following scripts:

- model_train.py: This script is responsible for training the neural network model on the MNIST dataset.
- model_inference.py: This script performs inference using the trained model on a test dataset.
  
### Contents
#### parse_train_images_labels.py
This script handles the following tasks:

- Reads the MNIST training images from the .idx3-ubyte file.
- Parses the images and saves them into the mnist_train directory, organized by digit labels.

#### parse_t10k_images_labels.py
This script performs the following tasks:

- Reads the MNIST test labels from the .idx1-ubyte file.
- Parses the labels and saves the corresponding images into the mnist_test directory, organized by digit labels.

#### model_train.py
This script handles the following tasks:

- Loads the training and test datasets from the ./mnist_train and ./mnist_test directories respectively.
- Applies transformations to the images, including converting them to grayscale and tensor format.
- Initializes and trains a neural network using the Adam optimizer and cross-entropy loss.
- Logs the progress of the training process, including dataset lengths, batch information, and loss values.
- Saves the trained model to mnist.pth.

#### model_inference.py
This script performs the following tasks:

- Loads the test dataset from the ./mnist_test directory and applies necessary transformations.
- Loads the pre-trained model from mnist.pth.
- Performs inference on the test dataset and prints cases where the model's predictions do not match the actual labels.
- Calculates and prints the accuracy of the model on the test dataset.

#### How to Use
1.  Dataset Preparation: Ensure the MNIST dataset is available in the ./mnist_train and ./mnist_test directories. The dataset should be organized such that each digit (0-9) has its own subdirectory containing corresponding images.

2.  Training the Model:

```
python model_train.py
```
This command will train the model and save the trained weights to mnist.pth.

3.  Running Inference:
```
python model_inference.py
```
This command will evaluate the trained model on the test dataset and print the accuracy along with any misclassified images.

#### Dependencies
- PyTorch
- torchvision
- PIL (Python Imaging Library)
- numpy
You can install the required dependencies using pip:
```
pip install torch torchvision pillow numpy
```

Project Structure
```
├── MNIST_data
│   ├── train-images-idx3-ubyte
│   ├── train-labels-idx1-ubyte
│   ├── t10k-images-idx3-ubyte
│   ├── t10k-labels-idx1-ubyte
├── mnist_train
│   ├── 0
│   ├── 1
│   ├── ...
│   ├── 9
├── mnist_test
│   ├── 0
│   ├── 1
│   ├── ...
│   ├── 9
├── model.py
├── model_inference.py
├── model_train.py
├── parse_t10k_images_labels.py
├── parse_train_images_labels.py
```

#### License
This project is licensed under the MIT License - see the LICENSE file for details.

#### Acknowledgements
The MNIST dataset is provided by Yann LeCun and can be found here.
This project is built using PyTorch, an open-source machine learning library.
This description should give a comprehensive overview of your project and provide clear instructions for users on how to set it up and use it.