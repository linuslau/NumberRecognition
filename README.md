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
      - [Test result (97.77%)](#test-result-9777)
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

#### Test result (97.77%)
```
C:\Users\kz\.conda\envs\nn\python.exe C:\Users\kz\Documents\Code\NumberRecognition\model_inference.py 
test_dataset length:  10000
wrong case: predict = 9 actual = 0 img_path = ./mnist_test\0\mnist_test_126.png
wrong case: predict = 2 actual = 0 img_path = ./mnist_test\0\mnist_test_1748.png
wrong case: predict = 9 actual = 0 img_path = ./mnist_test\0\mnist_test_1987.png
wrong case: predict = 4 actual = 0 img_path = ./mnist_test\0\mnist_test_2033.png
wrong case: predict = 9 actual = 0 img_path = ./mnist_test\0\mnist_test_3251.png
wrong case: predict = 4 actual = 0 img_path = ./mnist_test\0\mnist_test_3818.png
wrong case: predict = 8 actual = 0 img_path = ./mnist_test\0\mnist_test_4065.png
wrong case: predict = 8 actual = 0 img_path = ./mnist_test\0\mnist_test_4880.png
wrong case: predict = 4 actual = 0 img_path = ./mnist_test\0\mnist_test_6400.png
wrong case: predict = 7 actual = 0 img_path = ./mnist_test\0\mnist_test_6597.png
wrong case: predict = 6 actual = 0 img_path = ./mnist_test\0\mnist_test_7216.png
wrong case: predict = 6 actual = 0 img_path = ./mnist_test\0\mnist_test_8325.png
wrong case: predict = 3 actual = 0 img_path = ./mnist_test\0\mnist_test_9634.png
wrong case: predict = 2 actual = 1 img_path = ./mnist_test\1\mnist_test_2182.png
wrong case: predict = 2 actual = 1 img_path = ./mnist_test\1\mnist_test_3073.png
wrong case: predict = 3 actual = 1 img_path = ./mnist_test\1\mnist_test_3906.png
wrong case: predict = 7 actual = 1 img_path = ./mnist_test\1\mnist_test_4201.png
wrong case: predict = 6 actual = 1 img_path = ./mnist_test\1\mnist_test_5331.png
wrong case: predict = 8 actual = 1 img_path = ./mnist_test\1\mnist_test_5457.png
wrong case: predict = 5 actual = 1 img_path = ./mnist_test\1\mnist_test_5642.png
wrong case: predict = 8 actual = 1 img_path = ./mnist_test\1\mnist_test_619.png
wrong case: predict = 6 actual = 1 img_path = ./mnist_test\1\mnist_test_6783.png
wrong case: predict = 0 actual = 1 img_path = ./mnist_test\1\mnist_test_7928.png
wrong case: predict = 8 actual = 1 img_path = ./mnist_test\1\mnist_test_8020.png
wrong case: predict = 2 actual = 1 img_path = ./mnist_test\1\mnist_test_956.png
wrong case: predict = 3 actual = 2 img_path = ./mnist_test\2\mnist_test_1395.png
wrong case: predict = 6 actual = 2 img_path = ./mnist_test\2\mnist_test_1609.png
wrong case: predict = 7 actual = 2 img_path = ./mnist_test\2\mnist_test_1790.png
wrong case: predict = 0 actual = 2 img_path = ./mnist_test\2\mnist_test_2098.png
wrong case: predict = 4 actual = 2 img_path = ./mnist_test\2\mnist_test_2488.png
wrong case: predict = 7 actual = 2 img_path = ./mnist_test\2\mnist_test_321.png
wrong case: predict = 8 actual = 2 img_path = ./mnist_test\2\mnist_test_3796.png
wrong case: predict = 4 actual = 2 img_path = ./mnist_test\2\mnist_test_3817.png
wrong case: predict = 8 actual = 2 img_path = ./mnist_test\2\mnist_test_4248.png
wrong case: predict = 7 actual = 2 img_path = ./mnist_test\2\mnist_test_4289.png
wrong case: predict = 4 actual = 2 img_path = ./mnist_test\2\mnist_test_4615.png
wrong case: predict = 4 actual = 2 img_path = ./mnist_test\2\mnist_test_4876.png
wrong case: predict = 4 actual = 2 img_path = ./mnist_test\2\mnist_test_5086.png
wrong case: predict = 7 actual = 2 img_path = ./mnist_test\2\mnist_test_583.png
wrong case: predict = 8 actual = 2 img_path = ./mnist_test\2\mnist_test_613.png
wrong case: predict = 6 actual = 2 img_path = ./mnist_test\2\mnist_test_6574.png
wrong case: predict = 1 actual = 2 img_path = ./mnist_test\2\mnist_test_659.png
wrong case: predict = 7 actual = 2 img_path = ./mnist_test\2\mnist_test_7457.png
wrong case: predict = 4 actual = 2 img_path = ./mnist_test\2\mnist_test_7886.png
wrong case: predict = 8 actual = 2 img_path = ./mnist_test\2\mnist_test_8094.png
wrong case: predict = 4 actual = 2 img_path = ./mnist_test\2\mnist_test_8198.png
wrong case: predict = 7 actual = 2 img_path = ./mnist_test\2\mnist_test_9664.png
wrong case: predict = 0 actual = 2 img_path = ./mnist_test\2\mnist_test_9768.png
wrong case: predict = 8 actual = 2 img_path = ./mnist_test\2\mnist_test_9811.png
wrong case: predict = 3 actual = 2 img_path = ./mnist_test\2\mnist_test_9839.png
wrong case: predict = 7 actual = 3 img_path = ./mnist_test\3\mnist_test_1128.png
wrong case: predict = 5 actual = 3 img_path = ./mnist_test\3\mnist_test_1531.png
wrong case: predict = 7 actual = 3 img_path = ./mnist_test\3\mnist_test_1681.png
wrong case: predict = 8 actual = 3 img_path = ./mnist_test\3\mnist_test_18.png
wrong case: predict = 9 actual = 3 img_path = ./mnist_test\3\mnist_test_2109.png
wrong case: predict = 9 actual = 3 img_path = ./mnist_test\3\mnist_test_2408.png
wrong case: predict = 5 actual = 3 img_path = ./mnist_test\3\mnist_test_2618.png
wrong case: predict = 2 actual = 3 img_path = ./mnist_test\3\mnist_test_2921.png
wrong case: predict = 2 actual = 3 img_path = ./mnist_test\3\mnist_test_2927.png
wrong case: predict = 7 actual = 3 img_path = ./mnist_test\3\mnist_test_381.png
wrong case: predict = 2 actual = 3 img_path = ./mnist_test\3\mnist_test_4437.png
wrong case: predict = 2 actual = 3 img_path = ./mnist_test\3\mnist_test_4443.png
wrong case: predict = 6 actual = 3 img_path = ./mnist_test\3\mnist_test_5078.png
wrong case: predict = 4 actual = 3 img_path = ./mnist_test\3\mnist_test_5140.png
wrong case: predict = 7 actual = 3 img_path = ./mnist_test\3\mnist_test_5734.png
wrong case: predict = 8 actual = 3 img_path = ./mnist_test\3\mnist_test_5955.png
wrong case: predict = 8 actual = 3 img_path = ./mnist_test\3\mnist_test_5973.png
wrong case: predict = 9 actual = 3 img_path = ./mnist_test\3\mnist_test_6009.png
wrong case: predict = 9 actual = 3 img_path = ./mnist_test\3\mnist_test_6011.png
wrong case: predict = 9 actual = 3 img_path = ./mnist_test\3\mnist_test_6023.png
wrong case: predict = 9 actual = 3 img_path = ./mnist_test\3\mnist_test_6045.png
wrong case: predict = 8 actual = 3 img_path = ./mnist_test\3\mnist_test_6046.png
wrong case: predict = 0 actual = 3 img_path = ./mnist_test\3\mnist_test_6059.png
wrong case: predict = 2 actual = 3 img_path = ./mnist_test\3\mnist_test_7800.png
wrong case: predict = 2 actual = 3 img_path = ./mnist_test\3\mnist_test_7821.png
wrong case: predict = 9 actual = 3 img_path = ./mnist_test\3\mnist_test_8246.png
wrong case: predict = 8 actual = 3 img_path = ./mnist_test\3\mnist_test_9944.png
wrong case: predict = 8 actual = 3 img_path = ./mnist_test\3\mnist_test_9975.png
wrong case: predict = 6 actual = 4 img_path = ./mnist_test\4\mnist_test_1112.png
wrong case: predict = 9 actual = 4 img_path = ./mnist_test\4\mnist_test_115.png
wrong case: predict = 9 actual = 4 img_path = ./mnist_test\4\mnist_test_1242.png
wrong case: predict = 9 actual = 4 img_path = ./mnist_test\4\mnist_test_2053.png
wrong case: predict = 2 actual = 4 img_path = ./mnist_test\4\mnist_test_247.png
wrong case: predict = 9 actual = 4 img_path = ./mnist_test\4\mnist_test_3490.png
wrong case: predict = 9 actual = 4 img_path = ./mnist_test\4\mnist_test_3718.png
wrong case: predict = 9 actual = 4 img_path = ./mnist_test\4\mnist_test_5936.png
wrong case: predict = 9 actual = 4 img_path = ./mnist_test\4\mnist_test_8527.png
wrong case: predict = 3 actual = 5 img_path = ./mnist_test\5\mnist_test_1003.png
wrong case: predict = 4 actual = 5 img_path = ./mnist_test\5\mnist_test_1272.png
wrong case: predict = 9 actual = 5 img_path = ./mnist_test\5\mnist_test_1289.png
wrong case: predict = 3 actual = 5 img_path = ./mnist_test\5\mnist_test_1393.png
wrong case: predict = 3 actual = 5 img_path = ./mnist_test\5\mnist_test_2035.png
wrong case: predict = 4 actual = 5 img_path = ./mnist_test\5\mnist_test_2040.png
wrong case: predict = 8 actual = 5 img_path = ./mnist_test\5\mnist_test_2369.png
wrong case: predict = 3 actual = 5 img_path = ./mnist_test\5\mnist_test_2597.png
wrong case: predict = 9 actual = 5 img_path = ./mnist_test\5\mnist_test_3117.png
wrong case: predict = 3 actual = 5 img_path = ./mnist_test\5\mnist_test_340.png
wrong case: predict = 0 actual = 5 img_path = ./mnist_test\5\mnist_test_3558.png
wrong case: predict = 8 actual = 5 img_path = ./mnist_test\5\mnist_test_3565.png
wrong case: predict = 3 actual = 5 img_path = ./mnist_test\5\mnist_test_3702.png
wrong case: predict = 8 actual = 5 img_path = ./mnist_test\5\mnist_test_3776.png
wrong case: predict = 3 actual = 5 img_path = ./mnist_test\5\mnist_test_4360.png
wrong case: predict = 3 actual = 5 img_path = ./mnist_test\5\mnist_test_5937.png
wrong case: predict = 3 actual = 5 img_path = ./mnist_test\5\mnist_test_5972.png
wrong case: predict = 9 actual = 5 img_path = ./mnist_test\5\mnist_test_5981.png
wrong case: predict = 9 actual = 5 img_path = ./mnist_test\5\mnist_test_5997.png
wrong case: predict = 4 actual = 5 img_path = ./mnist_test\5\mnist_test_6392.png
wrong case: predict = 8 actual = 5 img_path = ./mnist_test\5\mnist_test_720.png
wrong case: predict = 8 actual = 5 img_path = ./mnist_test\5\mnist_test_8062.png
wrong case: predict = 4 actual = 5 img_path = ./mnist_test\5\mnist_test_951.png
wrong case: predict = 6 actual = 5 img_path = ./mnist_test\5\mnist_test_9729.png
wrong case: predict = 6 actual = 5 img_path = ./mnist_test\5\mnist_test_9749.png
wrong case: predict = 0 actual = 5 img_path = ./mnist_test\5\mnist_test_9770.png
wrong case: predict = 5 actual = 6 img_path = ./mnist_test\6\mnist_test_1014.png
wrong case: predict = 8 actual = 6 img_path = ./mnist_test\6\mnist_test_1044.png
wrong case: predict = 8 actual = 6 img_path = ./mnist_test\6\mnist_test_1182.png
wrong case: predict = 4 actual = 6 img_path = ./mnist_test\6\mnist_test_1444.png
wrong case: predict = 7 actual = 6 img_path = ./mnist_test\6\mnist_test_1569.png
wrong case: predict = 4 actual = 6 img_path = ./mnist_test\6\mnist_test_1800.png
wrong case: predict = 9 actual = 6 img_path = ./mnist_test\6\mnist_test_1981.png
wrong case: predict = 5 actual = 6 img_path = ./mnist_test\6\mnist_test_1982.png
wrong case: predict = 4 actual = 6 img_path = ./mnist_test\6\mnist_test_2118.png
wrong case: predict = 5 actual = 6 img_path = ./mnist_test\6\mnist_test_217.png
wrong case: predict = 1 actual = 6 img_path = ./mnist_test\6\mnist_test_2654.png
wrong case: predict = 8 actual = 6 img_path = ./mnist_test\6\mnist_test_2995.png
wrong case: predict = 0 actual = 6 img_path = ./mnist_test\6\mnist_test_3422.png
wrong case: predict = 4 actual = 6 img_path = ./mnist_test\6\mnist_test_3520.png
wrong case: predict = 4 actual = 6 img_path = ./mnist_test\6\mnist_test_3749.png
wrong case: predict = 2 actual = 6 img_path = ./mnist_test\6\mnist_test_3853.png
wrong case: predict = 0 actual = 6 img_path = ./mnist_test\6\mnist_test_445.png
wrong case: predict = 5 actual = 6 img_path = ./mnist_test\6\mnist_test_4536.png
wrong case: predict = 4 actual = 6 img_path = ./mnist_test\6\mnist_test_4814.png
wrong case: predict = 4 actual = 6 img_path = ./mnist_test\6\mnist_test_5199.png
wrong case: predict = 2 actual = 6 img_path = ./mnist_test\6\mnist_test_6558.png
wrong case: predict = 4 actual = 6 img_path = ./mnist_test\6\mnist_test_6847.png
wrong case: predict = 4 actual = 6 img_path = ./mnist_test\6\mnist_test_8143.png
wrong case: predict = 4 actual = 6 img_path = ./mnist_test\6\mnist_test_8311.png
wrong case: predict = 0 actual = 6 img_path = ./mnist_test\6\mnist_test_965.png
wrong case: predict = 3 actual = 6 img_path = ./mnist_test\6\mnist_test_9679.png
wrong case: predict = 5 actual = 6 img_path = ./mnist_test\6\mnist_test_9782.png
wrong case: predict = 5 actual = 6 img_path = ./mnist_test\6\mnist_test_9793.png
wrong case: predict = 8 actual = 6 img_path = ./mnist_test\6\mnist_test_9858.png
wrong case: predict = 4 actual = 6 img_path = ./mnist_test\6\mnist_test_9940.png
wrong case: predict = 2 actual = 7 img_path = ./mnist_test\7\mnist_test_1039.png
wrong case: predict = 9 actual = 7 img_path = ./mnist_test\7\mnist_test_1194.png
wrong case: predict = 2 actual = 7 img_path = ./mnist_test\7\mnist_test_1226.png
wrong case: predict = 1 actual = 7 img_path = ./mnist_test\7\mnist_test_1260.png
wrong case: predict = 9 actual = 7 img_path = ./mnist_test\7\mnist_test_1328.png
wrong case: predict = 9 actual = 7 img_path = ./mnist_test\7\mnist_test_1496.png
wrong case: predict = 1 actual = 7 img_path = ./mnist_test\7\mnist_test_1500.png
wrong case: predict = 9 actual = 7 img_path = ./mnist_test\7\mnist_test_1522.png
wrong case: predict = 9 actual = 7 img_path = ./mnist_test\7\mnist_test_1581.png
wrong case: predict = 9 actual = 7 img_path = ./mnist_test\7\mnist_test_2024.png
wrong case: predict = 9 actual = 7 img_path = ./mnist_test\7\mnist_test_2070.png
wrong case: predict = 4 actual = 7 img_path = ./mnist_test\7\mnist_test_2607.png
wrong case: predict = 4 actual = 7 img_path = ./mnist_test\7\mnist_test_2730.png
wrong case: predict = 3 actual = 7 img_path = ./mnist_test\7\mnist_test_2915.png
wrong case: predict = 9 actual = 7 img_path = ./mnist_test\7\mnist_test_3333.png
wrong case: predict = 9 actual = 7 img_path = ./mnist_test\7\mnist_test_3451.png
wrong case: predict = 8 actual = 7 img_path = ./mnist_test\7\mnist_test_3808.png
wrong case: predict = 4 actual = 7 img_path = ./mnist_test\7\mnist_test_3838.png
wrong case: predict = 1 actual = 7 img_path = ./mnist_test\7\mnist_test_4027.png
wrong case: predict = 9 actual = 7 img_path = ./mnist_test\7\mnist_test_4199.png
wrong case: predict = 4 actual = 7 img_path = ./mnist_test\7\mnist_test_4966.png
wrong case: predict = 5 actual = 7 img_path = ./mnist_test\7\mnist_test_5887.png
wrong case: predict = 1 actual = 7 img_path = ./mnist_test\7\mnist_test_6576.png
wrong case: predict = 3 actual = 7 img_path = ./mnist_test\7\mnist_test_684.png
wrong case: predict = 4 actual = 7 img_path = ./mnist_test\7\mnist_test_7268.png
wrong case: predict = 2 actual = 7 img_path = ./mnist_test\7\mnist_test_9009.png
wrong case: predict = 2 actual = 7 img_path = ./mnist_test\7\mnist_test_9015.png
wrong case: predict = 2 actual = 7 img_path = ./mnist_test\7\mnist_test_9024.png
wrong case: predict = 3 actual = 8 img_path = ./mnist_test\8\mnist_test_1319.png
wrong case: predict = 7 actual = 8 img_path = ./mnist_test\8\mnist_test_1530.png
wrong case: predict = 3 actual = 8 img_path = ./mnist_test\8\mnist_test_1878.png
wrong case: predict = 3 actual = 8 img_path = ./mnist_test\8\mnist_test_2004.png
wrong case: predict = 9 actual = 8 img_path = ./mnist_test\8\mnist_test_3289.png
wrong case: predict = 5 actual = 8 img_path = ./mnist_test\8\mnist_test_3559.png
wrong case: predict = 5 actual = 8 img_path = ./mnist_test\8\mnist_test_3567.png
wrong case: predict = 0 actual = 8 img_path = ./mnist_test\8\mnist_test_3662.png
wrong case: predict = 4 actual = 8 img_path = ./mnist_test\8\mnist_test_3727.png
wrong case: predict = 3 actual = 8 img_path = ./mnist_test\8\mnist_test_4123.png
wrong case: predict = 7 actual = 8 img_path = ./mnist_test\8\mnist_test_4497.png
wrong case: predict = 4 actual = 8 img_path = ./mnist_test\8\mnist_test_4601.png
wrong case: predict = 9 actual = 8 img_path = ./mnist_test\8\mnist_test_4639.png
wrong case: predict = 7 actual = 8 img_path = ./mnist_test\8\mnist_test_4731.png
wrong case: predict = 0 actual = 8 img_path = ./mnist_test\8\mnist_test_4807.png
wrong case: predict = 2 actual = 8 img_path = ./mnist_test\8\mnist_test_495.png
wrong case: predict = 4 actual = 8 img_path = ./mnist_test\8\mnist_test_4956.png
wrong case: predict = 2 actual = 8 img_path = ./mnist_test\8\mnist_test_582.png
wrong case: predict = 3 actual = 8 img_path = ./mnist_test\8\mnist_test_6024.png
wrong case: predict = 9 actual = 8 img_path = ./mnist_test\8\mnist_test_6555.png
wrong case: predict = 2 actual = 8 img_path = ./mnist_test\8\mnist_test_6625.png
wrong case: predict = 9 actual = 8 img_path = ./mnist_test\8\mnist_test_6755.png
wrong case: predict = 4 actual = 8 img_path = ./mnist_test\8\mnist_test_691.png
wrong case: predict = 0 actual = 8 img_path = ./mnist_test\8\mnist_test_7921.png
wrong case: predict = 4 actual = 8 img_path = ./mnist_test\8\mnist_test_8279.png
wrong case: predict = 6 actual = 8 img_path = ./mnist_test\8\mnist_test_8522.png
wrong case: predict = 5 actual = 8 img_path = ./mnist_test\8\mnist_test_877.png
wrong case: predict = 5 actual = 8 img_path = ./mnist_test\8\mnist_test_9280.png
wrong case: predict = 9 actual = 8 img_path = ./mnist_test\8\mnist_test_947.png
wrong case: predict = 4 actual = 9 img_path = ./mnist_test\9\mnist_test_1232.png
wrong case: predict = 5 actual = 9 img_path = ./mnist_test\9\mnist_test_1247.png
wrong case: predict = 3 actual = 9 img_path = ./mnist_test\9\mnist_test_1553.png
wrong case: predict = 4 actual = 9 img_path = ./mnist_test\9\mnist_test_1901.png
wrong case: predict = 3 actual = 9 img_path = ./mnist_test\9\mnist_test_1952.png
wrong case: predict = 4 actual = 9 img_path = ./mnist_test\9\mnist_test_2293.png
wrong case: predict = 1 actual = 9 img_path = ./mnist_test\9\mnist_test_2387.png
wrong case: predict = 4 actual = 9 img_path = ./mnist_test\9\mnist_test_2414.png
wrong case: predict = 0 actual = 9 img_path = ./mnist_test\9\mnist_test_2648.png
wrong case: predict = 5 actual = 9 img_path = ./mnist_test\9\mnist_test_2939.png
wrong case: predict = 3 actual = 9 img_path = ./mnist_test\9\mnist_test_3060.png
wrong case: predict = 1 actual = 9 img_path = ./mnist_test\9\mnist_test_3503.png
wrong case: predict = 3 actual = 9 img_path = ./mnist_test\9\mnist_test_3926.png
wrong case: predict = 3 actual = 9 img_path = ./mnist_test\9\mnist_test_4078.png
wrong case: predict = 4 actual = 9 img_path = ./mnist_test\9\mnist_test_4154.png
wrong case: predict = 4 actual = 9 img_path = ./mnist_test\9\mnist_test_4369.png
wrong case: predict = 4 actual = 9 img_path = ./mnist_test\9\mnist_test_4425.png
wrong case: predict = 4 actual = 9 img_path = ./mnist_test\9\mnist_test_4823.png
wrong case: predict = 4 actual = 9 img_path = ./mnist_test\9\mnist_test_4918.png
wrong case: predict = 7 actual = 9 img_path = ./mnist_test\9\mnist_test_6571.png
wrong case: predict = 5 actual = 9 img_path = ./mnist_test\9\mnist_test_6608.png
wrong case: predict = 4 actual = 9 img_path = ./mnist_test\9\mnist_test_9587.png
wrong case: predict = 4 actual = 9 img_path = ./mnist_test\9\mnist_test_9808.png
test accuracy = 9777 / 10000 = 0.9777000000000000135003119794419

Process finished with exit code 0
```

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
The MNIST dataset is provided by Yann LeCun and can be found [here](https://yann.lecun.com/exdb/mnist/).
This project is built using PyTorch, an open-source machine learning library.
This description should give a comprehensive overview of your project and provide clear instructions for users on how to set it up and use it.