# Assignment 3 - Using pretrained CNNs for image classification

## Contribution
The code in this assignment was developed in collaboration with my classmates.

## Description of the Assignment
For this assignment, we will be working with a dataset of Indo fashion images. The goal is to train a classifier using a pretrained Convolutional Neural Network (CNN) model, specifically VGG16. The code will save the training and validation history plots as well as the classification report.

## Data
The dataset used in this assignment is available on UCloud. Unfortunately, the dataset is around 3GB in size and cannot be uploaded to the repository. To access the dataset, follow these steps:
1. Visit the [Kaggle dataset](https://www.kaggle.com/myusername/indo-fashion-dataset).
2. Download the dataset from the provided link.
3. Save the dataset in a folder named "431824" on UCloud.

## Methods
The code uses TensorFlow and Keras to implement the image classification task. It loads the VGG16 model with pretrained weights and freezes the layers to avoid retraining. Additional classifier layers are added on top of the VGG16 model. The data is generated using TensorFlow's ImageDataGenerator and fed into the model for training. The model is compiled with a learning rate schedule and categorical cross-entropy loss. After training, the model predicts the labels for the test dataset, and a classification report is generated.

## Usage and Reproducibility

### Obtaining the Dataset
1. Find the Kaggle dataset [here](https://www.kaggle.com/myusername/indo-fashion-dataset).
2. Download the dataset and save it in the folder names "images".

### Running the Script
1. Clone this GitHub repository to your local machine.
2. Install the required packages by navigating to the root folder and running the following command in your terminal: `pip install -r requirements.txt`
3. Run the script using the following command: `python3 main.py`
4. The script will train the classifier, save the learning curve plots in the "out" folder, and generate a classification report in the "out" folder.

## Discussion of Results
The code was executed on a smaller subset of the data, which impacted the results negatively. Due to resource limitations on uCloud and time constraints, only a small fraction of the dataset was used for training and evaluation. As a result, the accuracy and performance of the model are lower compared to what could have been achieved with the entire dataset.
