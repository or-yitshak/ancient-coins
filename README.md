# Ancient Coin Image Classification
###### ![image](https://github.com/or-yitshak/RentIt/assets/77110578/0137510a-1532-4802-b6f6-0bbf310d3220)

##### Authors: Or Yitshak, Yahalom Chasid, Leead Jacobowitz
##### Collaboration with: Mr. Yaniv Levi and Profesor Lee-ad  Gotlib
---

### Table of Content
* [About The Project](#About-The-Project)
* [Dependencies](#Dependencies)
* [Usage](#Usage)
* [Coin Pictures](#Coin-Pictures)
* [License](#License)


---
### About The Project
This repository contains Python code for classifying coin images using various machine learning techniques. The goal of this project is to develop a classifier capable of accurately classifying various coins originating from the ancient Ptolemaic kingdom according to the issuing ruler, by considering their distinct and intricate patterns.


---

## Dependencies

- Pandas (`import pandas as pd`)
- Seaborn (`import seaborn as sns`)
- Matplotlib (`import matplotlib.pyplot as plt`)
- NumPy (`import numpy as np`)
- OpenCV (`import cv2`)
- Scikit-learn (`from sklearn.ensemble import RandomForestClassifier, from sklearn.metrics import classification_report, confusion_matrix`)
- TensorFlow (`import tensorflow as tf, from tensorflow.keras import datasets, layers, models`)

Please ensure that these libraries are installed in your Python environment.

---

## Usage

The code is divided into different functions and sections, each serving a specific purpose. Here's an overview of the main components:

1. Data Loading and Preprocessing:
   - `label_img(img)`: Function to assign labels to coin images based on their filenames.
   - `load_df(dir_name, size, grayscale)`: Function to load coin images from a directory, resize them, and create a DataFrame with image data and labels.
   - `train_test_split(data, samples_number, size, grayscale)`: Function to split the data into training and testing sets.

2. Coin Image Classification Models:
   - Model 1: Convolutional Neural Network (CNN) with multiple Conv2D and MaxPooling2D layers.
   - Model 2: ResNet50 pre-trained model.

3. Model Training and Evaluation:
   - Model training and evaluation code for each model.
   - Displaying classification reports and confusion matrices.

To use the code:
1. Ensure that the required libraries are installed.
2. Update the file paths (`dir_n`) and other parameters as needed.
3. Run the code sections individually or as a whole to train and evaluate the models.

Note: The code provided is a sample and may need modifications based on your specific use case and dataset.

---

## Coin Pictures

![image](https://github.com/or-yitshak/ancient-coins/assets/77110578/70386c6c-f1ae-4471-bee2-6172dcac723c)

![image](https://github.com/or-yitshak/ancient-coins/assets/77110578/b5c46a3c-fb91-4380-93ec-a216b2eca686)

![image](https://github.com/or-yitshak/ancient-coins/assets/77110578/22d7448a-8fcf-4b90-abfb-15aa3c978d73)

![image](https://github.com/or-yitshak/ancient-coins/assets/77110578/7b3ec57f-2d22-4192-ac71-067f8c9ec659)
---
## License

This project is licensed under the [MIT License](LICENSE).

Feel free to use and modify the code according to your needs.





