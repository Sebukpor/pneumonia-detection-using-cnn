# Pneumonia Detection on Chest X-Ray using CNN and TFLite

## Overview

This project aims to develop a convolutional neural network (CNN) for detecting pneumonia from chest X-ray images. The model is built using TensorFlow and Keras, and it is trained on a dataset obtained from Kaggle. After training, the model is converted into TensorFlow Lite (TFLite) format for deployment on edge devices.

## Table of Contents

1. [What is Pneumonia?](#what-is-pneumonia)
2. [Project Structure](#project-structure)
3. [Dataset Description](#dataset-description)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Model Training](#model-training)
7. [Model Evaluation](#model-evaluation)
8. [Convert to TFLite](#convert-to-tflite)
9. [Results](#results)
10. [Contributors](#contributors)
11. [License](#license)

## What is Pneumonia?

Pneumonia is an inflammatory condition of the lung affecting primarily the small air sacs known as alveoli. Symptoms typically include a combination of productive or dry cough, chest pain, fever, and difficulty breathing. The severity of the condition varies and it is usually caused by infection with viruses or bacteria.


![image](https://github.com/user-attachments/assets/4dec3415-5856-43c6-8881-008344590faa)

## Project Structure

- `Pneumonia_Detection_Using_Chest_X_Ray.ipynb`: The main Jupyter notebook contains all steps for data loading, preprocessing, model training, evaluation, and conversion to TFLite.
- `model/`: Directory where the trained model and TFLite model will be saved.
- `data/`: Directory containing the chest X-ray dataset (not included in the repository; see Dataset Description for instructions on downloading).
- `README.md`: Project overview and instructions.

## Dataset Description

The dataset is organized into 3 folders (train, test, val) and contains subfolders for each image category (Pneumonia/Normal). There are 5,863 X-Ray images (JPEG) and 2 categories (Pneumonia/Normal). Chest X-ray images (anterior-posterior) were selected from retrospective cohorts of pediatric patients of one to five years old from Guangzhou Women and Children’s Medical Center, Guangzhou. All chest X-ray imaging was performed as part of patients’ routine clinical care. For the analysis of chest x-ray images, all chest radiographs were initially screened for quality control by removing all low-quality or unreadable scans. Two expert physicians then graded the diagnoses for the pictures before being cleared for training the AI system. In order to account for any grading errors, a third expert also checked the evaluation set.

- **Normal**: X-ray images of healthy lungs.
- **Pneumonia**: X-ray images showing signs of pneumonia.

You can download the dataset [here](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia).

## Installation

To run this project, you'll need to have Python installed along with several libraries. You can install the required libraries using:

```bash
pip install -r requirements.txt
```

Note: You need to create the `requirements.txt` file with the necessary dependencies like TensorFlow, Keras, numpy, pandas, matplotlib, etc.

## Usage

1. **Clone the repository:**

```bash
git clone https://github.com/sebukpor/pneumonia-detection-using-cnn.git
cd pneumonia-detection-using-cnn
```

2. **Download the dataset:**
   - Download the Chest X-ray dataset from Kaggle.
   - Place the dataset in the `data/` directory.

3. **Run the notebook:**
   - Open the `Pneumonia_Detection_Using_Chest_X_Ray.ipynb` notebook.
   - Execute the cells step by step to train the model and evaluate it.

## Model Training

The model is a convolutional neural network built with TensorFlow and Keras. It uses several convolutional layers, max-pooling layers, and dense layers to classify X-ray images into Normal or Pneumonia categories.

Key steps in training:
- Data augmentation using `ImageDataGenerator`.
- Model architecture definition.
- Model training and validation.
- Saving the trained model.

## Model Evaluation

The model's performance is evaluated using accuracy and confusion matrix on the test dataset. Visualizations of the model's performance are provided in the notebook.

## Convert to TFLite

After training, the model is converted into TFLite format to be deployed on mobile and edge devices.

## Results

- The trained CNN model achieved an accuracy of X% on the test set.
- The confusion matrix shows that the model correctly classifies Y% of the pneumonia cases.

## Contributors

- **Divine Sebukpor** - [Your GitHub Profile](https://github.com/sebukpor)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

You can customize the placeholders and add more details as per your project specifics.
