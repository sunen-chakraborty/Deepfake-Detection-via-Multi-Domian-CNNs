# Deepfake-Detection-via-Multi-Domian-CNNs
Code for the research paper **Enhancing Deepfake Detection through Multi-Domain Feature Fusion in Dual-Branch CNNs**

## Project Overview 
Code for facial Deepfake Detection using multi-domain features like Error Level Analysis (ELA), Noise Residuals from Spatial Rich Model (SRM), and Discrete Cosine Transform (DCT) using Dual-Branch Convolutional Neural Networks.

### Key Features
* **Dual-Branch CNNs:** CNNs with two branches followed by a feature concatenation layer where different branches extracts signs of deepfake from two different features.
* **Datasets:** All the models of our works was trained and evaluated on four prominent datasets:
  * VidTIMIT and DFTIMIT
  * Celeb-DF v2
  * FaceForensics++ c23
  * DFDC Preview

## Requirements
To run this project, you'll need to have the following libraries installed:
1. Clone this repository
   ```
   git clone https://github.com/sunen-chakraborty/Deepfake-Detection-via-Multi-Domian-CNNs.git
   ```
2. Install the required dependencies
   ```
   conda env create -f dfd.yml
   ```

## Dataset Preparation

### Step 1: Convert Videos into Frames
You need to extract frames from videos using the **face_extraction.py** script and generate the features using the **preprocessing.py** script.  

### Step 2: Dataset Structure

Organize your dataset in the following structure:

```
dataset_name/
  DCT/
    Real/
    Fake/
  ELA/
    Real/
    Fake/
  SRM/
    Real/
    Fake/
```

## Training and Testing the Model

Make sure you've set the paths to your dataset correctly.

### Train and Evaluation

Train your model and save the best model with minimum validation loss. You can load this model for further evaluation or inference.

## Results Visualization

The script includes a plot of training and validation accuracy, loss, and confusion matrix.

### Loss and Accuracy Plots

After training, the script generates plots for training, validation loss, and accuracy.
