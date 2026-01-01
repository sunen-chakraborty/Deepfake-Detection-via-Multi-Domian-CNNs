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
