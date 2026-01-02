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
Inside the folder **model_notebooks** there are two folders:
1. **proposed_models** contains notebook of models related to the training and evaluation of the proposed dual-branch CNNs.
2. **ablation_experiments** contains notebook of single-branch CNNs used for ablation experiments.

## Results Visualization

The script includes a plot of training and validation accuracy, loss, and confusion matrix.

### Loss and Accuracy Plots

After training, the script generates plots for training, validation loss, and accuracy.

### Citations for the datasets
1. **VidTIMIT:** Sanderson, C., Lovell, B.C.: Multi-region probabilistic histograms for robust and scalable identity inference. In: International Conference on Biometrics, pp. 199– 208 (2009). Springer
2. **DFTIMIT:** Korshunov, P., Marcel, S.: DeepFakes: a New Threat to Face Recognition? Assessment and Detection (2018). <https://arxiv.org/abs/1812.08685>
3. **Celeb-DF v2:** Li, Y., Yang, X., Sun, P., Qi, H., Lyu, S.: Celeb-df: A large-scale challenging dataset for deepfake forensics. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 3207–3216 (2020)
4. **FaceForesics++:** Rossler, A., Cozzolino, D., Verdoliva, L., Riess, C., Thies, J., Nießner, M.: Face- forensics++: Learning to detect manipulated facial images. In: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 1–11 (2019)
5. **DFDC Preview:** Dolhansky, B., Howes, R., Pflaum, B., Baram, N., Ferrer, C.C.: The DeepfakeDetection Challenge (DFDC) Preview Dataset (2019). <https://arxiv.org/abs/1910.08854>

## Repository Citation

If you use this repository in your research, please cite it as follows:
All researches that use this repository in your research or any part of it must cite it as follows:

BibTeX format:
```
@misc{Deepfake-Detection-via-Multi-Domian-CNNs,
  author = {Sunen Chakraborty},
  title = {Deepfake Detection via Multi Domian CNNs},
  year = {2025},
  url = {[https://github.com/sunen-chakraborty/Deepfake-Detection-via-Multi-Domian-CNNs]}
}
```
