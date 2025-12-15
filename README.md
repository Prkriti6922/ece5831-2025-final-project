# Automatic Image Colorization Using Deep Learning  
**ECE 5831 – Neural Networks | Final Project**

---

## 📌 Project Overview
Automatic image colorization is the task of assigning plausible color information to grayscale images.  
This problem is inherently **ill-posed**, as a single grayscale image can correspond to multiple valid color interpretations.

In this project, a **deep learning–based image colorization pipeline** is implemented using a **convolutional neural network (CNN)**.  
The model learns to predict color information by leveraging semantic cues present in grayscale intensity patterns.

The approach is inspired by prior work on CNN-based image colorization and focuses on:
- Practical implementation of image-to-image translation using deep learning  
- Separation of luminance and chrominance using the LAB color space  
- Qualitative and quantitative evaluation of colorization performance  

---

## 🧠 Method Summary
- **Input:** Grayscale image (L channel)
- **Output:** Colorized image (predicted a and b channels)
- **Color Space:** LAB
- **Model:** Encoder–decoder CNN
- **Loss Function:** Mean Squared Error (MSE)
- **Optimizer:** Adam

---

## 📊 Dataset
- **Dataset:** COCO (Common Objects in Context)
- **Usage:** A subset of the dataset was used due to computational constraints
- **Content:** Diverse real-world scenes including people, animals, indoor and outdoor environments
- **Image Size:** All images resized to **128 × 128**
- **Split:** Separate training and testing sets

The diversity of the COCO dataset enables the model to learn generalized color patterns rather than overfitting to a specific scene type.

📁 **Dataset Link:**  
👉 [Download Dataset](https://cocodataset.org/), 
[Training Images](https://images.cocodataset.org/zips/train2017.zip),

[Validation Images](https://images.cocodataset.org/zips/val2017.zip)

---

## How to Run the Project

### 1. Clone the repository

git clone https://github.com/Prkriti6922/ece5831-2025-final-project.git


### 2. Install Dependencies:
pip install tensorflow opencv-python matplotlib numpy

### 3. Open and run
final-project.ipynb

This notebook loads the pretrained model, performs inference on test images, and saves the results to the results/ directory.

---

## Evaluation
Because image colorization has no single ground-truth solution, evaluation is performed using both quantitative metrics and qualitative visual inspection.

**Metrics Used:**
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)

Quantitative metrics are computed by comparing the predicted colorized image with the original color image.
Qualitative evaluation compares grayscale input, predicted colorization, and original image side-by-side.

---

## Results

The model generates visually plausible colorizations
Structural details are well preserved
Common objects (sky, vegetation, skin tones) are colorized consistently
Some desaturation is observed, which is expected with MSE-based regression models
Sample outputs, training curves, and evaluation plots are saved in the results/ directory.

---

## Submission Links:
## 🔗 Submission Links

- **Pre-recorded Presentation Video**  
  https://drive.google.com/drive/u/2/folders/1uhjc76yLx0WU4-tSu5xiNiTE2vr3IODq

- **Presentation Slides**  
  https://docs.google.com/presentation/d/1XlHbCQUlbvBFj9wDvzVj4H3eAB5ZTk9G/edit

- **Final Report**  
  https://drive.google.com/drive/u/2/folders/1uhjc76yLx0WU4-tSu5xiNiTE2vr3IODq

- **Demo Video**  
  https://www.youtube.com/watch?v=RueoSGibCSY

