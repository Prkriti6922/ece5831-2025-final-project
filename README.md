# Automatic Image Colorization Using Deep Learning  
**ECE 5831 – Neural Networks / Deep Learning | Final Project**

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
👉 [Download Dataset (Google Drive)](https://drive.google.com/drive/u/2/folders/1Pb6eSOvGjF5iwpB2sYOf7LmYIRbc2Ij9)

---

## ▶️ How to Run the Project

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/ece5831-2025-final-project.git
cd ece5831-2025-final-project
---

2. Install Dependencies:
pip install tensorflow opencv-python matplotlib numpy

3. Open and run
final-project.ipynb

This notebook loads the pretrained model, performs inference on test images, and saves the results to the results/ directory.

## Evaluation
Because image colorization has no single ground-truth solution, evaluation is performed using both quantitative metrics and qualitative visual inspection.

**Metrics Used:**
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)

Quantitative metrics are computed by comparing the predicted colorized image with the original color image.
Qualitative evaluation compares grayscale input, predicted colorization, and original image side-by-side.


## Results

The model generates visually plausible colorizations
Structural details are well preserved
Common objects (sky, vegetation, skin tones) are colorized consistently
Some desaturation is observed, which is expected with MSE-based regression models
Sample outputs, training curves, and evaluation plots are saved in the results/ directory.

## Submission Links:
1. Pre recorded presentation video:
https://drive.google.com/drive/u/2/folders/1uhjc76yLx0WU4-tSu5xiNiTE2vr3IODq    

2. Presentation slides
https://docs.google.com/presentation/d/1XlHbCQUlbvBFj9wDvzVj4H3eAB5ZTk9G/edit?slide=id.p1#slide=id.p1

3. Final report
https://drive.google.com/drive/u/2/folders/1uhjc76yLx0WU4-tSu5xiNiTE2vr3IODq    

4. Dataset
https://drive.google.com/drive/u/2/folders/1Pb6eSOvGjF5iwpB2sYOf7LmYIRbc2Ij9

5. Demo Video
https://www.youtube.com/watch?v=RueoSGibCSY
