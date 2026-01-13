# Taekwondo-Movement-Classification-Using-CNN-LSTM-Based-on-Body-Pose-Estimation
Final project: Classifying Taekwondo kicks (Apchagi, Idan Dollyo Chagi, Yeopchagi) from pose sequences using MediaPipe Pose and a CNN-LSTM model.

## Project Overview
This repository contains an end-to-end pipeline:
1. **Pose extraction** from videos using **MediaPipe Pose Landmarker (Tasks API)**.
2. **Feature construction** per frame:
   - Normalized pose landmarks: 33 landmarks × 4 values (x, y, z, visibility) → 132 features
   - Joint-angle features: 7 angles (radians) → 7 features  
   Total per-frame features: **139**
3. **Sequence modeling** using a **CNN-LSTM** classifier on sequences of **30 frames**.

## Classes
- `apchagi`
- `idandollyochagi`
- `yeopchagi`

## Data & Storage (Google Drive)
Large files (videos, pose outputs, trained models) are stored on Google Drive and are **not tracked** in this GitHub repository.

### Google Drive Folder
https://drive.google.com/drive/folders/1Z8Ho3aIu00EnSvFz7fkPO3qFkJ8O2xgY?usp=drive_link

### Drive Folder Structure
```
TA_Taekwondo/
├─ Data/
│  ├─ Training/                 # raw training videos
│  ├─ Testing/                  # raw testing videos
│  └─ MC1_Pose_Estimation_Output/
│     ├─ annotated_videos/      # annotated videos with skeleton overlay
│     ├─ landmarks_csv/         # per-video CSV landmarks + angles
│     └─ merged_per_class/      # merged videos per class (optional)
└─ Model/
   └─ MODEL_CNNLSTM_2LAYER.h5    # trained model checkpoint (or .keras)
```
Note: If the Drive link is restricted, please request access

Repository Contents
Model_TA.ipynb — main Colab notebook (pose extraction → dataset building → training → evaluation)
LICENSE — MIT License
TA_Taekwondo/Data/**/.gitkeep — placeholders to show the expected folder structure (no large files stored here)

Requirements
This project was developed and run in Google Colab. Main dependencies:
Python 3.x
TensorFlow / Keras
MediaPipe (Tasks API)
OpenCV
NumPy, Pandas, Matplotlib
scikit-learn

In Colab, most packages are available by default. If needed, install manually:
```
pip install mediapipe opencv-python scikit-learn pandas matplotlib
```
How to Run (Google Colab)
1. Open Model_TA.ipynb in Google Colab.
2. Mount Google Drive:
```
from google.colab import drive
drive.mount('/content/drive')
```
3. Ensure the Drive folders match the structure shown above (Training, Testing, outputs, and Model folder).
4. Run all notebook cells in order:
   - Export annotated videos + CSV landmarks
   - Apply data augmentation (rotation/flip/noise/time-shift)
   - Normalize features (z-score using training statistics)
   - Train CNN-LSTM and evaluate on the test set
```
### Model Configuration (CNN-LSTM)
Key settings used in the notebook:
1. Input shape: (30, 139)
2. Masking: mask_value = 0.0 (ignores padding)
3. Conv1D: 64 filters, kernel size 5, ReLU, padding="same"
4. Batch Normalization + MaxPooling1D (pool size 2)
5. LSTM layers: 64 units (return_sequences=True) → 32 units
6. Dense layer: 64 units (ReLU)
7. Dropout: 0.2
8. Output: Softmax (3 classes)
9. Optimizer: Adam (learning rate 1e-4)
10. Loss: Categorical Cross-Entropy
11. Batch size: 8, epochs up to 50, EarlyStopping on validation loss

### Outputs
The pipeline produces:
Annotated videos with skeleton overlay (annotated_videos/)
Landmark + angle CSV files (landmarks_csv/)
Optional merged annotated videos per class (merged_per_class/)
Trained model checkpoint saved under Model/


## License
This project is licensed under the MIT License. See the `LICENSE` file for details.
