# Predicting Lung Cancer Incidence from CT Imagery

**Bachelor Thesis — University of Science and Technology of Hanoi (USTH)**  
Department of Information and Communication Technology

| | |
|---|---|
| **Author** | Vu Hung Son |
| **Defense Date** | October 15, 2024 |

---

## Overview

This project investigates whether a **Three-Dimensional Convolutional Neural Network (3D CNN)** can directly predict lung cancer incidence from raw low-dose CT scan imagery, without relying on intermediate nodule detection steps. The pipeline covers end-to-end data preprocessing of DICOM files, model training, and performance evaluation on a subset of the National Lung Screening Trial (NLST) dataset.

Lung cancer is the leading cause of cancer death worldwide, with approximately 2.2 million new cases and 1.8 million deaths reported globally in 2020 (WHO). Early and accurate diagnosis is critical for improving patient survival rates, yet manual interpretation of CT scans is time-consuming, skill-dependent, and prone to both false positives and false negatives.

---

## Dataset

The dataset is a subset of the publicly available data provided by the **National Cancer Institute (NCI)** and the **National Lung Screening Trial (NLST)**. CT scans are stored in **DICOM** format.

| Property | Value |
|---|---|
| Total patients | 769 |
| Non-cancer cases (label 0) | 574 |
| Cancer cases (label 1) | 195 |
| Class imbalance ratio | ~74.6% / 25.4% |
| Raw image dimensions | 195 slices × 512 × 512 px |
| Raw voxel spacing | (2.5, 0.5, 0.5) mm |

> **Note:** The dataset is heavily imbalanced and represents a known challenge throughout this project.

---

## Methodology

### Preprocessing Pipeline

Raw DICOM data is transformed through four sequential steps before being fed into the model:

```
Non-isomorphic 3D DICOM data
         │
         ▼
┌─────────────────────────┐
│ Step 1: Load & Sort     │  Arrange slices by Z-position (ImagePositionPatient);
│         DICOM Slices    │  compute and inject missing SliceThickness metadata.
│  [-1024, 3071] HU       │  Output spacing: (2.5, 0.5, 0.5) mm — 195×512×512 px
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│ Step 2: Convert to      │  Apply RescaleSlope and RescaleIntercept from DICOM
│         Hounsfield      │  metadata: HU = Slope × pixel + Intercept
│         Units (HU)      │  Output: [-1000, 4000] HU — 195×512×512 px
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│ Step 3: Resample to     │  Nearest-neighbor interpolation to 1×1×1 mm voxels;
│         Isomorphic Form │  limit to 20 slices; resize each slice to 50×50 px.
│                         │  Output: [-1000, 4000] HU — 20×50×50 px
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│ Step 4: Normalize       │  Min-max normalization → [0, 1];
│                         │  zero-center by subtracting PIXEL_MEAN.
│                         │  Output: [0, 1] — 20×50×50 px
└─────────────────────────┘
```

### Model Architecture

A custom 3D CNN is constructed and trained end-to-end for binary classification (cancer vs. non-cancer).

| Layer | Configuration |
|---|---|
| Conv3D — 1 | 32 filters, kernel (3,3,3), ReLU activation, `same` padding |
| Max Pooling — 1 | Pool size (2,2,2), `same` padding |
| Conv3D — 2 | 64 filters, kernel (3,3,3), ReLU activation |
| Max Pooling — 2 | Pool size (2,2,2), `same` padding |
| Flatten | Converts 3D feature maps to a 1D vector |
| Fully Connected | 1024 neurons, ReLU activation |
| Dropout | Rate 0.2 |
| Output (Fully Connected) | 2 neurons, Softmax activation |

### Training Configuration

| Hyperparameter | Value |
|---|---|
| Input size per sample | 20 slices × 50 × 50 px |
| Number of classes | 2 (cancer / non-cancer) |
| Batch size | 10 |
| Epochs | 10 |
| Optimizer | Adam |
| Learning rate | 1e-3 |
| Loss function | Categorical cross-entropy |
| Dropout (keep rate) | 0.8 |
| Train / Test split | 90% / 10% |

Labels are one-hot encoded (e.g., non-cancer → `[1, 0]`, cancer → `[0, 1]`). The trained model is saved as `cnn3d_model.h5` and training history as `training_history.pkl`.

---

## Results

| Metric | Value |
|---|---|
| Accuracy | 0.68 |
| Sensitivity (Recall) | 0.0 |
| Specificity | 1.0 |
| Precision | 1.0 |
| AUC-ROC | 0.61 |

**Test set composition:** 68 non-cancer cases, 32 cancer cases.

The model is entirely biased toward predicting the majority class (non-cancer). While it achieves 68% accuracy, this figure reflects the class distribution rather than genuine discriminative ability. Sensitivity of 0.0 means all 32 cancer cases in the test set were missed. The AUC-ROC of 0.61 indicates marginal ability to distinguish between classes, only slightly above random chance (0.5).

---

## Discussion

The limited performance is attributed to several compounding factors:

- **Data imbalance:** The non-cancer class represents ~74.6% of samples. Without class balancing, the model learns to always predict non-cancer.
- **Low signal-to-noise ratio:** Detecting cancer directly from full-volume CT imagery — without prior segmentation of the lung region or candidate nodules — is an inherently difficult task, even for experienced radiologists.
- **No lung segmentation / ROI extraction:** Training on full-lung volumes introduces large amounts of irrelevant anatomical information that obscures the discriminative signal.
- **No data augmentation:** The training set was used as-is, limiting generalization and contributing to overfitting.
- **Suboptimal architecture and hyperparameters:** The chosen 3D CNN configuration and training settings were not extensively tuned for this task.

---

## Tools and Libraries

| Tool / Library | Purpose |
|---|---|
| Python | Primary programming language |
| TensorFlow / Keras | Model construction and training |
| NumPy | Array manipulation and numerical computation |
| Pandas | Tabular data and label management |
| Matplotlib | Data visualization |
| OpenCV | Image processing |
| SciPy | Scientific computing and resampling |
| scikit-learn | Evaluation metrics |
| pydicom | Reading and parsing DICOM files |

---

## Limitations

- The dataset (769 patients) is small relative to what is typically needed for robust deep learning in medical imaging.
- The model was not validated on an independent external dataset.
- Preprocessing steps may not account for all scanner-specific artifacts and variations present in real-world CT data.
- Hyperparameter search was not exhaustive.
- The study was conducted under time and computational resource constraints.

---

## Future Work

- **Lung segmentation:** Isolate the lung region prior to training to reduce noise and focus the model on clinically relevant anatomy.
- **Data augmentation:** Apply random cropping, rotation, and flipping to increase training set diversity and improve generalization.
- **Advanced architectures:** Explore 3D ResNet, DenseNet, or other architectures with demonstrated strength in volumetric medical image analysis.
- **Hyperparameter tuning:** Conduct a systematic search over model configuration and training settings.
- **Cancer subtype classification:** Extend the task from binary detection to multi-class lung cancer subtype identification.
- **Cancer stage prediction:** Develop models that predict disease stage, supporting treatment planning and prognosis.
- **Tumor segmentation:** Implement voxel-level segmentation of tumor regions to support surgical planning and radiotherapy.
- **Comprehensive diagnostic system:** Integrate imaging features with clinical metadata for a multi-modal decision support tool.

---

## References

1. World Health Organization. *Cancer*. 2020.
2. International Agency for Research on Cancer. *Globocan 2020*. 2020.
3. de Koning et al. "Reduced lung-cancer mortality with volume CT screening in a randomized trial." *New England Journal of Medicine*, 382(1), 2020.
4. Setio et al. "Pulmonary nodule detection in CT images: false positive reduction using multi-view convolutional networks." *IEEE Transactions on Medical Imaging*, 35(5), 2016.
5. Anthimopoulos et al. "Lung pattern classification for interstitial lung diseases using a deep convolutional neural network." *IEEE Transactions on Medical Imaging*, 35(5), 2016.
6. Jiang et al. "3D convolutional neural networks for lung nodule classification." *Journal of Healthcare Engineering*, 2019.
7. Nibali et al. "Pulmonary nodule classification with deep convolutional neural networks." *International Workshop on Pulmonary Image Analysis*, Springer, 2017.
8. NLST Research Team. "Reduced lung-cancer mortality with low-dose computed tomographic screening." *New England Journal of Medicine*, 365(5), 2011.

---

## License

This project was produced as a Bachelor Thesis at the University of Science and Technology of Hanoi (USTH). Please contact the author or supervisors before reusing any part of this work.
