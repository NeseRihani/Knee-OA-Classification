# 🦴 Knee Osteoarthritis Classification via Radiomic Analysis

This project explores the classification of Knee Osteoarthritis (OA) stages using advanced radiomic features extracted from X-ray images.

## 📁 Repository Structure
* **Dataset_ARFF/**: Contains processed data in .arff format for Weka analysis.
* **ipynb_Scripts/**: Python notebooks for image pre-processing (CLAHE, ROI) and feature extraction.
* **Reports_and_Outputs/**: Statistical results and performance tables.

## 🔬 Methodology
* **Pre-processing:** Soft ROI and CLAHE enhancement.
* **Feature Extraction:** GLCM, LBP, FFT, HOG, Gabor and Discrete Wavelet Transform (DWT).
* **Parameter Analysis:** Conducted detailed experiments using different GLCM displacement ($d$) values (e.g., $d=2, 16, 32, 64$) to identify the most discriminative texture scales
* **Classifiers:** Comparative study of 7 algorithms (Random Forest, SVM, MLP, KNN(3,5), J48, Adaboost+48.)

