# 🦴 Knee Osteoarthritis Classification via Radiomic Analysis

This project explores the classification of Knee Osteoarthritis (OA) stages using advanced radiomic features extracted from X-ray images.

## 📁 Repository Structure
* **Dataset_ARFF/**: Contains processed data in .arff format for Weka analysis.
* **ipynb_Scripts/**: Python notebooks for image pre-processing (CLAHE, ROI) and feature extraction.
* **Reports_and_Outputs/**: Statistical results and performance tables.

## 🔬 Methodology
* **Pre-processing:** Soft ROI and CLAHE enhancement.
* **Feature Extraction:** GLCM, LBP, FFT, and Discrete Wavelet Transform (DWT).
* **Classifiers:** Comparative study of Random Forest, SVM, MLP, and KNN.

## 📊 Performance Notes
* The project achieved a maximum accuracy of **~66%** using feature fusion and Random Forest.
* While Kappa values are moderate, the analysis provides deep insights into the discriminative power of specific radiomic features in medical imaging.
