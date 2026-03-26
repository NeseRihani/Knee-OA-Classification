# -*- coding: utf-8 -*-
"""
BLG447 Görüntü İşleme – Final Projesi
Knee Osteoarthritis X-ray Texture Analysis
Amaç:
- Diz X-ray görüntülerinden doku tabanlı özellikler çıkarmak
- Osteoartrit (OA / no_OA) sınıflandırması için veri hazırlamak



Kullanılan Özellikler:
- GLCM (çoklu mesafe)
- LBP
- HOG
"""
# GEREKLİ KÜTÜPHANELER
import os
import cv2
import numpy as np
import pandas as pd
import pywt
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from sklearn.preprocessing import MinMaxScaler
from skimage.feature import hog



def map_kl_to_binary(folder_name):
    """
    Binary sınıflandırma:
    0  -> no_OA
    1,2,3,4 -> OA
    """
    if folder_name == "0":
        return "no_OA"
    else:
        return "OA"




#  SOFT ROI – GENİŞ MERKEZ
# Görüntünün merkezini alır
# Eklem bölgesini koruyup kenar gürültüsünü azaltır

def soft_roi(image):
    """
    Soft ROI:
    - Eklem bölgesini kesmez
    - Sadece kenarları temizler
    """
    h, w = image.shape[:2]

    y1 = int(h * 0.15)
    y2 = int(h * 0.90)
    x1 = int(w * 0.15)
    x2 = int(w * 0.85)

    return image[y1:y2, x1:x2]



#  PREPROCESSING
# - ROI uygula
# - Gri seviye
# - Gaussian blur
# - CLAHE (kontrast artırma)
# - Sabit boyut (128x128)

def preprocess_image(image):
    roi = soft_roi(image)

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    gray = cv2.resize(gray, (128, 128))
    return gray



#  GLCM QUANTIZATION
# Gri seviye değerlerini 32 seviyeye indirger
# GLCM hesaplamasını daha stabil hale getirir

def quantize(gray, levels=32):
    return (gray / 256 * levels).astype(np.uint8)



#  GLCM FEATURES
# Çoklu mesafe + çoklu açı kullanılır
# Her özellik için mesafeye göre ortalama alınır

def extract_glcm_features(gray):
    gray_q = quantize(gray, levels=32) # GLCM için gri seviye azaltımı

    distances = [64]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]

    features = {}

    glcm = graycomatrix(
        gray_q,
        distances=distances,
        angles=angles,
        levels=32,
        symmetric=True,
        normed=True
    )
    

    props = ["contrast", "dissimilarity", "homogeneity",
             "energy", "correlation", "ASM"]

    for d_idx, d in enumerate(distances):
        for prop in props:
            value = np.mean(graycoprops(glcm, prop)[d_idx, :])
            features[f"glcm_d{d}_{prop}"] = value

    return features



#  LBP (Local Binary Pattern)
# Lokal doku bilgisini yakalar
# Histogram normalize edilir

def extract_lbp_features(gray):
    features = {}
    for (P, R) in [(8, 1), (16, 2)]:
        lbp = local_binary_pattern(gray, P, R, method="uniform")
        n_bins = int(lbp.max() + 1)

        hist, _ = np.histogram(
            lbp, bins=n_bins, range=(0, n_bins), density=True
        )

        for i, v in enumerate(hist):
            features[f"lbp_P{P}_R{R}_{i}"] = v

    return features




# HOG (Histogram of Oriented Gradients)
# Kenar ve şekil bilgisini temsil eder

def extract_hog_features(gray):
    features = {}

    hog_features = hog(
        gray,
        orientations=9,          # yön sayısı
        pixels_per_cell=(8, 8),  # hücre boyutu
        cells_per_block=(2, 2),  # blok yapısı
        block_norm="L2-Hys",
        visualize=False,
        feature_vector=True
    )

    # HOG çok uzun bir vektör verdiği için özet istatistik alıyoruz
    features["hog_mean"] = np.mean(hog_features)
    features["hog_std"] = np.std(hog_features)
    features["hog_energy"] = np.sum(hog_features ** 2)

    return features


# DATASET OKUMA VE ÖZELLİK ÇIKARIMI
# train / val / test  klasörlerini gezer
# Tüm görüntülerden özellik çıkarır

def process_dataset(dataset_path):
    data = []
    splits = ["train", "val", "test"]

    for split in splits:
        split_path = os.path.join(dataset_path, split)
        if not os.path.isdir(split_path):
            continue
        for class_name in sorted(os.listdir(split_path)):
            class_folder = os.path.join(split_path, class_name)
            if not os.path.isdir(class_folder):
                continue

            mapped_class = map_kl_to_binary(class_name)


            if mapped_class is None:
                continue

            print(f"{split}/{class_name} -> {mapped_class}")
            for file_idx, file in enumerate(os.listdir(class_folder)):
                if file_idx % 50 == 0:
                    print(f"{split}/{class_name}: {file_idx} images processed")

                if file.lower().endswith((".jpg", ".png", ".jpeg", ".bmp", ".tiff")):
                    img_path = os.path.join(class_folder, file)
                    img = cv2.imread(img_path)
                    if img is None:
                        continue

                    gray = preprocess_image(img)

                    features = {}
                    features.update(extract_glcm_features(gray))
                    features.update(extract_lbp_features(gray))
                    features.update(extract_hog_features(gray))

                    features["class"] = mapped_class
                    
                    data.append(features)

    return pd.DataFrame(data)

# MIN-MAX NORMALIZATION
# Tüm sayısal özellikleri [0,1] aralığına çeker   
       
def apply_minmax_scaling(df):
    feature_cols = df.select_dtypes(include=[np.number]).columns
    scaler = MinMaxScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    return df



# ARFF DOSYASI KAYDETME
# WEKA uyumlu çıktı üretir

def save_to_arff(df, filename):
    with open(filename, "w") as f:
        f.write("@RELATION knee_OA_MULTICLASS_SOFT_ROI\n\n")


        for col in df.columns[:-1]:
            f.write(f"@ATTRIBUTE {col} NUMERIC\n")

        classes = ",".join(sorted(df["class"].unique()))
        f.write(f"@ATTRIBUTE class {{{classes}}}\n\n@DATA\n")

        for _, row in df.iterrows():
            f.write(",".join(map(str, row.values)) + "\n")

    print("✔ ARFF oluşturuldu:", filename)



# MAIN

if __name__ == "__main__":

    dataset_path = r"C:\Users\Asus\Downloads\Knee_Osteoarthritis_Dataset_with_Severity_Grading"

    df = process_dataset(dataset_path)
    df = apply_minmax_scaling(df)
    save_to_arff(df, "GLCM_LBP_HOGdeneme.arff")