# Plant Disease Detection using CNNs

**CPS 847 – Toronto Metropolitan University**

This project implements a deep learning pipeline for classifying plant leaf diseases using the PlantVillage dataset. It follows the CPS847 project roadmap and includes EDA, baseline modeling, transfer learning with ResNet-50, and explainability using Grad-CAM.

---

## 📘 Project Overview

Plant diseases significantly affect global agriculture. This project trains and evaluates computer vision models to recognize leaf diseases from images.

**Main steps include:**
- Exploratory Data Analysis (EDA)
- Baseline CNN training
- Transfer learning (ResNet-50)
- Data augmentation
- Model explainability (Grad-CAM)
- Results and analysis for the final IEEE report

---

## 📁 Repository Structure

```
notebooks/            → EDA, baseline CNN, ResNet-50, Grad-CAM, results
src/                  → dataset loader, model definitions, training utilities
outputs/              → saved models, logs, generated figures
data/                 → PlantVillage dataset (ignored by Git)
requirements.txt      → Python dependencies
README.md             → Documentation
```

---

## 🌱 Dataset: PlantVillage

The PlantVillage dataset contains **~54,000 labeled images** of healthy and diseased leaves across **14 crop species**.

**Download the dataset from Kaggle:**  
[https://www.kaggle.com/datasets/mohitsingh1804/plantvillage](https://www.kaggle.com/datasets/mohitsingh1804/plantvillage)

Place the extracted dataset inside:
```
data/PlantVillage/
```

> **Note:** The dataset is excluded from version control using `.gitignore`.

---

## ▶️ How to Run the Project

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Ensure the dataset is located at:
```
data/PlantVillage/
```

### 3. Run notebooks in this order:

1. `01_eda.ipynb` – dataset exploration
2. `02_Data_Pipeline.ipynb` 
3. `03_Baseline_CNN.ipynb` 
4. `04_Transfer_Learning_ResNet18.ipynb` 
5. `05_Model_Evaluation.ipynb` 

---

## 🎯 Project Goal

To evaluate the effectiveness of CNN-based and transfer-learning-based models in detecting plant diseases from images, and to interpret model decisions using visual explainability tools.

---

## 📬 Contact

For questions or collaboration, feel free to reach out.
