# TextMiningSentimentAnalysis
# 🧠 Sentiment Analysis from Reviews

**Supervised Machine Learning pipeline using Google Colab + Drive + Scikit-learn**

![Colab](https://img.shields.io/badge/Notebook-Google%20Colab-orange?logo=googlecolab)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## Overview

This project performs **sentiment analysis** on software amazon product reviews using classical ML models (SVM, Random Forest, KNN).  
It includes **data preprocessing**, **feature extraction**, and **model training**, all from **Colab notebooks** with **Drive integration** to save progress across sessions.

---

## Setup

### 1. Open notebooks with Google Colab

- Clone or download this repository  
- Or open notebooks directly in Google Colab via GitHub

### 2. Mount your Google Drive

Paste this in the **first cell** of each notebook:

```python
from google.colab import drive
drive.mount('/content/drive')

### 3. Create project directory in Drive

```python
import os
os.makedirs('/content/drive/MyDrive/sentiment_project', exist_ok=True)

### Folder Structure 
/MyDrive/sentiment_project/
│
├── processed_reviews.csv        # Cleaned and labeled dataset
├── svm_model.pkl                # SVM model (joblib)
├── rf_model.pkl                 # Random Forest model (joblib)
├── knn_model.pkl                # KNN model (joblib)
├── results.txt                  # Accuracy, F1 scores, etc.
└── plots/                       # Confusion matrix, wordclouds, etc.
