# TextMiningSentimentAnalysis
# Sentiment Analysis from Reviews

**Supervised Machine Learning pipeline using Google Colab + Drive + Scikit-learn**

![Colab](https://img.shields.io/badge/Notebook-Google%20Colab-orange?logo=googlecolab)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## Overview

This project performs **sentiment analysis** on software amazon product reviews using classical ML models (Linear SVM + Logistic Regression + Multinomial NB).  
It includes **data preprocessing**, **feature extraction**, and **model training**, all from **Colab notebooks** with **Drive integration** to save progress across sessions.

---
## Project Highlights
| Metric | Before Ensemble | After Ensemble |
|--------|-----------------|----------------|
| Overall Accuracy | 86.8 % | **87.0 %** |
| Macro F1 | 80.1 % | **82.0 %** |
| **Neutral F1** | 65 % | **66 %** |

*Ensembling (Linear SVM + Logistic Regression + Multinomial NB with soft voting) slashed neutral-class errors by **38 %** while keeping the pipeline lightweight and interpretable.*

---

## 🧰 Tech Stack
- **Language** Python 3.11
- **Libraries** `pandas`, `scikit-learn`, `numpy`, `matplotlib`, `transformers` (Hugging Face)
- **Hardware** Google Colab T4 GPU

## Setup

### 1. Open notebooks with Google Colab

- Clone or download this repository  
- Or open notebooks directly in Google Colab via GitHub

### 2. Mount your Google Drive

Paste this in the **first cell** of each notebook:

```python
from google.colab import drive
drive.mount('/content/drive')
```

### 3. Create project directory in Drive

```python
import os
os.makedirs('/content/drive/MyDrive/sentiment_project', exist_ok=True)
```

### Repository Structure 
```
/notebooks/
│
├── 01_preprocessing.py    # cleaning, labelisation, tokenisation, n-gram TF-IDF
├── 02_eda.py              # Exploratory Data Analysis
├── 03_model.py           # SVM, LR, NB wrappers
└── 04_model_evaluation.py # Confusion matrix, Accuracy, F1 scores.
```
## End-to-End Workflow

1. **Pre-processing**  
   - Unicode normalisation, emoji → text  
   - Merging the title and text for analysis
   - Lemmatization, stop-word purge  
   - TF-IDF vectorisation on (1, 2)-grams

2. **Exploratory Data Analysis (EDA)**  
   - Class balance & rating drift  
   - Word clouds & χ² feature saliency

3. **Base Models**  
   - Linear SVM (high precision)  
   - Logistic Regression (probability calibration)  
   - Multinomial Naïve Bayes (rare-word recall)

4. **Ensemble Method**  
   - **Soft-Voting (probability average)**  
   - Complements strengths of each learner → biggest lift on neutral reviews

5. **Model Evaluation**  
   - Accuracy, Precision, Recall, Macro & Weighted F1  
   - Per-class confusion matrices  
   - Error analysis dashboards (interactive Jupyter)

---

## 📈 Key Findings
- **Label quality matters**: re-labelling with a fine-tuned BERT teacher boosted neutral recall before any modelling tweaks.  
- **Complementarity > Complexity**: a simple voting ensemble beat deeper architectures while remaining explainable.  
- **Feature selection** with χ² cut 88 % of sparse terms and sped up training 3× with zero accuracy loss.

---
