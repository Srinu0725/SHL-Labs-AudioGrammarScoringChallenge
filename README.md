# SHL-Labs-AudioGrammarScoringChallenge

## Overview
This project implements an end-to-end machine learning pipeline to evaluate spoken audio and predict grammar quality scores. It combines deep learning-based speech representations with traditional audio signal processing and ensemble learning techniques. The system is optimized for Pearson correlation, which is the primary evaluation metric in the challenge.

## Key Features
- Audio preprocessing including silence trimming and normalization  
- Deep feature extraction using :contentReference[oaicite:0]{index=0}  
- Handcrafted acoustic features such as MFCCs, spectral features, pitch, and energy  
- Dimensionality reduction using PCA  
- Ensemble learning with Ridge, LightGBM, SVR, and optional XGBoost  
- Model stacking using a meta-learner  
- Efficient processing using batching and caching  

## Pipeline Architecture

### 1. Environment Setup
- Import required libraries  
- Detect GPU (CUDA) availability  
- Set computation device (CPU/GPU)  
- Initialize random seeds for reproducibility  

### 2. Data Loading
- Traverse dataset directories  
- Identify train.csv and test.csv  
- Detect audio folders  
- Load data into pandas DataFrames  
- Automatically determine filename and label columns  

### 3. Audio Preprocessing
- Resolve correct file paths  
- Load audio at 16 kHz sampling rate  
- Trim silence from audio  
- Limit duration to a fixed length  
- Normalize audio signals  

### 4. Feature Extraction

#### Handcrafted Features
- MFCC and delta MFCC  
- Spectral centroid, bandwidth, rolloff  
- Zero-crossing rate and RMS energy  
- Chroma features  
- Pitch and speech rate  

#### Deep Features
- Extract embeddings using pretrained speech model  
- Use hidden states from all transformer layers  
- Apply weighted aggregation  
- Perform mean and standard deviation pooling  

### 5. Feature Combination
- Concatenate deep and handcrafted features  
- Create a unified feature vector per audio sample  

### 6. Optimization
- Batch processing for efficiency  
- Parallel audio loading  
- Feature caching to avoid recomputation  

### 7. Feature Scaling and PCA
- Replace NaN and infinite values  
- Apply StandardScaler  
- Reduce dimensionality using PCA  
- Retain maximum variance  

### 8. Model Training
- Train multiple models:
  - Ridge Regression  
  - LightGBM  
  - SVR  
  - Optional XGBoost  
- Use K-Fold cross-validation  
- Generate out-of-fold predictions  
- Evaluate using RMSE and Pearson correlation  

### 9. Model Stacking
- Combine predictions from base models  
- Train meta-model (Ridge)  
- Learn optimal weights for each model  

### 10. Final Prediction
- Generate predictions on test data  
- Clip predictions to valid range  
- Save submission file in CSV format  

## Evaluation Metric
The model is optimized for Pearson correlation between predicted and actual scores, ensuring strong alignment with ranking-based evaluation.

## Requirements
- Python 3.8+  
- PyTorch  
- Transformers  
- Librosa  
- NumPy, Pandas  
- Scikit-learn  
- LightGBM  
- XGBoost (optional)  

## How to Run
1. Place dataset in the input directory  
2. Update dataset paths if required  
3. Run the notebook step by step  
4. Generated submission will be saved as:
   `/kaggle/working/submission.csv`  

## Output
The final output is a CSV file containing predicted grammar scores for the test dataset, formatted as required by the competition.
