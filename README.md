# Spam Detection AI

## Overview
A robust machine learning pipeline designed to classify text messages and emails as either "Spam" or "Ham" (legitimate). Built using Natural Language Processing (NLP) techniques, the system utilizes TF-IDF vectorization and a Logistic Regression classifier optimized specifically for imbalanced datasets.

## Key Features
* **Semantic Feature Extraction:** Utilizes TF-IDF (Term Frequency-Inverse Document Frequency) with n-gram extraction (1-2 words) to capture spam-specific phrasing rather than just isolated vocabulary.
* **Imbalanced Data Optimization:** Implements balanced class weighting during model training to prevent the model from skewing toward the majority class (~87% ham / ~13% spam).
* **Interactive Evaluation:** Features an interactive CLI for real-time text classification with confidence scoring.
* **Automated Data Visualization:** Automatically generates distribution charts and confusion matrix heatmaps using Matplotlib and Seaborn for immediate performance insight.

## Tech Stack
* Python
* Scikit-Learn (Logistic Regression, TF-IDF)
* Pandas & NumPy
* Matplotlib & Seaborn

## Installation & How to Run
1. Clone the repository:
   ```
   git clone [https://github.com/Zniniz/Spam-Detection-AI.git](https://github.com/Zniniz/Spam-Detection-AI.git)
   cd spam-detection-ai
   ```
2. Install the required dependencies:
   ```
   pip install pandas numpy scikit-learn matplotlib seaborn
   ```
3. Ensure the `spam.csv` dataset is located in the root directory.
4. Run the application
   ```
   python main.py
   ```
