# 📚 Kindle Review Sentiment Analysis
This project performs sentiment analysis on Kindle product reviews using machine learning. It classifies reviews as either positive or negative based on their content. The pipeline includes text preprocessing, feature extraction (Bag of Words & TF-IDF), and classification using Naive Bayes.

## 🔍 Project Overview
Dataset: Raw Kindle reviews with ratings.

Objective: Predict sentiment (positive/negative) from review text.

Model Used: Gaussian Naive Bayes

Feature Engineering: Bag of Words and TF-IDF

Evaluation Metrics: Accuracy, Confusion Matrix, Classification Report

## 🛠️ Features
End-to-end pipeline from raw data to classification

Text cleaning: lowercase, punctuation removal, stopword removal, lemmatization

Train-test split and vectorization

Comparison of BoW vs TF-IDF performance

Model evaluation using precision, recall, and F1-score

## 📊 Sample Results
Vectorizer	Accuracy
Bag of Words	~80%
TF-IDF	~78%

(Your exact values may vary depending on preprocessing and data)

## 🔮 Future Improvements
Hyperparameter tuning and cross-validation

Try additional models: Logistic Regression, SVM, etc.

Deploy as a web app using Flask or Streamlit

## 🧠 Skills Demonstrated
Text Preprocessing

Feature Engineering (BoW & TF-IDF)

Binary Classification

Model Evaluation
