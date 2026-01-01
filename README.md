# News-Article-Classification
Applied NLP techniques to classify news articles using TF-IDF features. Trained and evaluated multiple classifiers and explored topic modeling to uncover hidden themes—essentially teaching the model to tell sports from politics without reading the whole newspaper.

📰 News Article Classification & Topic Modeling
Introduction

With the growing volume of digital news, manually organizing articles into categories is no longer practical. This project applies Natural Language Processing (NLP) techniques to automatically classify news articles and uncover underlying themes within large text corpora.

In simple terms, the goal was to teach a machine learning model to tell politics from sports and technology—without having to read the entire newspaper every morning.

Problem Statement

The objective of this project is to:

Classify news articles into predefined categories

Identify hidden themes within the text using topic modeling

Compare multiple machine learning models for text classification

Dataset

The dataset consists of thousands of news articles spanning multiple categories. Each article includes raw text and an associated label, making it suitable for both supervised classification and unsupervised topic discovery tasks.

Methodology
Text Preprocessing

Text cleaning and normalization

Stopword removal

Tokenization

Feature Engineering

TF-IDF vectorization to capture word importance across documents

Model Training

Logistic Regression

Naive Bayes

Support Vector Machines

Topic Modeling

Latent Dirichlet Allocation (LDA) to extract underlying topics from the corpus

Evaluation Metrics

Accuracy

Precision

Recall

F1-score

These metrics were used to compare model performance and ensure reliable classification results.

Results

The trained models achieved strong classification performance using TF-IDF features. Topic modeling revealed coherent and interpretable themes aligned with real-world news categories, validating the effectiveness of the preprocessing and feature extraction pipeline.

Key Takeaways

TF-IDF remains a strong baseline for text classification

Preprocessing quality significantly impacts model performance

Topic modeling adds valuable insights beyond supervised labels

Future Enhancements

Incorporate word embeddings such as Word2Vec or GloVe

Experiment with transformer-based models like BERT

Improve topic visualization and interpretability

Tech Stack

Python

Pandas, NumPy

Scikit-learn

NLTK / SpaCy

Matplotlib

Conclusion

This project demonstrates a complete NLP workflow—from raw text preprocessing to classification and topic discovery. It highlights how machine learning can effectively extract structure and meaning from large-scale unstructured text data.

