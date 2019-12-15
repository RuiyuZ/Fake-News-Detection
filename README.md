# Fake News Detection

Fake News Detection in Python

In this project, we have used various natural language processing techniques and machine learning algorithms to classifty fake news articles using sci-kit libraries from python. It is built using Sklearn, NLTK, spaCy, re libraries. This project was done for the course "Introduction to NLP in Python" at Brandeis University.

### Prerequisites

What things you need to install the software and how to install them:

1. Python 3.6 
   - This setup requires that your machine has python 3.6 installed on it. you can refer to this url https://www.python.org/downloads/ to download python.

2. The project requires following librarys
   - Sklearn (scikit-learn)
   - numpy
   - nltk
   - pandas
   - os
   - re
   - matplotlib
   - cvs
   - warnings
   - itertools
   
### Dataset used

We use the LIAR: a publicly available dataset for fake news detection. The data comprises 12.8K manually labeled short statement with detailed analysis and reference, which was collected from PolitiFact.com. We use a version of processed data, which can be acquired from the https://github.com/nishitpatel01/Fake_News_Detection.

Below are the colomns used to create 3 datasets that have been in used in this project

Column 1: Statement (News headline or text).
Column 2: Label (Label class contains: True, False)

   - The data are in the placed in the /data folder in csv format

### File descriptions

This project contains the following files:

- The data folder contains the fake and real news data in csv format.
- The tokenizer.py script contains our own tokenizer method.
- The features.py script contains 4 features that are used in this project:
   - Word Count Feature
   - POS Tagger Feature
   - Sentiment Score Feature
   - Word/sentence length Feature
- The classifier.py script contains 5 classifiers from sklearn that are used in this project:
   - Naive-Bayes Classifier
   - Logistic Regression Classifier
   - Random Forest Classifier
   - Decision Tree Classifier
   - Linear SVM Classifier
   
   Each of the extracted features was used in all of the classifiers. Once fitting the model, we compared the f1 score and checked the confusion matrix. To view the performance of each classifier easily, we plot the confusion matrix with matplotlib and save the result in the folder /plots.

The files that have a main method than can be executed are:

- classifier.py (generates Precision, Recall and F1 Score table of each classifier and visualization files of corresponding confusion matrix)
 
