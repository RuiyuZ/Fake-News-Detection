# Fake News Detection

Fake News Detection in Python

In this project, we have used various natural language processing techniques and machine learning algorithms to classify fake news articles using sci-kit libraries from python. It is built using Sklearn, NLTK, spaCy, re libraries. This project was done for the course "Introduction to NLP in Python" at Brandeis University.

### Prerequisites

What things you need to install the software and how to install them:

1. Python 3.6 
   - This setup requires that your machine has python 3.6 installed on it. You can refer to this URL https://www.python.org/downloads/ to download python.

2. The project requires following libraries
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

Below are the columns used to create 3 datasets that have been in use in this project

Column 1: Statement (News headline or text).
Column 2: Label (Label class contains: True, False)

   - The data are in the placed in the /data folder in CSV format

### File descriptions

This project contains the following files:

- The data folder contains the fake and real news data in CSV format.

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
   
 - The plots/ folder that contains 4 folders for each classifier and inside which includes the plots of normalized confusion matrix of each feature

 - The liar_dataset/ folder that contains original LIAR data, which is a publicly available dataset for fake news detection.

The files that contain the main method and can be executed is:

- classifier.py (generates Precision, Recall and F1 Score table of each classifier and visualization files of corresponding confusion matrix)
 
### How to run
- run below command in the terminal
    ```
    python3 classifier.py run
    ```
- After hitting the enter, the program will ask for an input of which classifier you want to use like the following:
 ```
 Choose a classifier:
    1 - All 5 Classifiers
    2 - Multinomial Naive Bayes
    3 - Logistic Regression
    4 - Random Forest Classifier
    5 - Decision Tree Classifier
    6 - Linear SVM
 Type a number:
 ```
 You need to type a number 1-6, and the corresponding classifier will be executed. The result will be printed to the terminal.
