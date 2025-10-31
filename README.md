[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/UcP9Py08)
# ML Project 1: Girl_power team 
## Introduction
Cardiovascular diseases (CVD) are a leading global health emergency, particularly in the context of an aging population. 
Thus, developing effective detection and prevention systems is essential. This report presents our work on designing robust data processing and machine learning
techniques to analyze CVD risk and identify the key factors influencing its development. Specifically, we compare
the performance of linear and logistic regression models trained on the Behavioral Risk Factor Surveillance System (BRFSS) dataset.

## Core files description
- data_preprocessing.ipynb: notebook containing the data preprocessing pipeline. In particular, the dataset on which we were able to perform the best predictions is produced. 
- run_research.ipynb : notebook that contains the complete training and model validation pipeline. In particular, it performs k-cross validation to find the best linear model, hyperparamters tuning, 
final training and validation.
- run.ipynb : contains the final K-fold cross validation and validation part implemented in run_research.ipynb. It runs for less than 10 minutes, allowing to see the best model results withourt waiting for hyperparamters tuning training. 
- implmentations.py: contains all the machine learning models implemented (Mean Squared Error with gradient descent and stochastic gradient discent; Least Squares; Ridge Regression; Logistic Regression with gradient discent; L2-Regularized Logistic Regression with gradient discent;L2-Regularized Logistic Regression with Adam and class weights) 
- helpers.py: contains methods used to load the dataset and create a submission.
-submission_ridge.csv : contains the prediction made on the test set provided to us, using our best performing model.

## Additional Files
- run_research_3.ipynb and run_research_4.ipynb are notebooks containing the complete training and model validation pipeline on datasets with different preprocessings that did not lead to the best performances.

## Folders 
- plots : contains useful plots and graphics produced during preprocessing and training.
- report: contains the PDF report and its Latex files.
- data: contains the folder dataset, where the original dataset and the proprocessed ones can be found.


