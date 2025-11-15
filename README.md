# Cardiovascular Disease Detection and Prevention

## Introduction
Cardiovascular diseases (CVD) are a leading global health emergency, particularly in the context of an aging population.  
Thus, developing effective detection and prevention systems is essential. This report presents our work on designing robust data processing and machine learning
techniques to analyze CVD risk and identify the key factors influencing its development. Specifically, we compare
the performance of linear and logistic regression models trained on the Behavioral Risk Factor Surveillance System (BRFSS) dataset.

## Core files description
- `data_preprocessing.ipynb`: notebook containing the data preprocessing pipeline. In particular, the dataset on which we were able to perform the best predictions is produced.  
- `run_research.ipynb`: notebook that contains the complete training and model validation pipeline. In particular, it performs k-cross validation to find the best linear model, hyperparameters tuning, final training and validation.  
- `run.ipynb`: contains the final K-fold cross validation and validation part implemented in `run_research.ipynb`. It runs for less than 10 minutes, allowing to see the best model results without waiting for hyperparameter tuning training.  
- `implementations.py`: contains all the machine learning models implemented (Mean Squared Error with gradient descent and stochastic gradient descent; Least Squares; Ridge Regression; Logistic Regression with gradient descent; L2-Regularized Logistic Regression with gradient descent; L2-Regularized Logistic Regression with Adam and class weights).  
- `helpers.py`: contains methods used to load the dataset and create a submission.  

**⚠️ PLEASE:**  
In order to obtain the best model performances described in the report, create a folder called `data` and a folder inside called `dataset`;  
then upload in `dataset` `x_train`, `y_train`, and `x_test`;  
run `data_preprocessing.ipynb` with the current options; 
then you will obtain datasets called `X_train_pca_ohe.csv`and `X_test_pca_ohe` in dataset folder; 
finally, run `run.ipynb` (to see just the final model and generate the submission on `x_test`, it takes less than 10 minutes and it corresponds to the final part of `run_research.ipynb`) or `run_research.ipynb` to see all training and hyperparameter tuning described in the report and also the final snippets contained in `run.ipynb`.  

## Additional Files
- `run_research_2.ipynb`, `run_research_3.ipynb`, and `run_research_4.ipynb` are notebooks containing the complete training and model validation pipeline on datasets with different preprocessings that did not lead to the best performances. To obtain these results, please create the preprocessed datasets with these different options by changing the boolean variables at the beginning of `data_preprocessing.ipynb` and then run it.  

## Folders 
- `plots`: contains useful plots and graphics produced during preprocessing and training.  
- `report`: contains the PDF report and its LaTeX files. Check `ML_Project_1_report_Girl_Power_team_FINAL2.pdf` to read the submitted report!



