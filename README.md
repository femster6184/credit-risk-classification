# credit-risk-classification

## Overview of the Analysis

Lending companies lend money/properties to borrowers with the expectation that the borrower will either return the asset or repay the lender. Credit Risk is
associated with a borrower not returning an asset or paying a loan back causing a lender to lose money. This is measured by lenders in many ways, however in
this analysis we will use Machine Learning to analyze a dataset of historical lending activity from a peer-to-peer lending services company to build a model
that can identify the creditworthiness of borrowers.


Using a machine learning model, I will try to determine which loans are healthy (low-risk) or non-healthy (high-risk) based on the loan status provided by the
lending company. The Logistic Regression Algorithm is the best tool to use for our machine learning model since it is widely used to predict the probability of
a target variable in classification problems.

Using the dataset provided by the lending company, I created a Logistic Regression Model that generated an accuracy score of 94%. Although the model generated
a high-accuracy, the models recall value (89%) for non-healthy loans is lower than the recall value (100%) for healthy loans. This indicates that the model
will predict loan status's as healthy better than being able to predict loan status's as non-healthy. This can be attributed to the dataset being imbalanced,
meaning that most of the data belongs to one class label (in this case healthy loans (0) greatly outweighed non-healthy loans (1)). The value_count function of
the y labels (step 3 of "Split the Data into Training and Testing Sets") showed that the data is highly imbalanced by indicating healthy loans (majority class)
and non-healthy loans (minority class)having responses of 75036 and 2500 respectively.

According to the confusion matrix in step 3 [Create a LRM w/ Original Imbalanced Data]:

Out of the 18,759 loan status's that are healthy (low-risk), the model predicted 18,679 as healthy correctly and 80 as healthy incorrectly. Out of the 625 loan
status's that are non-healthy (high-risk), the model predicted 558 as non-healthy correctly and 67 as non-healthy incorrectly.

To generate a higher accuracy score and have the model catch more mistakes when classifying non-healthy loans, we can oversample the data using the
RandomOverSampler module from the imbalanced-learn library, which adds more copies of the minority class (non-healthy loans) to obtain a balanced dataset (Step
1 of "Predict a Logistic Regression Model with Resampled Training Data"). Using the dataset provided by the lending company, I created a Logistic Regression
Model fit with the oversampled data that generated an accuracy score of 99.6%, which turns out to be higher than the model fitted with the imbalanced data. The
oversampled model performs better due to the dataset being balanced. The models non-healthy loans recall value increased from 89% to 100% indicating that the
model does an exceptional job in catching mistakes such as labeling non-healthy (high-risk) loans as healthy (low-risk).

According to the confusion matrix in step 3 [Create a LRM w/ Resampled(oversampled) Data]:

* Out of the 18,759 loan status's that are healthy, the model predicted 18,668 as healthy correctly and 91 as healthy incorrectly.

* Out of the 625 loan status's that are non-healthy (high-risk), the model predicted 623 as non-healthy correctly and 2 as non-healthy incorrectly.



## Results


### Logistic Regression Model fitted with Imbalanced Data:


The Logistic Regression model fitted with the Imbalanced DataSet predicted correctly healthy loans 100% of the time non-healthy loans 87% of the time.


The model fitted with imbalanced data has a higher possibility of making these mistakes:

* a healthy loan (low-risk) is classified as a non-healthy loan (high-risk).
* a non-healthy loan (high-risk) is classified as a healthy loan (low-risk).

According to the models recall scores, the model made no mistakes when predicting healthy loans (100% recall) and was 11% inaccurate when predicted non-healthy loans(89% recall). The model also generated an accuracy score of 94% but could be improved if the dataset was more balanced.



### Logistic Regression Model fitted with Balanced (oversampled) Data:


The Logistic Regression model fitted with the OverSampled DataSet predicted accurately healthy loans 100% of the time and non-healthy loans 87% of the time.


The model fitted with balanced (oversampled) data has a much lower possibility of making these mistakes:

* a healthy loan (low-risk) is classified as a non-healthy loan (high-risk).
* a non-healthy loan (high-risk) is classified as a healthy loan (low-risk).

According to the models recall scores, the model made no mistakes when predicting both healthy and non-healthy loans. The model also generated an accurace score of 99.6% due to the dataset being balanced.

## Summary


A lending company might want a model that requires classifying healthy loans and non-healthy loans correctly most of the time:

* healthy loans being identified as a non-healthy loan might be costly for a lending company since it might cause the loss of customers.

* non-healthy loans being identified as a healthy loan might be more costly for a lending company due to the loss of funds being provided by the lender.

The Logistic Regression model fitted with OverSampled data performed much better than the model fitted with Imbalanced data due to the data being balanced and generating a higher accuracy score and a higher recall, indicating that the model will make extremely fewer mistakes when classifying non-healthy loans.


The lending company would most likely want fewer False Positives due to the high possibility of a lender loosing provided funds when classifying non-healthy loans as healthy. The data below is shown in the confusion matrices which indicates how many healthy/non-healthy loans the model predicted correctly/incorrectly:

Model fitted with Imbalanced Data:

* 67 (FALSE POSITIVES) --> The actual value is non-healthy and the predicted value is healthy

* 80 (FALSE NEGATIVES) --> The actual value is healthy and the predicted value is non-healthy


Model fitted with Oversampled, Balanced Data:

* 2 (FALSE POSITIVES) --> The actual value is non-healthy and the predicted value is healthy

* 91 (FALSE NEGATIVES) --> The actual value is healthy and the predicted value is non-healthy

According to the confusion matrices, the number of False Postives drastically decreases indicating the model will classify healthy & non-healthy loans significantly more correct for the oversampled data than with the imbalanced data. Based off of this analysis, I would recommend using Model 2 (Logistic Regression Model fitted with Balanced (oversampled) data.
