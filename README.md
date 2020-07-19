# credit-card

Problem Statement:- The credit card fraud detection problem includes modeling past credit card transactions with the knowledge with the knowledge of the ones that turned out to be fraud. This model is then used to identify whether a new transaction is fraudulent or not.

For Dataset:- https://www.kaggle.com/mlg-ulb/creditcardfraud/

Dataset:- This dataset presents transactions made by credit cards that occured in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class(frauds) account for 0.172% of all transactions. It contains only numerical input variable which are the result of a PCA transformation. Features V1,V2,.....,V28 are the principal components obtained. Feature 'class' is the response variable and it takes value 1 in case of fraud and 0 otherwise. 

Dataset Analysis:
  1-The dataset is highly biased containing 0.172% fraud cases.
  2-The dataset has no missing values.
  3-The time and amount features are not  transformed data.
  4-Due to confidentiality issue the original data could not be provided.
  
  Algorithms used:- This is a classification problem, therefore classifications algorithms are practised.
  Naive Bayes, KNN, Logistics Regression, Random Forest
  The best result was given by random forest with an accuracy of 95.95%.
