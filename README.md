# Catch Me If You Can
Predicting credit card defaults through classification analysis.

## Context 

To credit card companies, when customers are unable to pay their credit card bills, a.k.a. defaulting on their credit card payment, it could be due to different circumstances. 
 - Unexpected issues: There is a sudden change to a person’s income due to job loss, health issues, inability to work, etc. 
 - Deliberate means: Customer has no plans on paying their bills and continue to use the card until it’s stopped by the bank.

Either way, defaulting on credit card payment, is a type of fraud. This imposes huge risk to credit card companies, and we need a way to catch them!

To address this issue, we can predict potential default accounts based on certain attributes. The idea is that the earlier the potential default accounts are detected, the lower the losses we’ll have. On the other hand, we can also take proactive actions to help our customers by providing suggestions to avoid default and minimize our losses that way.

## Objective
- To predict whether the customer will default on their credit card payment next month.
- To have the **highest recall score**, while not suffering precision too much. We care most about recall, because as mentioned, credit card default could be considered as a fraud, hence it’s important that we correctly classify all default cases to minimize risks and losses. 

## Data
 - The dataset, obtained from [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients), contains information on credit card clients in Taiwan. It includes 30,000 observations and 24 features. 
 - Features
   - Credit Info: credit line
   - Demographics: gender, highest education degree, age, and marital status
   - Payment History (Apr ~ Sep 2005): repayment status, payment amount, and bill amount by month
 - Target: Whether the credit card client will default or not next month

## Approach
1. Input the data into the database in PostgreSQL.
2. Perform data cleaning in Python. 
3. Perform exploratory data anlysis.
   - The target distribution entails 22% as default ans 78% as no default. 
   - There isn't a very clear distinction between default and no default on any of the demographic data. 
   - There are no NULL values. 
   - Create corrleation heatmaps to ensure features aren't highly correlated with each other.
4. Perform feature engineering. Note that I've gone back and forth between EDA, feature engineer, and modeling, hence transformed features are created below after modeling results.
   - Credit Line Usage: On a scale from 0 to 1, this is the total percentage of credit used from Apr to Sep 2005.
   - Generation: Using `age` to calculate which generation this person belongs to, i.e., "Generation X", "Millennials", etc.
   - Total Number of Months with Delayed Payment: Out of 6 months (Apr~Sep 2005), the number of months that have delayed payments. 
   - Payment Delayed: This is a 1/0 flag, indicating whether there was ever a payment delayed.
   - Dummy Variables for `education`
5. Perform Modeling.
   0. Use [Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) to do majority of the following.
   1. Apply scaling using [MinMaxSacler()](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html), transforming features by scaling each feature to a range of 0 to 1.
   2. Apply oversampling technique using [ADASYN](https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.over_sampling.ADASYN.html).
   3. Try out various classification models: Logistic Regression, K-Nearest Neighbor, Random Forest, Extra Trees, and Naive Bayes. 
   4. Finalized features in the model:
       1. Credit Line Usage
       2. Total Number of Months with Delayed Payment
       3. Dummy Variables for `education`: `education_graduate`, `education_university`, `education_high_school`, and `education_others`
   5. Top 3 best-performing models are the following, along with their ROC-AUC score. 
       1. Random Forest: 74.88%
       2. Extra Trees: 73.34%
       3. Logistic Regression: 71.81%
## Results
- Random Forest is the winner, with recall score of 67.07%.
- For feature importance of Random Forest model, the following features (along feature importance percentage) end up being very helpful in predicting whether the customer will default on their credit card payment next month:
    1. Total Number of Months with Delayed Payment - 71.2%
    2. Credit Line Usage - 26.3%
- See Confusion Matrix below. You'll see that False Positives are on the higher end and False Negativs are relative low, as we want to focus on Recall. 
    - True Positives: 14.83%
    - False Positives: 22.33% 
    - True Negatives: 55.55%
    - False Negatives: 7.28%
- Currently, the threshold is set at 0.465 to achieve the recall score at 0.67. If the business is more comfortable with higher or lower False Positives/Negatives, we can always adjust the threshold to achieve it.

## Workflow
Follow the jupyter notebooks in the order below:
- 01 PostgreSQL.ipynb
- 02 EDA.ipynb
- 03 Classification Analysis.ipynb