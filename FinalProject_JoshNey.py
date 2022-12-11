#!/usr/bin/env python
# coding: utf-8

st.markdown("Final Project")

# 
# ## Josh Ney
# 
# ### December 11, 2022
# ---

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# #### Q1
import os

os.chdir('/Users/joshuaney/Desktop/Georgetown/Fall2022/OPIM-607-201/FinalProject')

s = pd.read_csv("social_media_usage.csv")
print(s.shape) # Check dimensions


# ---
# #### Q2


def clean_sm(x):
    x = np.where(x == 1, 1, 0)
    return x



# Data frame to test the function
toy_data = [["a",1], ["b",2], ["c",0]]
toy_df = pd.DataFrame(toy_data, columns = ['Test', 'Data'])
print(toy_df)



clean_sm(toy_df["Data"]) # test works


# ---
# #### Q3

# Clean data
ss = pd.DataFrame({
    "income":np.where(s["income"] > 9, np.nan, s["income"]),
    "education":np.where(s["educ2"] > 8, np.nan, s["educ2"]),
    "parent":np.where(s["par"] == 1, 1, 0), # if the person is a parent, put a 1, if not, put a 0
    "married":np.where(s["marital"] == 1, 1, 0), # if the person is married, put a 1, if not, put a 0
    "female":np.where(s["gender"] == 2, 1, 0), # if the person is female, put a 1, if not, put a 0
    "age":np.where(s["age"] > 98, np.nan, s["age"]),
    "sm_li":np.where(clean_sm(s["web1h"]) == 1, 1, 0)})



clean_sm(ss["sm_li"])

ss


ss.isnull().sum()


# Drop all missing data
ss_clean = ss.dropna()



ss_clean.isnull().sum() # check to ensure missing values were dropped



# Exploratory analysis - swapped x variables in and out to view how the features were related to the target
alt.Chart(title = "Exploratory Analysis of LinkedIn Data", data = ss_clean.groupby(["income", "education"],     as_index=False)["sm_li"].mean()).     mark_circle().encode(
        x="income",
        y="sm_li",
        color="education:N").configure_axis(
        titleFontSize=14
    )


# ---
# #### Q4


# Target (y) and feature set (x)
y = ss_clean["sm_li"]
x = ss_clean[["income", "education", "parent", "married", "female", "age"]]


# ---
# #### Q5



# Split data into training and test set
x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    stratify=y,       # same number of target in training & test set
                                                    test_size=0.2,    # hold out 20% of data for testing
                                                    random_state=1)   # set for reproducibility

# x_train contains 80% of the data and contains the features used to predict the target when training the model. 
# x_test contains 20% of the data and contains the features used to test the model on unseen data to evaluate performance. 
# y_train contains 80% of the the data and contains the target that we will predict using the features when training the model. 
# y_test contains 20% of the data and contains the target we will predict when testing the model on unseen data to evaluate performance.


# ---
# #### Q6


# Initialize algorithm for the logistic regression with class_weight set to balanced
lr = LogisticRegression(class_weight = "balanced")


# Fit algorithm to training data
lr.fit(x_train, y_train)


# ---
# #### Q7
# 
# The accuracy of the model is 70%



# Make predictions using the model and the testing data
y_pred = lr.predict(x_test)


confusion_matrix(y_test, y_pred) # confusion matrix is interpreted in Q8


# Model accuracy is 70%
round((113 + 63) / (113 + 55 + 21 + 63),1)

# Model accuracy check
print(classification_report(y_test, y_pred))


# ---
# #### Q8

# Comparing those predictions to the actual test data using a confusion matrix (positive class=1)

pd.DataFrame(confusion_matrix(y_test, y_pred),
             columns=["Predicted Negative", "Predicted Positive"],
             index=["Actual Negative","Actual Positive"]).style.background_gradient(cmap="PiYG")


# The confusion matrix above depicts the accuracy of the model. 
# + The top-left quadrant (True Negatives: 113) represent people who were predicted to not be LinkedIn users who indeed are not LinkedIn users.
# + The top-right quadrant (False Positives: 55) represent people who were predicted to be LinkedIn users who actually are not LinkedIn users.
# + The bottom-left quadrant (False Negatives: 21) represent people who were predicted to not be LinkedIn users who actually are LinkedIn users.
# + The bottom-right quadrant (True Positives: 63) represent people who were predicted to be LinkedIn users who indeed are LinkedIn users.

# ---
# #### Q9
# + Other metrics to evaluate model performance:
#     + **Precision**: Precision is calculated as $\frac{TP}{(TP+FP)}$ and is important when the goal is to minimize incorrectly predicting positive cases. This may be the preferred evalutation metric when looking at something such as cancer screening becuase it is imperative to be accurate in how many of the positive predictions were indeed positive.
#     + **Recall**: Recall is calculated as $\frac{TP}{(TP+FN)}$ and is important when the goal is to minimze the chance of missing positive cases. This may be the preferred evaluation metric when looking at something like fraud as it is important to not miss actual cases of fraud. This is also known as sensitivity.
#     + **F1 score**: F1 score is the weighted average of recall and precision calculated as $2\times\frac{(precision x recall)}{(precision+recall)}$. It is essentially a measure of the test's accuracy and is important to check in every model to evaluate the model overall.


# Precision: TP/(TP+FP)
precision = 63/(63+55)
precision



# Recall: TP/(TP+FN)
recall = 63/(63+21)
recall


# F1 Score
f1_score = 2 * (precison * recall)/(precision + recall)
f1_score



# Checking work with a classificaiton_report
print(classification_report(y_test, y_pred))


# ---
# #### Q10
# Making Predictions


# First, checking with a dataframe

# New data for predictions
newdata = pd.DataFrame({
    "income": [8], # high income = 8
    "education": [7], # high level of education = 7
    "parent": [0], # not a parent = 0
    "married": [1], # married
    "female": [1], # is a female (females are depicted by 1's per Q3)
    "age": [42] # 42 years-old
})


newdata


# Use model to make predictions
newdata["sm_li"] = lr.predict(newdata)


newdata


# 42 year-old prediction with a probability

# New data for features: "income", "education", "parent", "married", "female", "age"
person = [8, 7, 0, 1, 1, 42]

# Predict class, given input features
predicted_class = lr.predict([person])

# Generate probability of positive class (=1)
probs = lr.predict_proba([person])

# Warning is saying I used feature names originally, but not this time. It is okay, still ran as intended


# Print predicted class and probability
print(f"Predicted class: {predicted_class[0]}") # 0 = Not a LinkedIN user, 1 = A LinkedIn user
print(f"Probability that this person is a LinkedIn user: {probs[0][1]}")


# 82 year-old prediction with a probability

# New data for features: "income", "education", "parent", "married", "female", "age"
person2 = [8, 7, 0, 1, 1, 82]

# Predict class, given input features
predicted_class2 = lr.predict([person2])

# Generate probability of positive class (=1)
probs2 = lr.predict_proba([person2])

# Warning is saying I used feature names originally, but not this time. It is okay, still ran



# Print predicted class and probability
print(f"Predicted class: {predicted_class2[0]}") # 0 = Not a LinkedIN user, 1 = A LinkedIn user
print(f"Probability that this person is a LinkedIn user: {probs2[0][1]}")


# Probability delta - how the probability changed between the 42 and 82 year-old people
probs-probs2


# **The probability of a person being a LinkedIn user decreased by 24.8% when the age of the person increased from 42 to 82 years-old keeping everything else the same - ceteris paribus.**
