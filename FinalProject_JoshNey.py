#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

st.title("Predicting LinkedIn Users with Machine Learning")
st.subheader("Final Project Georgetown MSBA Programming II")
st.subheader("Created by: Josh Ney")
st.markdown("---")

st.markdown("**Fill out the questions on the left to determine who is likely to be a LinkedIn User!**")
st.text(" \n")

#### Text input box
#### text_income = st.text_input("Enter text:", value_income = "Enter text here")

with st.sidebar:
    inc = st.number_input("Income (low=1 to high=9)", 1, 9)
    deg = st.number_input("College degree? (no=0 to yes=1)", 0, 1)
    par = st.number_input("Parent? (no=0 to yes=1)", 0, 1)
    mar = st.number_input("Married? (0=no, 1=yes)", 0, 1)
    gen = st.number_input("Female? (0=no, 1=yes)", 0, 1)
    age = st.number_input("Age:", 1, 120)

# # Create labels from numeric inputs

# Income
if inc <= 3:
    inc_label = "low income"
elif inc > 3 and inc < 7:
    inc_label = "middle income"
else:
    inc_label = "high income"

# Degree   
if deg == 1:
    deg_label = "college graduate"
else:
    deg_label = "non-college graduate"

# Parent   
if par == 1:
    par_label = "parent"
else:
    par_label = "non-parent"

# Marital
if mar == 1:
    mar_label = "married"
else:
    mar_label = "non-married"

# Gender  
if gen == 1:
    gen_label = "female"
else:
    gen_label = "not a female"

# Age  
if age == 0:
    age_label = "error"
else:
    age_label = age


st.write(f"This person is {age_label} years old, {gen_label}, {mar_label}, a {par_label}, a {deg_label}, and in the {inc_label} bracket.")



s = pd.read_csv("social_media_usage.csv")
print(s.shape)


def clean_sm(x):
    x = np.where(x == 1, 1, 0)
    return x

# toy_data = [["a",1], ["b",2], ["c",0]]
# toy_df = pd.DataFrame(toy_data, columns = ['Test', 'Data'])
# print(toy_df)
# clean_sm(toy_df["Data"])

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

#ss

ss.isnull().sum()

# Drop all missing data
ss_clean = ss.dropna()

ss_clean.isnull().sum() # check to ensure missing values were dropped



# Exploratory analysis - swapped x variables in and out to view how the features were related to the target
# alt.Chart(title = "Exploratory Analysis of LinkedIn Data", data = ss_clean.groupby(["income", "education"],     as_index=False)["sm_li"].mean()).     mark_circle().encode(
#        x="income",
#        y="sm_li",
#        color="education:N").configure_axis(
#        titleFontSize=14
#    )

# Target (y) and feature set (x)
y = ss_clean["sm_li"]
x = ss_clean[["income", "education", "parent", "married", "female", "age"]]


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


# Initialize algorithm for the logistic regression with class_weight set to balanced
lr = LogisticRegression(class_weight = "balanced")

# Fit algorithm to training data
lr.fit(x_train, y_train)


# Make predictions using the model and the testing data
y_pred = lr.predict(x_test)


confusion_matrix(y_test, y_pred) # confusion matrix is interpreted in Q8


# Model accuracy is 70%
round((113 + 63) / (113 + 55 + 21 + 63),1)

# Model accuracy check
print(classification_report(y_test, y_pred))


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
#precision = 63/(63+55)
#precision

# Recall: TP/(TP+FN)
#recall = 63/(63+21)
#recall

# F1 Score
#f1_score = 2 * (precision * recall)/(precision + recall)
#f1_score

# Checking work with a classificaiton_report
#print(classification_report(y_test, y_pred))


# First, checking with a dataframe

# New data for predictions
#newdata = pd.DataFrame({
#    "income": [8], # high income = 8
#    "education": [7], # high level of education = 7
#    "parent": [0], # not a parent = 0
#   "married": [1], # married
#    "female": [1], # is a female (females are depicted by 1's per Q3)
#    "age": [42] # 42 years-old
#})

#newdata

# Use model to make predictions
#newdata["sm_li"] = lr.predict(newdata)

#newdata


# Prediction and probability

# New data for features: "income", "education", "parent", "married", "female", "age"
person = [inc, deg, par, mar, gen, age]

# Predict class, given input features
predicted_class = lr.predict([person])

# Generate probability of positive class (=1)
probs = lr.predict_proba([person])

probs_num = float(probs[:, 1])
probs_num2 = "{0:.1%}".format(probs_num)

st.write(f"Predicted class (0 = Not a LinkedIn User; 1 = LinkedIn User): **{predicted_class[0]}**")
st.write(f"Probability this person is a LinkedIn User: ", probs_num2)


# Print predicted class and probability
#print(f"Predicted class: {predicted_class[0]}") # 0 = Not a LinkedIN user, 1 = A LinkedIn user
#print(f"Probability that this person is a LinkedIn user: {probs[0][1]}")


#### Sentiment Gauge
fig = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = probs_num,
    title = "Probability Gauge of if Someone is a LinkedIn User",
    gauge = {"axis": {"range": [0, 1]},
        "steps": [
            {"range": [0, 0.33], "color": "red"},
            {"range": [0.33, 0.66], "color": "gray"},
            {"range": [0.66, 1], "color": "lightgreen"}
        ],
        "bar": {"color":"yellow"}}
))

st.plotly_chart(fig)