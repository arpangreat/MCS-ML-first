import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model

# Read the csv file and load that into dataframe df
df = pd.read_csv("canada_per_capita_income.csv")

# Then we create a new dataframe, where we drop the 'per capita income' column to have only the year.
new_df = df.drop("per capita income (US$)", axis="columns")

# We took a new variable income to save the income columns
income = df["per capita income (US$)"]

# We created a Linear Regression Model
reg = linear_model.LinearRegression()

# Then we train the model using the new data
reg.fit(new_df.values, income)

# The co-efficient of the best fit line
print("the co-efficient is: ", reg.coef_)

# The intercept of the best fit line
print("the intercept is: ", reg.intercept_)

# The predicted income according to the linear regression model
print("the income will be: ", reg.predict([[2020]]))
