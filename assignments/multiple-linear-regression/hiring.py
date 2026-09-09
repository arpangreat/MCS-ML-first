import pandas as pd
from sklearn import linear_model

# Load the dataset
hiring = pd.read_csv("hiring.csv")

# Clean column name if UTF-8 BOM is present (e.g. 'ï»¿experience')
hiring.rename(columns={hiring.columns[0]: "experience"}, inplace=True)

# Data Preprocessing:
# Experience: Fill missing experience with 0 and map words to numbers
hiring["experience"] = hiring["experience"].fillna(0)
hiring["experience"] = hiring["experience"].replace(
    {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
        "eleven": 11,
    }
)

# Test score: Fill NA with median value (as in df.bedrooms.fillna(df.bedrooms.median()))
hiring["test_score(out of 10)"] = hiring["test_score(out of 10)"].fillna(
    hiring["test_score(out of 10)"].median()
)

print(hiring)
print()

# Train Linear Regression Model
reg = linear_model.LinearRegression()
reg.fit(hiring.drop("salary($)", axis="columns").values, hiring["salary($)"])

print("The co-eff is: ", reg.coef_.tolist())
print("The intercept is: ", reg.intercept_)
print()

# Predictions:
# Candidate 1: 2 yr experience, 9 test score, 6 interview score
print(
    "Salary for candidate (2 yr experience, 9 test score, 6 interview score):",
    reg.predict([[2, 9, 6]]),
)

# Candidate 2: 12 yr experience, 10 test score, 10 interview score
print(
    "Salary for candidate (12 yr experience, 10 test score, 10 interview score):",
    reg.predict([[12, 10, 10]]),
)
