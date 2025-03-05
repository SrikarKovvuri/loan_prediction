import numpy as numpy
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv("loan_data.csv")

y = df['Loan_Status']

x = df[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Term_Months', 'Credit_History']]


#Next we want to see how categorical variables come into play. So it's ideal if we can do one-hot encoding, 
#where we can covnert each category into a new column with 0 or 1 value
#Our current dataset doesn't have any categorical variables so lets make some dummy ones and then just add them to our x.

# One-Hot Encode categorical features
df = pd.get_dummies(df, columns=['Gender', 'Married', 'Self_Employed', 'Education', 'Property_Area'], drop_first=True)

# Add the encoded categorical variables to X
x = pd.concat([x, df[['Gender_Male', 'Married_Yes', 'Self_Employed_Yes', 'Education_Not Graduate',
                      'Property_Area_Semiurban', 'Property_Area_Urban']]], axis=1)

scaler = StandardScaler()

test_size_ratio = 0.3
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size_ratio, random_state=45)


x_train_scaled = scaler.fit_transform(x_train)

model = LogisticRegression()
model.fit(x_train_scaled, y_train)

print("Model training is complete!")
x_test_scaled = scaler.fit_transform(x_test)
y_pred = model.predict(x_test_scaled)

#Now, let's actually evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")