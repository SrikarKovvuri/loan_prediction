import numpy as numpy
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
df = pd.read_csv("loan_data.csv")

y = df['Loan_Status']

x = df[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Term_Months', 'Credit_History']]


#Next we want to see how categorical variables come into play. So it's ideal if we can do one-hot encoding, 
#where we can covnert each category into a new column with 0 or 1 value

