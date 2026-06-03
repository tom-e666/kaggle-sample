import os
import pandas as pd

#path
data_train_path = "titanic\\data\\train.csv"
data_test_path  = "titanic\\data\\test.csv"
submission_path = "titanic\\data\\gender_submission.csv" #optional

#load data
data= pd.read_csv(data_train_path)
print(data.head())

#
