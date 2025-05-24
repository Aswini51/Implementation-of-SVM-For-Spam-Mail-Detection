# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import libraries.

2.Read the CSV file and display data using head().

3.Split the dataset using train_test_split().

4.Calculate predictions and accuracy.

5.Print the outputs.

6.End the program.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: G ASWINI
RegisterNumber:  212224040037
*/
```
```
import chardet, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn import metrics

# Detect encoding
with open('spam.csv', 'rb') as f:
    print(chardet.detect(f.read(100000)))

# Load data
data = pd.read_csv('spam.csv', encoding='windows-1252')
print(data.head())
print(data.info())
print(data.isnull().sum())

# Split data
x = data['v1'].values   # Labels
y = data['v2'].values   # Messages
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Text vectorization
cv = CountVectorizer()
x_train = cv.fit_transform(x_train)
x_test = cv.transform(x_test)

# Train & predict
model = SVC()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print("Predictions:", y_pred)

# Accuracy
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

```

## Output:

![Screenshot 2025-05-24 130352](https://github.com/user-attachments/assets/ba1cc275-a99e-4105-bcae-a680c4436dcc)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
