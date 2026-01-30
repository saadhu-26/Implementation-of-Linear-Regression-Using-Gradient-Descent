# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:

To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:

1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Load the dataset from the CSV file and extract input (x) and output (y) values.

2.Initialize weight (w), bias (b), and learning rate (α).

3.Predict output using y = wx + b, compute error, and update w and b using gradient descent.

4.Repeat the process for a fixed number of iterations and display the regression line and results. 

## Program:

/*
Program to implement the linear regression using gradient descent.
Developed by:SAADHANA A 
RegisterNumber:25018432  
*/

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load CSV'
data = pd.read_csv(r"C:\Users\acer\Desktop\data.csv")

x = data["X"].values.astype(float)
y = data["Y"].values.astype(float)

# Parameters
w, b = 0.0, 0.0
alpha = 0.01
n = len(x)

# Gradient Descent
for _ in range(100):
    y_pred = w*x + b
    dw = (2/n)*np.sum((y_pred-y)*x)
    db = (2/n)*np.sum(y_pred-y)
    w -= alpha*dw
    b -= alpha*db

# Plot
plt.scatter(x,y)
plt.plot(x, w*x+b)
plt.show()

print("w =", w)
print("b =", b)
```

## Output:

<img width="484" height="334" alt="Screenshot 2026-01-30 200934" src="https://github.com/user-attachments/assets/db86ef74-a098-41d5-8989-b8d6e2a500af" />

## Result:

Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
