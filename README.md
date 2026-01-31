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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data=pd.read_csv(r"C:\Users\acer\Downloads\startup.csv")
x=data["R&D Spend"].values
y=data["Profit"].values
x_mean = np.mean(x)
x_std = np.std(x)
x = (x - x_mean) / x_std
w = 0.0
b = 0.0
alpha = 0.01
epochs = 100
n = len(x)
losses = []
for _ in range(epochs):
   y_hat = w * x + b
   loss = np.mean((y_hat - y) ** 2)
   losses.append(loss)
   dw = (2/n) * np.sum((y_hat - y) * x )
   db = (2/n) * np.sum(y_hat - y)
   w -= alpha * dw
   b -= alpha * db
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(losses)
plt.xlabel("Iterations")
plt.ylabel("Loss (MSE)")
plt.title("Loss vs Iterations")
plt.subplot(1,2,2)
plt.scatter(x,y)
x_sorted = np.argsort(x)
plt.plot(x[x_sorted],(w*x+b)[x_sorted],color='red')
plt.xlabel("R&D Spend (scaled)")
plt.ylabel("Profit")
plt.tight_layout()
plt.show()
print("Final weight (w):",w)
print("Final bias (b):",b)
```

## Output:

<img width="863" height="368" alt="Screenshot 2026-01-31 122100" src="https://github.com/user-attachments/assets/9c9e9dda-49b5-4547-9945-6ad70f46265b" />

## Result:

Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
