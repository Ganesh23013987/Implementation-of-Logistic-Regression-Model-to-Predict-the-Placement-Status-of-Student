<img width="635" alt="image" src="https://github.com/user-attachments/assets/e5815320-0d30-4065-be36-2406ba76d056"># Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: GANESH D
RegisterNumber:  212223240035
*/
```

```
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

dataset = pd.read_csv("/content/Placement_Data.csv")
dataset.head()
dataset.tail()
```

<img width="450" alt="image" src="https://github.com/user-attachments/assets/e9d0298f-1063-4cfe-a9a2-ddd8048233ea">

```
dataset.info()
```

<img width="450" alt="image" src="https://github.com/user-attachments/assets/0915a300-76bc-4899-86e8-ed0bdccb6a4d">

```
dataset=dataset.drop(['sl_no'],axis=1)
dataset.info()
```

<img width="450" alt="image" src="https://github.com/user-attachments/assets/a83ea31d-c444-4325-8355-1564aaa71d60">

```
dataset["gender"]=dataset["gender"].astype('category')
dataset["ssc_b"]=dataset["ssc_b"].astype('category')
dataset["hsc_b"]=dataset["hsc_b"].astype('category')
dataset["degree_t"]=dataset["degree_t"].astype('category')
dataset["workex"]=dataset["workex"].astype('category')
dataset["specialisation"]=dataset["specialisation"].astype('category')
dataset["status"]=dataset["status"].astype('category')
dataset["hsc_s"]=dataset["hsc_s"].astype('category')
```

```
dataset.info()
```

<img width="450" alt="image" src="https://github.com/user-attachments/assets/285591ed-1310-4f11-bd75-1ec2edd14c5d">

```
dataset["gender"]=dataset["gender"].cat.codes
dataset["ssc_b"]=dataset["ssc_b"].cat.codes
dataset["hsc_b"]=dataset["hsc_b"].cat.codes
dataset["degree_t"]=dataset["degree_t"].cat.codes
dataset["workex"]=dataset["workex"].cat.codes
dataset["specialisation"]=dataset["specialisation"].cat.codes
dataset["status"]=dataset["status"].cat.codes
dataset["hsc_s"]=dataset["hsc_s"].cat.codes
```

```
dataset.info()
```

<img width="450" alt="image" src="https://github.com/user-attachments/assets/f19cf331-74d5-47d8-b091-b7343b769dce">

```
dataset.head()
```

<img width="450" alt="image" src="https://github.com/user-attachments/assets/8ef7cf84-d8b5-425a-bab8-4263389c46e1">

```
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values
```

```
x.shape
```

<img width="450" alt="image" src="https://github.com/user-attachments/assets/739203e8-b839-476a-8f28-f177cfde1b6f">


```
y.shape
```

<img width="450" alt="image" src="https://github.com/user-attachments/assets/5fd7edb4-2548-4758-a64d-5f6326d27998">

```
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state=5,stratify=y)
```

```
x_train.shape
```

<img width="450" alt="image" src="https://github.com/user-attachments/assets/49594014-d4e8-48d2-af5c-ba15620c48d6">

```
y_train.shape
```

<img width="450" alt="image" src="https://github.com/user-attachments/assets/4c740bc6-b392-4bdb-9242-5f9a42fd8fac">


```
dataset=LogisticRegression()
dataset.fit(x_train,y_train)
```

<img width="450" alt="image" src="https://github.com/user-attachments/assets/d7d097bc-32f8-496f-b582-b5d63c9d51b4">

```
y_pred=dataset.predict(x_test)
print(y_pred)
```

<img width="450" alt="image" src="https://github.com/user-attachments/assets/d1ce7f63-ccd5-4ac8-a07b-e0ba51219eeb">

```
from sklearn.metrics import accuracy_score,confusion_matrix
matrix=confusion_matrix(y_test,y_pred)
print(matrix)
```

<img width="450" alt="image" src="https://github.com/user-attachments/assets/defda650-0392-44ea-864a-c1fd87164618">

```
accuracy=accuracy_score(y_test,y_pred)
print(accuracy)
```

<img width="450" alt="image" src="https://github.com/user-attachments/assets/46bda814-2d44-4e93-9635-b77e8d1686e7">


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
