# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. start.
2. Import chardet.
3. Read the dataset.
4. Import SVC from sklearn.
5. Fit the data in the model and run the algorithm.
6. stop.

## Program:
```

Program to implement the SVM For Spam Mail Detection..
Developed by: Thirukaalathessvarar S
RegisterNumber:  212222230161
```

```Python
import chardet 
file="CSVs/spam.csv"
with open(file,'rb')as rawdata: 
    result = chardet.detect(rawdata.read(100000)) 
result
import pandas as pd 
data=pd.read_csv("CSVs/spam.csv",encoding="'Windows-1252") 
data.head()
data.info()
data.isnull().sum()
x=data["v1"].values 
y=data["v2"].values
from sklearn.model_selection import train_test_split 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.feature_extraction.text import CountVectorizer 
cv=CountVectorizer()
x_train=cv.fit_transform(x_train) 
x_test=cv.transform(x_test)
from sklearn.svm import SVC 
svc=SVC() 
svc.fit(x_train,y_train) 
y_pred=svc.predict(x_test) 
y_pred
from sklearn import metrics 
accuracy=metrics.accuracy_score(y_test,y )  
accuracy
```
### Output:
**result:**<br>
![image](https://github.com/ROHITJAIND/EX-09-IMPLEMENTATION-OF-SVM-FOR-SPAM-MAIL-DETECTION/assets/118707073/f7b43b34-2b56-4b7c-af32-f5bb46494a1b)<br>
**head() data:**<br>
![image](https://github.com/ROHITJAIND/EX-09-IMPLEMENTATION-OF-SVM-FOR-SPAM-MAIL-DETECTION/assets/118707073/b36de329-74e2-4a74-9e7e-f4b307477d8c)<br>
**data.info()**<br>
![image](https://github.com/ROHITJAIND/EX-09-IMPLEMENTATION-OF-SVM-FOR-SPAM-MAIL-DETECTION/assets/118707073/ed62854e-7958-46a0-b410-797faecbc65e)<br>
**null values**<br>
![image](https://github.com/ROHITJAIND/EX-09-IMPLEMENTATION-OF-SVM-FOR-SPAM-MAIL-DETECTION/assets/118707073/46fc2c47-c7b2-44b7-a811-a8867738c362)<br>
**y_pred:** <br>
![image](https://github.com/ROHITJAIND/EX-09-IMPLEMENTATION-OF-SVM-FOR-SPAM-MAIL-DETECTION/assets/118707073/cc672e14-1734-4e42-a599-77b39dcea7f4)<br>
**Accuracy** <br>
![image](https://github.com/ROHITJAIND/EX-09-IMPLEMENTATION-OF-SVM-FOR-SPAM-MAIL-DETECTION/assets/118707073/0e83985c-e322-4c79-a392-260f20adeaa3)

### Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
