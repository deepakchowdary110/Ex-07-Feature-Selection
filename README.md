# Ex-07-Feature-Selection
## AIM
To Perform the various feature selection techniques on a dataset and save the data to a file. 

# Explanation
Feature selection is to find the best set of features that allows one to build useful models.
Selecting the best features helps the model to perform well. 

# ALGORITHM
### STEP 1
Read the given Data
### STEP 2
Clean the Data Set using Data Cleaning Process
### STEP 3
Apply Feature selection techniques to all the features of the data set
### STEP 4
Save the data to the file


# CODE
```
Developed by: O. Shanthan Kumar Reddy
Reg no: 212220040107
```
```
#Importing libraries
from sklearn.datasets import load_boston
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso

from sklearn.datasets import load_boston
boston = load_boston()

print(boston['DESCR'])

import pandas as pd
df = pd.DataFrame(boston['data'] )
df.head()

df.columns = boston['feature_names']
df.head()

df['PRICE']= boston['target']
df.head()

df.info()

plt.figure(figsize=(10, 8))
sns.distplot(df['PRICE'], rug=True)
plt.show()

#FILTER METHODS

X=df.drop("PRICE",1)
y=df["PRICE"]

from sklearn.feature_selection import SelectKBest, chi2
X, y = load_boston(return_X_y=True)
X.shape

#1.Variance Threshold
from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold()
selector.fit_transform(X)

#2.Information gain/Mutual Information
from sklearn.feature_selection import mutual_info_regression
mi = mutual_info_regression(X, y);
mi = pd.Series(mi)
mi.sort_values(ascending=False)
mi.sort_values(ascending=False).plot.bar(figsize=(10, 4))

#3.SelectKBest Model
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest,SelectPercentile
skb = SelectKBest(score_func=f_classif, k=2) 
X_data_new = skb.fit_transform(X, y)
print('Number of features before feature selection: {}'.format(X.shape[1]))
print('Number of features after feature selection: {}'.format(X_data_new.shape[1]))

#4.Correlation Coefficient
cor=df.corr()
sns.heatmap(cor,annot=True)

#5.Mean Absolute Difference
mad=np.sum(np.abs(X-np.mean(X,axis=0)),axis=0)/X.shape[0]
plt.bar(np.arange(X.shape[1]),mad,color='teal')

#Processing data into array type.
from sklearn import preprocessing
lab = preprocessing.LabelEncoder()
y_transformed = lab.fit_transform(y)
print(y_transformed)

#6.Chi Square Test
X = X.astype(int)
chi2_selector = SelectKBest(chi2, k=2)
X_kbest = chi2_selector.fit_transform(X, y_transformed)
print('Original number of features:', X.shape[1])
print('Reduced number of features:', X_kbest.shape[1])

#7.SelectPercentile method
X_new = SelectPercentile(chi2, percentile=10).fit_transform(X, y_transformed)
X_new.shape

#WRAPPER METHOD

#1.Forward feature selection

from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.linear_model import LinearRegression
sfs = SFS(LinearRegression(),
          k_features=10,
          forward=True,
          floating=False,
          scoring = 'r2',
          cv = 0)
sfs.fit(X, y)
sfs.k_feature_names_

#2.Backward feature elimination

sbs = SFS(LinearRegression(),
         k_features=10,
         forward=False,
         floating=False,
         cv=0)
sbs.fit(X, y)
sbs.k_feature_names_

#3.Bi-directional elimination

sffs = SFS(LinearRegression(),
         k_features=(3,7),
         forward=True,
         floating=True,
         cv=0)
sffs.fit(X, y)
sffs.k_feature_names_

#4.Recursive Feature Selection

from sklearn.feature_selection import RFE
lr=LinearRegression()
rfe=RFE(lr,n_features_to_select=7)
rfe.fit(X, y)
print(X.shape, y.shape)
rfe.transform(X)
rfe.get_params(deep=True)
rfe.support_
rfe.ranking_

#EMBEDDED METHOD

#1.Random Forest Importance

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier().fit(X,y_transformed)
importances=model.feature_importances_

final_df=pd.DataFrame({"Features":pd.DataFrame(X).columns,"Importances":importances})
final_df.set_index("Importances")
final_df=final_df.sort_values("Importances")
final_df.plot.bar(color="teal")
```

# OUPUT:

![out1](https://user-images.githubusercontent.com/94168395/170409238-777e953a-f1da-4dff-8e03-ab3c381962ec.jpeg)
![out2](https://user-images.githubusercontent.com/94168395/170409276-421edb67-aed5-44f0-8da6-adb5654b99bb.jpeg)
![out3](https://user-images.githubusercontent.com/94168395/170409306-44a62056-a7f7-491c-9fed-f00fd71dbd06.jpeg)
![out4](https://user-images.githubusercontent.com/94168395/170409336-a7971640-9d36-4429-9f01-b890785c1038.jpeg)
![out5](https://user-images.githubusercontent.com/94168395/170409365-6d82bf7e-124e-49a0-b752-0cd42f054cc2.jpeg)
![out6](https://user-images.githubusercontent.com/94168395/170409381-c71347be-7b35-4b1b-ba48-ece32d93ebb9.jpeg)
![out7](https://user-images.githubusercontent.com/94168395/170409393-9968161f-2c8e-4635-8a41-94afa7908f4e.jpeg)
## FILTER METHOD:
![out8](https://user-images.githubusercontent.com/94168395/170409415-f27bfa00-cabe-4325-a1a8-532f2ce80dbb.jpeg)
## INFORMATION GAIN/MUTAL INFORMATION:
![out9](https://user-images.githubusercontent.com/94168395/170409432-b89f3aa7-f747-4b52-985d-b028d8e8472a.jpeg)
## SELECTKBEST:
![0ut10](https://user-images.githubusercontent.com/94168395/170409465-6d349fc7-de95-48b3-86d4-cfdd01f64b67.jpeg)
## MEAN ABSOLUTE DIFFERENCE:
![out11](https://user-images.githubusercontent.com/94168395/170409501-d2e2e899-6bf4-4482-b0c9-0e4c1553010f.jpeg)
![out12](https://user-images.githubusercontent.com/94168395/170409561-bc69d1f2-3c96-45b3-b57a-8c1171f013e6.jpeg)
## CHI SQUARE TEST:
![out13](https://user-images.githubusercontent.com/94168395/170409576-899bf263-6f72-4a30-a458-54c7ffc7940b.jpeg)
## SELECT PERCENTILE METHOD:
![out14](https://user-images.githubusercontent.com/94168395/170409607-336110d0-f56b-40f5-a02a-fa385ef6bbed.jpeg)
## WRAPPER METHOD

## 1.FORWARD FEATURE ELEMINATION:
![out15](https://user-images.githubusercontent.com/94168395/170409629-70e47ffa-70ed-4e74-b8a5-dcf54b846b0f.jpeg)

## 2.BACKWARD FEATURE ELIMINATION:
![out16](https://user-images.githubusercontent.com/94168395/170409647-aef745b7-47fe-4240-8037-a35d692e46f4.jpeg)

## 3.BI-DIRECTIONAL ELEMINATION:
![out17](https://user-images.githubusercontent.com/94168395/170409664-a063884b-d4c5-4a51-905b-38b7994e3df1.jpeg)

## RECURSIVE FEATURE SELECTION:
![out18](https://user-images.githubusercontent.com/94168395/170409673-a5752ee9-096b-439b-837e-fa8f0d7b2baa.jpeg)

## EMBEDDED METHOD

## 1.RANDOM FOREST IMPORTANCE:
![out19](https://user-images.githubusercontent.com/94168395/170409690-35d72562-3076-4fd7-b391-a2e6554f36ad.jpeg)

# RESULT:
hence the various feature selection techniques are performed  on a dataset and data is saved to a file.
