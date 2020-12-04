# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 14:47:35 2020

@author: X3
"""

import pandas as pd

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report as cr,confusion_matrix

from sklearn import tree
from IPython.display import Image
import statsmodels.api as sts

from sklearn.metrics import classification_report as cr, confusion_matrix as cm
from sklearn.feature_selection import f_classif as fs
import numpy as np
from sklearn.feature_selection import RFE
from sklearn import preprocessing # for label encoding
from io import StringIO
import pydotplus

data=pd.read_excel("D://IMARticusProjectsandassignments/Python Project - Bank Lending/Sorted 5 lac data.xlsx")

data.shape

# Removing columns having more than 80% Null Values

a=data.isnull().sum()
cols=["id","member_id","desc","mths_since_last_record","mths_since_last_major_derog","annual_inc_joint","dti_joint","verification_status_joint","open_acc_6m","open_il_12m","open_il_24m","mths_since_rcnt_il","total_bal_il","il_util","open_rv_12m","open_rv_24m","max_bal_bc","all_util","inq_fi","total_cu_tl","inq_last_12m","zip_code","policy_code"]
data=data.drop(cols,axis=1)
data
data.shape
data=data.drop("open_il_6m",axis=1)
data.shape
# Removing columns having strings
cols1=["emp_title","purpose","title"]
data=data.drop(cols1,axis=1)
data.shape
data.columns
# Removing Columns having more than 80% 0's
cols2=["delinq_2yrs","pub_rec","total_rec_late_fee","recoveries","collection_recovery_fee","collections_12_mths_ex_med","acc_now_delinq","tot_coll_amt"]
data=data.drop(cols2,axis=1)
data.shape
data.columns

# Checking Unique and singularities from dataset
data.dtypes
data.pymnt_plan.value_counts()
data.addr_state.value_counts()
data.mths_since_last_delinq.isna().sum()
data.isna().sum()
data.dtypes==['object'].value_counts()



     # Removing columns having Biasness and 90% Singularities
     cols3=["pymnt_plan","application_type"]
     data=data.drop(cols3,axis=1)
     data.shape
     data.dtypes
#segregate numeric and categorical data
numcols=data.select_dtypes(exclude=['object']).columns.values    
factcols=data.select_dtypes(include=['object']).columns.values    
factcols     
# Check for multicollinearity using HeatMap
import seaborn as sns
numcols
cor=data[numcols[0:len(numcols)-1]].corr()    
cor=np.tril(cor,k=1)
sns.heatmap(cor,
            xticklabels=numcols[0:len(numcols)-1],
            yticklabels=numcols[0:len(numcols)-1],
            annot=True, linewidths=1,
            vmin=0,vmax=1,
            square=True)


# Extra Analysis
numcols            
factcols
data.emp_length.unique()
data.inq_last_6mths.unique()            
data.addr_state.unique()            
data=data.drop('addr_state',axis=1)            
data.shape            

# Date Columns
data.dtypes
cols5=["issue_d","earliest_cr_line","last_pymnt_d","next_pymnt_d","last_credit_pull_d"]
data=data.drop(cols5,axis=1)
data.dtypes
factcols
numcols


# 

data.home_ownership.value_counts()
data.emp_length.value_counts()
data.grade.value_counts()
data.term.value_counts()

factcols=factcols.drop("addr_state")
factcols
data.initial_list_status.value_counts()
factcols=factcols.drop("sub_grade")
data=data.drop("sub_grade",axis=1)




# Removing Spaces from Categorical Columns
for f in factcols:
    print("Factor variable = ", f)
    print(data[f].unique())
    print("***")

data.emp_length.value_counts()

def removeSpaces(x):
    x = x.strip()
    return(x)
for f in factcols:
    print(f)
    data[f] = data[f].apply(removeSpaces)


factcols

data.strip('term')


data.term.value_counts()

data.dtypes
factcols




# 2) level Merging
factv=[]; oldv=[]; newv=[]

factv.append("verification_status")
oldv.append(['Verified','Source Verified'])
newv.append('Verified')

factv.append("home_ownership")
oldv.append(['ANY', 'OTHER','NONE'])
newv.append("OTHER")

factv.append("grade")
oldv.append(['F','G'])
newv.append("G")

print(factv)
print(oldv)
print(newv)


# parameters:
# df = dataframe (pandas)
# factv: factor variable
# oldv: list of old values to be replaced
# newv: new value to be replaced with
    
def replaceFactorValues(df,factv,oldv,newv):
    if (len(factv) == len(oldv) == len(newv) ):
        
        for i in range(0,len(factv)):
            df[factv[i]] [df[factv[i]].isin(oldv[i])] = newv[i]
            
            # internally, the above code translates to ...
            # census.workclass [census.workclass.isin(['','',''])] = 'new'
        msg = "SUCCESS: 1 Updates done"
    else:
        msg = "ERRCODE:-1  Inconsistent length in the input lists"
    
    return(msg)
 

ret = replaceFactorValues(data,factv,oldv,newv)
print(ret)
data

data.grade.value_counts()

# Imputation of NA according to data type
    def imputenull(data):
        for col in data.columns:
            if data[col].dtypes == 'int64' or data[col].dtypes == 'float64':
                                data[col].fillna((data[col].median()), inplace=True)
            else:
                data[col].fillna(data[col].value_counts().index[0], inplace=True)

imputenull(data)
data.columns.isna()

# Now Label Encoding instead of creating Dummies
def convertFactorsToNum(df,factcols):
    le = preprocessing.LabelEncoder()
    
    for f in factcols:
        df[f] = le.fit_transform(df[f])
    
    return(1)

ret = convertFactorsToNum(data,factcols)
print(ret)

data.columns
data.dtypes  

# Now Standardize the whole dataset
# Standardize/Transform the dataset using MinMax Transformation 
data_std=data.copy(deep=True)
totalcols=len(data_std.columns)
minmax=preprocessing.MinMaxScaler()
vals=minmax.fit_transform(data_std.iloc[:,0:totalcols-1])
data_std.iloc[:,0:totalcols-1]=vals
data_std



# Now we used floor function on entire dataset as model building is not possible 
data=data.apply(np.floor) 




# Now split the data into Train and test for Model Building
totalcols=len(data.columns)
train,test=train_test_split(data,test_size=0.3)        
train.shape            
test.shape            

trainx=train.iloc[:,0:totalcols-1]
trainy=train.iloc[:,totalcols-1]
print("trainx= {},trainy= {}".format(trainx,trainy.shape))


testx=test.iloc[:,0:totalcols-1]
testy=test.iloc[:,totalcols-1]

print("testx={},testy={}".format(testx,testy.shape))

# Now Build the Logistic Regression Model
m2=sts.Logit(trainy,trainx).fit()
# Now summarize the model
m2.summary()


# prediction on the test data
p2=m2.predict(testx)

p2[0:6]

# count of y-variables in test set
testy.value_counts()

# start with the initial cutoff as 0.5
len(p2[p2<0.10])
len(p2[p2>0.10])

# converting probabilities into classes
predY = p2.copy()

predY[predY > 0.10] = 1
predY[predY < 0.10] = 0
predY.value_counts()

# confusion matrix
confusion_matrix(list(testy),list(predY))
# cm(testy,predY)
# testy.value_counts()

# classification report
print(cr(testy,predY))

# AUC and ROC curve
from sklearn import metrics
fpr, tpr, threshold = metrics.roc_curve(testy, predY)
roc_auc = metrics.auc(fpr,tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
