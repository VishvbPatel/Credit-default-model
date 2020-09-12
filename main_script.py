#!/usr/bin/env python
# coding: utf-8

# In[51]:


#importing all the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder #for one hot encoding 
from sklearn.compose import make_column_transformer 
from fancyimpute import KNN #for imputation
from sklearn.model_selection import GridSearchCV #for grid search
from sklearn.utils import check_random_state
from sklearn.metrics import confusion_matrix #for confusion matrix
from sklearn.metrics import accuracy_score #for accuracy score
from sklearn.metrics import roc_auc_score #for AUC value
from sklearn.metrics import precision_recall_fscore_support #for checking precision, recall and f_score
#Reading the dataset
df = pd.read_csv('/Users/vishvmac/Machine Learning/Machine learning research paper/Project proposal/Project code/german_credit_data.csv')


# In[52]:


#Using the KNN imputer to impute the missing categorical data
encoder = OrdinalEncoder()
imputer = KNN()
# create a list of categorical columns to iterate over
cat_cols = ['Saving accounts','Checking account']

def encode(data):
    '''function to encode non-null data and replace it in the original data'''
    #retains only non-null values
    nonulls = np.array(data.dropna())
    #reshapes the data for encoding
    impute_reshape = nonulls.reshape(-1,1)
    #encode date
    impute_ordinal = encoder.fit_transform(impute_reshape)
    #Assign back encoded values to non-null values
    data.loc[data.notnull()] = np.squeeze(impute_ordinal)
    return data

#create a for loop to iterate through each column in the data
for columns in cat_cols:
    encode(df[columns])


# In[53]:


#One hot encoding the categorical data
#removing the features which have "Nan" values as we can't one hot encode data with "Nan"or "missing" values
df_1 = df.loc[:,'Saving accounts']
df_2 = df.loc[:,'Checking account']
df = df.drop(['Saving accounts','Checking account'], axis=1)
X = df.loc[:,['Age','Housing','Credit amount','Duration','Purpose']]
Y = df.loc[:,['Risk']]
X_new=pd.get_dummies(X,dummy_na=True,drop_first=True)
X_new = X_new.drop(['Purpose_nan','Housing_nan'], axis=1)
Y_new=pd.get_dummies(Y,drop_first=True)


# In[54]:


"""print(Y_new)
print(X_new)"""


# In[55]:


#treating the job features saperately as they are having numbers but need to be one hot encoded
F = df.loc[:,'Job']
F = pd.get_dummies(F,prefix = 'Job')
F = F.drop(['Job_0'], axis =1)
X_new = pd.concat([F,X_new], axis =1) #Adding the job feature back to X_new


# In[56]:


df_new = pd.concat([X_new,Y_new,df_1,df_2],axis=1) #Combining X_new, Y_new and the 'Savings account' and 'Checking account' features for imputing the missing data in 'Savings account' and 'Checking account' features


# In[57]:


encode_data = pd.DataFrame(np.round(imputer.fit_transform(df_new)),columns = df_new.columns) #imputing the missing data


# In[58]:


#after imputing the missing data, getting dummy variables from the imputed features
F1 = encode_data.loc[:,'Saving accounts']
F1 = pd.get_dummies(F1,prefix = 'Saving accounts')
F1 = F1.drop(['Saving accounts_3.0'], axis =1)
F2 = encode_data.loc[:,'Checking account']
F2 = pd.get_dummies(F2,prefix = 'Checking account')
F2 = F2.drop(['Checking account_2.0'], axis =1)
encode_data = encode_data.drop(['Saving accounts','Checking account'],axis =1)


# In[59]:


encode_data = pd.concat([encode_data,F1,F2],axis =1) #again gettig the full dataset by combining all the features


# In[60]:


#Making the new features data set
X_new1 =encode_data.drop(['Risk_good'],axis=1)


# In[61]:


#scaling the dataset with minmax scaler
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
X_new1 =pd.DataFrame(min_max_scaler.fit_transform(X_new1), columns=X_new1.columns, index=X_new1.index)
encode_data =pd.DataFrame(min_max_scaler.fit_transform(encode_data), columns=encode_data.columns, index=encode_data.index)


# In[62]:


print(X_new1,Y_new)


# In[63]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

#apply SelectKBest class to extract top 10 best features
bestfeatures = SelectKBest(score_func=chi2, k='all')
fit = bestfeatures.fit(X_new1,Y_new)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X_new1.columns)
#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
print(featureScores.nlargest(23,'Score'))  #print all the features


# In[64]:


X_new1 = X_new1.drop(['Job_1','Job_2','Job_3'],axis =1) #removing the least related features


# In[65]:


#again checking correlation between features and output
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

#apply SelectKBest class to extract top 10 best features
bestfeatures = SelectKBest(score_func=chi2, k='all')
fit = bestfeatures.fit(X_new1,Y_new)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X_new1.columns)
#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
print(featureScores.nlargest(23,'Score'))  #print all the features


# In[66]:


#converting the both features and output to numpy array
X_new1 = X_new1.to_numpy()
Y_new = Y_new.to_numpy()


# In[67]:


# Using the stratified K-fold, the X_train, X_test and y_train, y_test are going to be used for SMOTE, LDA and SVM
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=10)
skf.get_n_splits(X_new1, Y_new.ravel())
StratifiedKFold(n_splits=10, random_state=None, shuffle=True)
for train_index, test_index in skf.split(X_new1, Y_new):
     #print("TRAIN:", train_index, "TEST:", test_index)
     X_train, X_test = X_new1[train_index], X_new1[test_index]
     y_train, y_test = Y_new.ravel()[train_index], Y_new.ravel()[test_index]


# In[68]:


from imblearn.over_sampling import SMOTE, ADASYN, SMOTENC, SVMSMOTE

X_train, y_train = SMOTE().fit_resample(X_train, y_train)


# In[69]:


#LDA for SMOTE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components = 1)
X_train = lda.fit_transform(X_train,y_train.ravel())
X_test = lda.fit_transform(X_test,y_test.ravel())


# In[ ]:


#Using grid search for optimizing the parameters of SVM for SMOTE, LDA and SVM
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
classifier = GridSearchCV(SVC(random_state = 0),{'C':[0.0001,0.001,0.01,0.1,1,10,20,30,40,50,60,100,1000,10000],
                                                     'kernel':['rbf','sigmoid','poly'],
                                                   'gamma':[0.0001,0.001,0.01,0.1,1,10,20,30,40,50,60,100,1000,10000],
                                                   'degree':[2,3,4] },
                          cv =5,scoring = 'accuracy',return_train_score =False)
classifier.fit(X_train,y_train)

print(classifier.cv_results_)
print(classifier.best_score_)
print(classifier.best_params_)


# In[70]:


#SVC for the SMOTE and LDA
from sklearn.svm import SVC
classifier = SVC(C =10,kernel = 'rbf',gamma =0.1)
classifier.fit(X_train,y_train.ravel())
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test,classifier.predict(X_test))
Accuracy =  accuracy_score(y_test, y_pred)
print(cm,'overall accuracy for SMOTE,LDA and SVM',Accuracy)
print('AUC score for SMOTE,LDA and SVM:',roc_auc_score(y_test, y_pred))
print('precision,recall,fscore for SMOTE,LDA and SVM:',precision_recall_fscore_support(y_test, y_pred, warn_for=('precision', 'recall', 'f-score'), average='binary'))


# In[71]:


# stratified K-fold for only LDA and SVM
skf1 = StratifiedKFold(n_splits=10)
skf1.get_n_splits(X_new1, Y_new.ravel())
StratifiedKFold(n_splits=10, random_state=None, shuffle=True)
for train_index, test_index in skf.split(X_new1, Y_new):
     #print("TRAIN:", train_index, "TEST:", test_index)
     X_train1, X_test1 = X_new1[train_index], X_new1[test_index]
     y_train1, y_test1 = Y_new.ravel()[train_index], Y_new.ravel()[test_index]


# In[72]:


#LDA for only LDA and SVM
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda1 = LDA(n_components = 1)
X_train1 = lda.fit_transform(X_train1,y_train1.ravel())
X_test1 = lda.fit_transform(X_test1,y_test1.ravel())


# In[ ]:


#Using grid search for optimizing the parameters of SVM for LDA and SVM
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
classifier1 = GridSearchCV(SVC(random_state = 0),{'C':[0.0001,0.001,0.01,0.1,1,10,20,30,40,50,60,100,1000,10000],
                                                     'kernel':['rbf','sigmoid','poly'],
                                                   'gamma':[0.0001,0.001,0.01,0.1,1,10,20,30,40,50,60,100,1000,10000],
                                                   'degree':[2,3,4] },
                          cv =5,scoring = 'accuracy',return_train_score =False)
classifier.fit(X_train1,y_train1)

print(classifier1.cv_results_)
print(classifier1.best_score_)
print(classifier1.best_params_)


# In[73]:


#SVC for the LDA
from sklearn.svm import SVC
classifier1 = SVC(C =10,kernel = 'rbf',gamma =0.1)
classifier1.fit(X_train1,y_train1.ravel())
y_pred1 = classifier1.predict(X_test1)
cm1 = confusion_matrix(y_test1,classifier1.predict(X_test1))
Accuracy1 =  accuracy_score(y_test1, y_pred1)
print(cm1,'overall accuracy for LDA and SVM',Accuracy1)
print('AUC score for LDA and SVM:',roc_auc_score(y_test1, y_pred1))
print('precision,recall,fscore for LDA and SVM:',precision_recall_fscore_support(y_test1, y_pred1, warn_for=('precision', 'recall', 'f-score'), average='binary'))


# In[74]:


#stratified k-fold for only SVM with original data
skf2 = StratifiedKFold(n_splits=10)
skf2.get_n_splits(X_new1, Y_new.ravel())
StratifiedKFold(n_splits=10, random_state=None, shuffle=True)
for train_index, test_index in skf.split(X_new1, Y_new):
     #print("TRAIN:", train_index, "TEST:", test_index)
     X_train2, X_test2 = X_new1[train_index], X_new1[test_index]
     y_train2, y_test2 = Y_new.ravel()[train_index], Y_new.ravel()[test_index]


# In[ ]:


#Using grid search for optimizing the parameters of SVM for only SVM
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
classifier2 = GridSearchCV(SVC(random_state = 0),{'C':[0.0001,0.001,0.01,0.1,1,10,20,30,40,50,60,100,1000,10000],
                                                     'kernel':['rbf','sigmoid','poly'],
                                                   'gamma':[0.0001,0.001,0.01,0.1,1,10,20,30,40,50,60,100,1000,10000],
                                                   'degree':[2,3,4] },
                          cv =5,scoring = 'accuracy',return_train_score =False)
classifier.fit(X_train2,y_train2)

print(classifier2.cv_results_)
print(classifier2.best_score_)
print(classifier2.best_params_)


# In[75]:


#the SVM with original data
from sklearn.svm import SVC
classifier2 = SVC(C =10,kernel = 'rbf',gamma =0.1)
classifier2.fit(X_train2,y_train2.ravel())
y_pred2 = classifier2.predict(X_test2)
cm2 = confusion_matrix(y_test2,classifier2.predict(X_test2))
Accuracy2 =  accuracy_score(y_test2, y_pred2)
print(cm2,'overall accuracy for only SVM',Accuracy2)
print('AUC score for only SVM:',roc_auc_score(y_test2, y_pred2))
print('precision,recall,fscore for only SVM:',precision_recall_fscore_support(y_test2, y_pred2, warn_for=('precision', 'recall', 'f-score'), average='binary'))


# In[ ]:




