#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import copy
import math
import joblib


# ### 特征处理

# In[2]:


# 根据cv训练集计算所有的LR值
def calculate_LR(df1,df2):
    """
    df1:cv training set
    df2:cv test set
    """
    # log ((2+c)/(1+c)) + log ((2+c)/ (1+c)), {c==1}

    # 有害和中性注释的字典
    p = {}
    n = {}
    for index,row in df1.iterrows():
        if(pd.isna(row['ancestor'])):
            continue
        for i in row['ancestor'].split(','):
            if i not in p.keys():
                p[i]=1
                n[i]=1
            if(row['is_del']==1):
                p[i]+=1
            else:
                n[i]+=1
                
    l = copy.deepcopy(p)
    for i in l.keys():
        l[i]=math.log(p[i]/n[i]) 
    l
    
    #求和计算每个蛋白的lr
    def LR_add(x):
        sum=0
        if(pd.isna(x)):
            return sum
        for i in x.split(','):
            if i in l:
                sum = sum + l[i]
        return sum
    df1['LR'] = df1['ancestor'].apply(lambda x:LR_add(x))
    df2['LR'] = df2['ancestor'].apply(lambda x:LR_add(x))
    df1 = df1.drop(columns=['ancestor'])
    df2 = df2.drop(columns=['ancestor'])
    return df1,df2


# In[ ]:


# 根据cv训练集计算所有的LR值
def calculate_PA(df1,df2):
    """
    df1:cv training set
    df2:cv test set
    """
    # log ((2+c)/(1+c)) + log ((2+c)/ (1+c)), {c==1}

    # 有害和中性注释的字典
    p = {}
    n = {}
    for index,row in df1.iterrows():
        if(pd.isna(row['site'])):
            continue
        for i in row['site'].split(','):
            if i!='':
                if i not in p.keys():
                    p[i]=1
                    n[i]=1
                if(row['is_del']==1):
                    p[i]+=1
                else:
                    n[i]+=1
                
    s = copy.deepcopy(p)
    for i in s.keys():
        s[i]=math.log(p[i]/n[i]) 
    s
    
    #求和计算每个蛋白的pa
    def PA_add(x):
        sum=0
        if(pd.isna(x)):
            return sum
        for i in x.split(','):
            if i != '' and i in s:
                sum = sum + s[i]
        return sum
    df1['PA'] = df1['site'].apply(lambda x:PA_add(x))
    df2['PA'] = df2['site'].apply(lambda x:PA_add(x))
    df1 = df1.drop(columns=['site'])
    df2 = df2.drop(columns=['site'])
    return df1,df2


# In[ ]:


# GO矩阵处理
def LR_matrix(df1,df2):
    """
    df1:cv training set
    df2:cv test set
    """
    GO = []
    for index,row in df1.iterrows():
        if(pd.isna(row['ancestor'])):
            continue
        for i in row['ancestor'].split(','):
            if i not in GO:
                GO.append(i)
    for index,row in df1.iterrows():
        if(pd.isna(row['ancestor'])):
            continue
        for i in row['ancestor'].split(','):
            if i not in GO:
                GO.append(i)
    
    GO.sort()
    
    for i in GO:
        df1.insert(len(df1.columns),i,0)
        df2.insert(len(df2.columns),i,0)
        
    def fill_matrix(row):
        if(pd.isna(row['ancestor'])):
            return row
        for i in row['ancestor'].split(','):
            row[i]=1
        return row
    
    df1 = df1.apply(lambda row:fill_matrix(row), axis=1)
    df2 = df2.apply(lambda row:fill_matrix(row), axis=1)
    df1 = df1.drop(columns=['ancestor'])
    df2 = df2.drop(columns=['ancestor'])
    return df1,df2


# In[ ]:


# Site矩阵
def FS_matrix(df1,df2):
    """
    df1:cv training set
    df2:cv test set
    """
    FS = []
    for index,row in df1.iterrows():
        if(pd.isna(row['site'])):
            continue
        for i in row['site'].split(','):
            if i not in FS:
                FS.append(i)
    for index,row in df1.iterrows():
        if(pd.isna(row['site'])):
            continue
        for i in row['site'].split(','):
            if i not in FS:
                FS.append(i)
    
    FS.sort()
    
    for i in FS:
        df1.insert(len(df1.columns),i,0)
        df2.insert(len(df2.columns),i,0)
        
    def fill_matrix(row):
        if(pd.isna(row['site'])):
            return row
        for i in row['site'].split(','):
            row[i]=1
        return row
    
    df1 = df1.apply(lambda row:fill_matrix(row), axis=1)
    df2 = df2.apply(lambda row:fill_matrix(row), axis=1)
    df1 = df1.drop(columns=['site'])
    df2 = df2.drop(columns=['site'])
    return df1,df2

def tolerance_metrics(y_true, y_pre):
    label = pd.DataFrame({'true': y_true, 'pre': y_pre})

    unique_state = label.true.unique()
    targets = {}
    state_map = {1: 'p', 0: 'n', '0': 'p', '0': 'n'}
    tp = fp = tn = fn = 0
    for i, (t, p) in label.iterrows():
        if t == 0 and p == 0:
            tn += 1
        if t == 0 and p == 1:
            fp += 1
        if t == 1 and p == 1:
            tp += 1
        if t == 1 and p == 0:
            fn += 1

    allp = tp + fn
    alln = fp + tn


    N = tp + tn + fp + fn
    # ppv
    ppv = tp / (tp + fp)
    # npv
    npv = tn / (tn + fn)
    # sensitivity -> TPR
    sen = tp / (tp + fn)
    # spciticity -> TNR
    spe = tn / (tn + fp)
    # acc
    acc = (tp + tn) / N
    # MCC
    mcc = (tp*tn-fp*fn) /(((tp+fp) * (tp+fn) * (tn+fp) * (tn+fn))**0.5)
    # OPM
    opm = (ppv+npv)*(sen+spe)*(acc+(1+mcc)/2)/8
    columns = ['tp', 'tn', 'fp', 'fn', 'ppv', 'npv', 'tpr', 'tnr', 'acc', 'mcc', 'opm', 'N']
    res = pd.DataFrame(
        [
            [tp, tn, fp, fn, ppv, npv, sen, spe, acc, mcc, opm, N]
        ], 
        columns=columns,
    )
    

    return res.T


def no_reject(X_train, y_train, X_test, y_test, model, selected, filename):
    if(selected == 10):
        rfe = joblib.load('Feature_selected_1/lightgbm_feature_select_10.rfe')
        model.fit(pd.DataFrame(rfe.transform(X_train)), y_train)
        y_pred = model.predict(pd.DataFrame(rfe.transform(X_test)))
    elif(selected == 20):
        rfe = joblib.load('Feature_selected_1/lightgbm_feature_select_20.rfe')
        model.fit(pd.DataFrame(rfe.transform(X_train)), y_train)
        y_pred = model.predict(pd.DataFrame(rfe.transform(X_test)))
    elif(selected == 50):
        rfe = joblib.load('Feature_selected_1/lightgbm_feature_select_50.rfe')
        model.fit(pd.DataFrame(rfe.transform(X_train)), y_train)
        y_pred = model.predict(pd.DataFrame(rfe.transform(X_test)))
    elif(selected == 100):
        rfe = joblib.load('Feature_selected_1/lightgbm_feature_select_100.rfe')
        model.fit(pd.DataFrame(rfe.transform(X_train)), y_train)
        y_pred = model.predict(pd.DataFrame(rfe.transform(X_test)))
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    performance = tolerance_metrics(y_test, y_pred)
    performance.to_csv('out/{}'.format(filename))

# 归一化
# 最小最大值归一化
def minmaxscaler(data):
    min = np.amin(data)
    max = np.amax(data)    
    return (data - min)/(max-min)

# Z值归一化
def feature_normalize(data):
    mu = np.mean(data,axis=0)
    std = np.std(data,axis=0)
    return (data - mu)/std