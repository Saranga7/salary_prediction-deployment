# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 08:40:54 2020

@author: Saranga
"""

import pickle
import pandas as pd

df=pd.read_csv("hiring.csv")
df=df.iloc[:9,:]

df['experience'].fillna(0,inplace=True)
df['test_score'].fillna(df['test_score'].mean(),inplace=True)

X=df.iloc[:,:3]
y=df.salary

def convert_to_int(word):
    word_dict = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8,
                'nine':9, 'ten':10, 'eleven':11, 'twelve':12, 'zero':0, 0: 0}
    return word_dict[word]


X['experience']=X['experience'].apply(lambda x:convert_to_int(x))

from sklearn.linear_model import LinearRegression
reg=LinearRegression()

reg.fit(X,y)

pickle.dump(reg,open('model.pkl','wb'))

model=pickle.load(open('model.pkl','rb'))

print(model.predict([[1,2,3]]))