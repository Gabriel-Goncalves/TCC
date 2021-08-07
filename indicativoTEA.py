# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 20:22:59 2021

@author: gabriel
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import keras as kr
from keras.models import Sequential   
from keras.layers import Dense       
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report, confusion_matrix 

def normalizeGender(genderColumn):
    return genderColumn.str.replace('M', '1').str.replace('m', '1').str.replace('F', '0').str.replace('f', '0').astype('category')

def normalizeAge(ageColumn):
    return ageColumn.str.replace('?', 'Unknown').astype('category')

#Função irá substituir os valores na coluna ethnicity a fim de normaliza-los talvez tirar essa coluna
def nomalizeEthnicity(ethnicityColumn):
    return ethnicityColumn.str.replace('?',"Unknown").str.replace(' ','_').str.replace('\'Middle_Eastern_\'',"ME" ).str.replace('\'South_Asian\'','SA').str.replace('others','Others').astype('category')


#Troca valores de Yes por 1 e No por 0
def normalizeJundice(jundiceColumn):
    return jundiceColumn.str.replace('yes', '1').str.replace('no', '0').astype('category')


#Normaliza a coluna que tem informação se alguém da família tem Autismo de yes para 1 e de no para 0
def normalizeRelativesWithAutism(autismColumn):
    return autismColumn.str.replace('yes', '1').str.replace('Yes', '1').str.replace('no', '0').str.replace('No', '0').astype('category')


#Normaliza a coluna de range de idade trocando de 4 a 11 por 1 de 12 a 15 por 2 e de 18+ para 3
def normalizeAgeRange(age_descColumn):
    return age_descColumn.str.replace('\'4-11 years\'', '1').str.replace('\'12-16 years\'', '2').str.replace('\'12-15 years\'', '2').str.replace('\'18 and more\'', '3').astype('category')


#Normaliza coluna da pessoa que está realizando o teste talvez excluir essa coluna seja melhor
def normalizeRelation(relationColumn):
    return relationColumn.str.replace('Self', 'self').str.replace('\'Health care professional\'', 'HealthProfessional').str.replace('?', 'Unknown').astype('category') 


#Normaliza a classificação de YES para 1 e de NO para 0   
def normalizeClassification(classColumn):
    return classColumn.str.replace('YES', '1').str.replace('Yes', '1').str.replace('yes', '1').str.replace('NO', '0').str.replace('No', '0').str.replace('no', '0').astype('category')

def normalizeData(dataFrame):
    dataFrame.gender = normalizeGender(dataFrame.gender)
    dataFrame.age = normalizeAge(dataFrame.age)
    dataFrame.ethnicity = nomalizeEthnicity(dataFrame.ethnicity)
    dataFrame.jundice = normalizeJundice(dataFrame.jundice)
    dataFrame.austim = normalizeRelativesWithAutism(dataFrame.austim)
    dataFrame.age_desc = normalizeAgeRange(dataFrame.age_desc)
    dataFrame.relation = normalizeRelation(dataFrame.relation)
    dataFrame.Class = normalizeClassification(dataFrame.Class)
    dataFrame = dataFrame.drop('contry_of_res',axis=1)
    dataFrame = dataFrame.drop('used_app_before',axis=1)
    return dataFrame


def fixPredictions(predictions):
    newArray = []
    for index, case in enumerate(predictions):
        if(case[0] > case[1]):
            newArray.append([1.0, 0.0])
        else:
            newArray.append([0.0, 1.0])
    return newArray

header=['A1_Score','A2_Score','A3_Score','A4_Score','A5_Score','A6_Score','A7_Score','A8_Score','A9_Score','A10_Score','age','gender','ethnicity','jundice','austim','contry_of_res','used_app_before','result','age_desc','relation','Class']
data = pd.read_csv('allData.data', header=None)
dataFrame =  pd.DataFrame(data.values,columns=header)

dataFrame = normalizeData(dataFrame)
#chamar as 4 primeiras funções de normalizar genero, etinia, jundice e parentes com autismo


#dataSet = pd.get_dummies(dataFrame,drop_first=True,columns=['gender','ethnicity','age_desc','relation'],prefix={'gender':'gen','ethnicity':'eth','age_desc':'ageD','relation':'rel'})
#dataSet = dataFrame.select_dtypes(include=['category','int64','float64','uint8']).apply(pd.to_numeric).astype(astype)
#dataSet = dataFrame.values
#entries = (dataSet[:, 0:10]).astype('float32')
#classification = dataFrame[:, 18]
#dataFrame.drop('Class')
classification = dataFrame.Class
dataFrame = dataFrame.drop('Class', axis=1)
dataSet = pd.get_dummies(dataFrame)
dataSet = dataSet.select_dtypes(include=['category','int64','float64','uint8']).apply(pd.to_numeric).astype('float64')


encoder = LabelEncoder()
encodedClass = encoder.fit_transform(classification)
classificationEncoded = np_utils.to_categorical(encodedClass)


X_train, X_test, y_train, y_test = train_test_split(dataSet, classificationEncoded, test_size = 0.4)

model = Sequential()
model.add(Dense(15, input_dim=117, activation='relu'))
model.add(Dense(7, input_dim=15, kernel_initializer='normal', activation='relu'))
model.add(Dense(2, activation='sigmoid'))

adam=Adam(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
historico = model.fit(X_train, y_train, epochs=40)

predictions = model.predict(X_test)

predictions = fixPredictions(predictions)
predictions = np.array(predictions)


print(confusion_matrix(y_test.argmax(axis=1), predictions.argmax(axis=1)))  
print(classification_report(y_test, predictions))


plt.style.use("ggplot")
plt.figure()
plt.plot(historico.history['loss'], label = 'Loss')
plt.plot(historico.history['accuracy'], label = 'Accuracy')

plt.title('Épocas x Perda/Precisão')
plt.xlabel('Quantidade de Épocas')
plt.ylabel('Accuracy, Loss')
plt.legend()
plt.axis([0, 50, -0.1, 1.1])
plt.show()
