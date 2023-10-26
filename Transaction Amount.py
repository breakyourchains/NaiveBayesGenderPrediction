# -*- coding: utf-8 -*-
import pymongo
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

client = MongoClient("mongodb://localhost:27017/")
database = client['local']
column = database['Customer Transactions']
nameColumn = database['NameSurname']


dataNew = pd.DataFrame(list(column.find()))
dataNew = dataNew.drop("_id", axis=1)
dataNew = dataNew.drop("Gender", axis=1)
dataNew['Gender'] = ''


data = pd.DataFrame(list(column.find()))
data = data.drop("_id", axis=1)
data = data.drop("Gender", axis=1)
data = data.drop("Surname", axis=1)
data = data.drop("Birthdate", axis=1)
data = data.drop("Transaction Amount", axis=1)
data = data.drop("Date", axis=1)
data = data.drop("Merchant Name", axis=1)
data = data.drop("Customer ID", axis=1)

data['Gender'] = ''
dataName = pd.DataFrame(list(nameColumn.find()))
dataName = dataName.drop("_id",axis=1)
dataName = dataName.drop("Count",axis=1)
dataName = dataName.drop("Probability",axis=1)


#Prediction for genders with Naive-Bayes

train_data = dataName
train_data['Name'] = train_data['Name'].astype(str)

tfidf_vectorizer = TfidfVectorizer()

X = tfidf_vectorizer.fit_transform(train_data['Name'])
y = train_data[['Gender']]


x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)


model = MultinomialNB()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)


accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Doğruluk: {accuracy:.2f}")
print("Classification Report")
print(report)


#------------------------------


def predict_gender(name):
    name_vector = tfidf_vectorizer.transform([name])
    
    # Modelle tahmin yapın
    Gender = model.predict(name_vector)[0]
    return Gender


data['Gender'] = data['Name'].apply(predict_gender)
dataNew['Gender'] = data['Gender']

#------------------------------
#Data visualization

sns.pairplot(dataNew)

sns.distplot(dataNew['Transaction Amount'])

sns.kdeplot(dataNew['Transaction Amount'], shade=True)

sns.histplot(data=dataNew, x='Transaction Amount', hue='Gender', kde=True)
plt.title("Cinsiyetlere Göre Histogram")

male_data = dataNew[dataNew['Gender']== 'M']
female_data = dataNew[dataNew['Gender'] == 'F']

#Male datas
male_data['Date'] = pd.to_datetime(male_data['Date'], format='%Y-%m-%d')
male_data.sort_values(by="Date", inplace=True)
male_transaction_amount = male_data['Transaction Amount']
transaction_date_male = male_data['Date']

#Female datas
female_data['Date'] = pd.to_datetime(female_data['Date'], format='%Y-%m-%d')
female_data.sort_values(by="Date", inplace=True)
female_transaction_amount = female_data['Transaction Amount']
transaction_date_female = female_data['Date']


#Male Diagram
plt.figure(figsize=(12, 6))  
plt.plot(transaction_date_male, male_transaction_amount, label='Transaction Amount', color='blue')
plt.title('Erkeklerin Transaction Amount Zaman Serisi')
plt.xlabel('Tarih')
plt.ylabel('Transaction Amount')
plt.grid(True)
plt.legend()
plt.xticks(rotation=45)

plt.show()


#Female Diagram
plt.figure(figsize=(12, 6))  
plt.plot(transaction_date_female, female_transaction_amount, label='Transaction Amount', color='Red')
plt.title('Kadınların Transaction Amount Zaman Serisi')
plt.xlabel('Tarih')
plt.ylabel('Transaction Amount')
plt.grid(True)
plt.legend()
plt.xticks(rotation=45)

plt.show()


