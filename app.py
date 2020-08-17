#import all necessary LIBRARIES

#flask web framework helps to deploy the model
from flask import Flask,render_template,url_for,request


import numpy as np
import pandas as pd
from nltk.stem.porter import PorterStemmer
import re
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

#regular expression (or "regex") is a find_all pattern
def remove_pattern(input_txt,pattern):
    r=re.findall(pattern,input_txt)
    for i in r:
        input_txt=re.sub(i,"",input_txt)
    return input_txt

#punctuation
def count_punct(text):
    count=sum([1 for char in text if char in string.punctuation])
    return round(count/(len(text) - text.count(" ")),3)*100

#Flask looks for templates in the subdirectory named template
app=Flask(__name__,template_folder='template')

#import the data
data=pd.read_csv(r"flipkart_cleandata .csv",index_col=None)

data.columns=["body_text","label"]
#print(data["label"])
#print(data["body_text"])

#feature the labels data

data["label"]=data["label"].map({"CLOTHING": 0 ,"ENTERTAINMENT":1,"FURNITURE":2,"LAPTOP":3,"MOBILES":4})
data["tidy_text"]=np.vectorize(remove_pattern)(data['body_text'],"@[\w]*")#re replace non word character with ""
data["label"].fillna(0,inplace=True)
#print(data["label"])

#tokenize the word
tokenized_text = data['tidy_text'].apply(lambda x: x.split())
#print(tokenized_text)

#we do stemming to get root word
stemmer = PorterStemmer()
tokenized_text = tokenized_text.apply(lambda x: [stemmer.stem(i) for i in x])
for i in range(len(tokenized_text)):
    tokenized_text[i] = ' '.join(tokenized_text[i])
data['tidy_name'] = tokenized_text
data['body_len'] =data['body_text'].apply(lambda x:len(x) - x.count(" "))
data['punct%'] =data['body_text'].apply(lambda x:count_punct(x))

#assign dependent and independent value
X = data['tidy_name']
y = data['label']
#X.isnull().sum()

#print(y)

# Extract Feature With CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(X) # Fit the Data

X = pd.concat([data['body_len'],data['punct%'],pd.DataFrame(X.toarray())],axis = 1)

#print(X)

#train and test data split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#print(X_train.shape,y_train.shape)

#Using classifier
from sklearn.naive_bayes import MultinomialNB
NB_classifier = MultinomialNB()
NB_classifier.fit(X_train, y_train)

y_pred= NB_classifier.predict(X_test)

#accuracy of the model

from sklearn.metrics import accuracy_score

#print("Accuracy:",accuracy_score(y_test,y_pred))

#confusion matrix

from sklearn.metrics import confusion_matrix
#print(confusion_matrix(y_test,y_pred))
from sklearn.metrics import classification_report
#print(classification_report(y_test,y_pred))

"""


clf = LogisticRegression(C=0.1, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=None, solver='liblinear', tol=0.0001, verbose=0,
                   warm_start=False)
clf.fit(X,y)

"""
#route to html file in template
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = pd.DataFrame(cv.transform(data).toarray())
        body_len = pd.DataFrame([len(data) - data.count(" ")])
        punct = pd.DataFrame([count_punct(data)])
        total_data = pd.concat([body_len,punct,vect],axis = 1)
        my_prediction = NB_classifier.predict(total_data)
    return render_template('result.html',prediction = my_prediction)

if __name__ == '__main__':
    app.run(debug=True)
