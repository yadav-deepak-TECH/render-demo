# %%
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %%
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

port_stem = PorterStemmer()

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# %%
df = pd.read_csv('/kaggle/input/sentiment140/training.1600000.processed.noemoticon.csv',encoding='ISO-8859-1')

# %%
df.head()

# %%
df.shape

# %%
df.duplicated().sum()



# %%
df.duplicated().sum()


# %%
df.isnull().sum().sum()


# %%
print(stopwords.words('english'))

# %%
# Naming the columns
column_names = ['target', 'id', 'date', 'flag', 'user', 'text']

# Reading the dataset
df = pd.read_csv('/kaggle/input/sentiment140/training.1600000.processed.noemoticon.csv', 
                 names=column_names, 
                 encoding='ISO-8859-1')


# %%
df.replace({'target':{4:1}},inplace=True)

# %%
# checking the distribution of target column
df['target'].value_counts()

# %%
print(df['target'])

# %%
port_stem = PorterStemmer()

# %%
def stemming(content):

    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if word not in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)

    return stemmed_content


# %%
df['stemmed_content'] = df['text'].apply(stemming)

# %%
df.head()

# %%
print(df['stemmed_content'])

# %%
print(df['target'])

# %%
# seprating the data and the label
x = df['stemmed_content'].values
y = df['target'].values

# %%
print(x)

# %%
print(y)

# %%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=2)

# %%
print(x.shape, x_train.shape, x_test.shape)

# %%
print(x_train)

# %%
print(x_test)

# %%
print(type(x_train))
print(x_train[:5])


# %%
# For sparse matrices, use the shape attribute to get the number of samples
print(f"Number of samples in x_train: {x_train.shape[0]}")
print(f"Number of samples in y_train: {len(y_train)}")



# %%
# converting the textual data to numercal data

vectorizer = TfidfVectorizer()


# Fit on training data and transform
x_train = vectorizer.fit_transform(x_train)

# Only transform on test data (don't fit again)
x_test = vectorizer.transform(x_test)

# %%
print(x_train)

# %%
print(x_test)

# %%
model = LogisticRegression(max_iter=1000)

# %%
# For sparse matrices, use the shape attribute to get the number of samples
print(f"Number of samples in x_train: {x_train.shape[0]}")
print(f"Number of samples in y_train: {len(y_train)}")




# %%
model.fit(x_train, y_train)

# %%
#acuracy score on the training data
x_train_prediction = model.predict(x_train)
training_data_accuracy = accuracy_score(x_train_prediction, y_train)

# %%
print('Accuracy score on the training data : ', training_data_accuracy)

# %%
import pickle

# Load model
with open('/kaggle/working/sentiment_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load vectorizer
with open('/kaggle/working/vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)


# %%
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()

def predict_sentiment(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    text = [stemmer.stem(word) for word in text if word not in stopwords.words('english')]
    text = ' '.join(text)
    text = [text]
    text = vectorizer.transform(text)
    sentiment = model.predict(text)
    if sentiment == 0:
        return "Negative"
    else:
        return "Positive"


# %%
# Take user input
tweet = input("Enter your tweet: ")  # user input
result = predict_sentiment(tweet)
print(f"Sentiment: {result}")

# Load the model
with open('sentiment_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the vectorizer
with open('vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)



