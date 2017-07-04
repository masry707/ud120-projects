#!/usr/bin/python

import pickle
import numpy
numpy.random.seed(42)


### The words (features) and authors (labels), already largely processed.
### These files should have been created from the previous (Lesson 10)
### mini-project.
words_file = "../text_learning/your_word_data.pkl" 
authors_file = "../text_learning/your_email_authors.pkl"
word_data = pickle.load(open(words_file, "rb"))
authors = pickle.load(open(authors_file, "rb"))



### test_size is the percentage of events assigned to the test set (the
### remainder go into training)
### feature matrices changed to dense representations for compatibility with
### classifier functions in versions 0.15.2 and earlier
from sklearn import model_selection
features_train, features_test, labels_train, labels_test = model_selection.train_test_split(word_data, authors, test_size=0.1, random_state=42)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                             stop_words='english')
features_train = vectorizer.fit_transform(features_train)
features_test = vectorizer.transform(features_test).toarray()

vocab = vectorizer.get_feature_names()
print(vocab[21323])  # houectect


### a classic way to overfit is to use a small number
### of data points and a large number of features;
### train on only 150 events to put ourselves in this regime
features_train = features_train[:150].toarray()
labels_train = labels_train[:150]



### your code goes here
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

# get the accuracy of the fitting
from sklearn.metrics import accuracy_score
acc = accuracy_score(labels_test, pred)
print(acc)

# get a list of the relative importance of all the features being used
# print(clf.feature_importances_)
for index, feature_importance in enumerate(clf.feature_importances_):
    if feature_importance > 0.2:
        print('The importance of that feature is:', feature_importance, 'and the Number of that feature is:', index)
        # The importance of that feature is: 0.764705882353 and the Number of that feature is: 33614

