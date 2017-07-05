#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "rb"))

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list, sort_keys='../tools/python2_lesson14_keys.pkl')
labels, features = targetFeatureSplit(data)



### your code goes here 

# splitting the data into train/test sets
from sklearn.model_selection import train_test_split

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

from sklearn.metrics import accuracy_score
acc = accuracy_score(labels_test, pred)
print(acc)

# Number Of POIs In Test Set
poi_num = [ii for ii in labels_test if ii == 1]
poi_ppl = len(poi_num)
print('Number Of POIs In Test Set:', poi_ppl)

# Number Of People In Test Set
total_ppl = len(labels_test)
print('Number Of People In Test Set:', total_ppl)

# Accuracy Of A Biased Identifier
acc = (total_ppl - poi_ppl) / total_ppl
print('Accuracy Of A Biased Identifier:', acc)

# Do we have true positives (ones in both predictions and test labels)
# print('Predictions\n', pred)
# print('Test labels\n', labels_test)
true_positives = [ii for index, ii in enumerate(pred) if pred[index] == labels_test[index] == 1]
print('True Positives in the original Dataset', len(true_positives))

# Compute precision_score and recall_score
from sklearn.metrics import precision_score, recall_score
pre_score = precision_score(labels_test, pred)
recall_scr = recall_score(labels_test, pred)
print('Precision Score:', pre_score)
print('Recall Score:', recall_scr)

# Get the true positives, true negatives, false positives and false negatives
"""
True Positives ==> true value is 1 and predicted as 1
False Positives ===> true value is 0 and predicted as 1
True Negatives ==> true value is 0 and predicted as 0
False Negatives ===> true value is 1 and predicted as 0
"""

predictions = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1]
true_labels = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]

true_pos, true_neg, false_pos, false_neg = 0, 0, 0, 0
for index, ii in enumerate(predictions):
    if predictions[index] == 0:
        if true_labels[index] == 0:
            true_neg += 1
        elif true_labels[index] == 1:
            false_neg += 1
    elif predictions[index] == 1:
        if true_labels[index] == 0:
            false_pos += 1
        elif true_labels[index] == 1:
            true_pos += 1

print('True Positives: ', true_pos)
print('True Negatives: ', true_neg)
print('False Positives: ', false_pos)
print('False Negatives: ', false_neg)

# Precision and recall
"""
precision = true_positives / (true_positives + false_positives)
recall = true_positives / (true_positives + false_negatives)
"""

precision = true_pos / (true_pos + false_pos)
recall = true_pos / (true_pos + false_neg)

print('Precision: ', precision)
print('Recall:', recall)


