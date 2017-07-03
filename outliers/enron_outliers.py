#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot as plt
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "rb"))
features = ["salary", "bonus"]
total = data_dict.pop('TOTAL', 0)  # remove the key-value ('TOTAL') paired from a dictionary and return it.
# print(total)  # for debugging
data = featureFormat(data_dict, features)


### your code below
salaries, bonuses = [], []
for point in data:
    salaries.append(point[0])
    bonuses.append(point[1])

plt.scatter(salaries, bonuses)
plt.xlabel("salary")
plt.ylabel("bonus")
plt.show()

# To Find outlier
for key, value in data_dict.items():
    if value['bonus'] == data.max():
        print(key)
# result of finding the outlier was 'TOTAL'


# To find the other 4 points
bonuses = [data_dict[key]['salary'] for key in data_dict.keys()]
salaries_with_keys = zip(data_dict.keys(), bonuses)
# removes nan from the list
salaries_without_nans = []
for bonus in salaries_with_keys:
    if str(bonus[1]).lower() == 'nan':
        continue
    salaries_without_nans.append((bonus[0], bonus[1]))

salaries_without_nans = sorted(salaries_without_nans, key=lambda x: x[1], reverse=True)
largest_4_points = salaries_without_nans[:4]
print('***********\n', largest_4_points)
