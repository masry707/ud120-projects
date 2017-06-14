#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle
import sys
sys.path.append('../tools/')
import feature_format as ff

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "rb"))
# get the size (no. of people in Enron DataSet)
print('The no. of people in Enron Dataset', len(enron_data))

# get how many features for each one of these persons
first_person = next(iter(enron_data.values()))
# print(first_person)
print('Features no. for each one of these persons', len(first_person.keys()))

# get the POI (Person of Interest) from the list
count = 0
for person_name, features_dict in enron_data.items():
    # print(features_dict)
    if features_dict["poi"] == 1:
        count += 1
print('The no. of POI: ', count)

# How Many POIs Exist?
poi_text = '../final_project/poi_names.txt'
with open(poi_text, 'r') as poi_names:
    fr = poi_names.readlines()
    print('POIs Exist no.', len(fr[2:]))

# How many stock belonging to James Prentice?
print('Total Stock Value of James Prentice:', enron_data['PRENTICE JAMES']['total_stock_value'])

# How many email messages do we have from Wesley Colwell to persons of interest?
print('From Wesley Colwell to persons of interest:', enron_data['COLWELL WESLEY']['from_this_person_to_poi'])

# What’s the value of stock options exercised by Jeffrey K Skilling?
print('The value of stock options exercised by Jeffrey K Skilling:',
      enron_data['SKILLING JEFFREY K']['exercised_stock_options'])

# Sort values
# print(sorted(enron_data.keys()))


# Of these three individuals (Lay, Skilling and Fastow),
# who took home the most money (largest value of “total_payments” feature)?


def get_max_payments():
    persons = ['LAY KENNETH L', 'SKILLING JEFFREY K', 'FASTOW ANDREW S']
    max_payment = 0
    result = {}
    for p in persons:
        total_payment = enron_data[p]['total_payments']
        if total_payment > max_payment:
            max_payment = total_payment
            result['max_payment'] = total_payment
            result['person'] = p
    return result

result_payments = get_max_payments()

print('The largest value of “total_payments” is', result_payments['max_payment'], 'took by', result_payments['person'])

# How many folks in this dataset have a quantified salary? What about a known email address?
salary_count = 0
emails_count = 0

for person in enron_data:
    if str(enron_data[person]['salary']).lower() != 'nan':
        salary_count += 1
    if str(enron_data[person]['email_address']).lower() != 'nan':
        emails_count += 1

print('Persons have quantified salary', salary_count)
print('Persons have email address', emails_count)

# Dict-to-array conversion
features_list = list(next(iter(enron_data.values())).keys())
np_result_arr = ff.featureFormat(enron_data, features_list)
target, features = ff.targetFeatureSplit(np_result_arr)
# print(target)
# print(features)


# Missing POIs 1
payments_count = 0
for person in enron_data:
    if str(enron_data[person]['total_payments']).lower() == 'nan':
        payments_count += 1

print('People who have “NaN" for their total payments', payments_count)
nan_payments_percent = (payments_count/len(enron_data.keys())) * 100
print('percentage of people in the dataset who have “NaN"', round(nan_payments_percent, 2))


# Missing POIs 2
payments_poi_count = 0
for person in enron_data:
    if str(enron_data[person]['total_payments']).lower() == 'nan' and enron_data[person]['poi'] == 1:
        payments_poi_count += 1

print('People who have “NaN" for their total payments', payments_poi_count)
nan_payments_percent = (payments_poi_count/len(enron_data.keys())) * 100
print('percentage of people in the dataset who have “NaN"', round(nan_payments_percent, 2))
