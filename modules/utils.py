import os, operator

import numpy as np

#Given a field ("column"), finds the most common value
def find_most_common_values_for_each_category(labels, data, exclude_symbols = None):
    most_common_values = {}
    
    for (i, label) in labels:
        #print(i)
        #print(label)
        values_count = {}
        for j in range(len(data)):
            #print(len(data[j]))
            #print(i)
            if exclude_symbols is not None and data[j][i] in exclude_symbols: 
                #print(data[j][i])
                #print(label)
                continue
            if data[j][i] in values_count:
                values_count[data[j][i]] += 1
            else:
                values_count[data[j][i]] = 1
                
        most_common_values[label] = max(values_count.items(), key=operator.itemgetter(1))[0]

    return most_common_values

def select_labels_range_from_dict(dictionary, labels, begin, end):
    result = {}
    for i in range(begin, end):
        result[list(labels)[i]] = dictionary[list(labels)[i]]
    return result

def dict_to_numpy_array(dictionary):
    arr = np.array([value for (_, value) in dictionary.items()])
    return arr

def merge_dicts(dict1, dict2):
    return {**dict1, **dict2}

# Returns a tuple (index, field_name) for each field ("column") in the dataset that is NULL for each dataset entry (whole column is NULL)
def get_null_fields(data, labels, null_symbol = "#NULL!"):
    null_fields = []
    for (i, field) in labels:
        all_null = True
        for j, sample in enumerate(data):
            if data[j][i] != null_symbol:
                all_null = False
                break
        if all_null:
            null_fields.append((i, field))

    return null_fields

def remove_null_fields_from_data(data, labels, null_symbol = "#NULL!"):
    null_fields = get_null_fields(data, labels, null_symbol = null_symbol)
    
    removed_count = 0
    for (j, field) in null_fields:
        #print("Removing field: "+field+" ("+str(j)+")")
        for i, sample in enumerate(data):
            data[i].pop(j - removed_count)
        
        removed_count += 1

    new_labels = []
    null_fields = [f for (_, f) in null_fields]
    
    new_label_index = 0
    for (_, label) in labels:
        if label not in null_fields:
            new_labels.append((new_label_index, label))
            new_label_index += 1
        
    return data, new_labels

def replace_null_values_in_data(data, labels, null_symbol, replace_with_dict, exclude_categories=None):
    for j, sample in enumerate(data):
        for (i, field) in labels:
            if exclude_categories is None or field not in exclude_categories:
                if sample[i] == null_symbol:
                    data[j][i] = replace_with_dict[field]
    return data

#Merges an array of labels and an array of values aligned positionally in a dictionary
def merge_labels_and_values_as_dict(labels, values):
    res_dict = {}
    for (key, value) in zip(labels, values):
        res_dict[key] = value
        
    return res_dict

#Creates two "vocabularies" or maps: index -> item and item -> index
def create_indices(l):
    item_to_idx = {}
    idx_to_item = {}

    for (index, item) in enumerate(l):
        item_to_idx[item] = index
        idx_to_item[index] = item

    return item_to_idx, idx_to_item

#Loads the dataset from a csv file by performing also the following cleaning:
# 1) Removes fields ("columns") from the dataset where each entry has a #NULL! value
# 2) For those fields where only some entries are #NULL!, the #NULL! is replaced with the most common value in that field ("column")
def load_data_from_csv(dir, filename):
    with open(os.path.join(dir, filename), mode="r") as f:
        labels = f.readline().strip().split(",")
        
        #labels is a list of tuples (label_position, label)
        labels = [(i, label) for (i, label) in enumerate(labels)]
        
        data = []
        for line in f:
            line = line.strip().split(",")
            data.append(line)

        #Remove columns with only #NULL! values
        data, labels = remove_null_fields_from_data(data, labels)

        #Create the label vocabulary
        #label_to_idx, idx_to_label = create_indices(labels)

        #Find most common value in each category to replace #NULL! values
        most_common_values = find_most_common_values_for_each_category(labels, data, exclude_symbols=["#NULL!"])

#NOTICE: Three possible ways to deal with null values
        #a) NULL values appear only in whole columns (which were already removed) or only in the survey part, so it might be useful to consider making the NULL an actual possible value
        #b) Replace #NULL! values with the most common value for that category [CHOSEN ONE]
        #c) Use a probabilistic approach (compute probability for both outcomes (0 or 1) for that question and toss a coin)

        data = replace_null_values_in_data(data, labels, "#NULL!", most_common_values)

        #Remove label index
        labels = [label for (idx, label) in labels]

    return data, labels

def generate_random_int_dataset(samples, columns, int_range=(0,100), save_to_file=None):
    import pandas as pd
    import numpy as np
    import random
    
    df = pd.DataFrame(np.random.randint(int_range[0],int_range[1]),size=(samples, columns), columns=["field"+i for i in range(columns)])
    
    if save_to_file is not None:
        assert isinstance(save_to_file, str), "save_to_file should be a file name"
        assert save_to_file.endswith(".csv"), "save_to_file should be a csv file name (should end with .csv)"
        df.to_csv (save_to_file, header=True) 