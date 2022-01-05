import os
import json
import numpy as np

abs_path = os.path.join(os.path.dirname(__file__), "config.json")

with open(abs_path) as file:
    config = json.load(file)


def get_attribute(name, default_value=None):
    try:
        return config[name]
    except KeyError:
        return default_value


def get_items_total(data_path):
    # items_set = set()
    maxn = -1
    with open(data_path, 'r') as f:
        data_dict = json.load(f)
        for key in ["train", "valid", "test"]:
            data = data_dict[key]
            for user in data:
                for basket in data[user]:
                    for item in basket:
                        maxn = max(maxn, int(item))
    return maxn + 1

def get_maxlen_basket(data_path):
    maxn = 0
    with open(data_path, 'r') as f:
        data_dict = json.load(f)
        for key in ["train", "valid", "test"]:
            data = data_dict[key]
            for user in data:
                for basket in data[user]:
                    if len(basket) > maxn:
                        maxn = len(basket)
    return maxn + 1


def get_codes_frequency(data_path):
    num_dim = get_items_total(data_path)
    result_vector = np.zeros(num_dim)
    with open(data_path, 'r') as f:
        data_dict = json.load(f)
        for key in ['train', 'valid', 'test']:
            data = data_dict[key]
            for user in data:
                for basket in data[user]:
                    for item in basket:
                        result_vector[item] += 1
    weights = np.zeros(num_dim)
    for idx in range(len(result_vector)):
        if result_vector[idx] > 0:
            weights[idx] = 1

        else:
            weights[idx] = 0
    weights[0] = 0
    return weights



config['items_total'] = get_items_total('./data/length7+_cleaned_cp_data_train_valid_test.json')
config['basket_maxlen'] = get_maxlen_basket('./data/length7+_cleaned_cp_data_train_valid_test.json')
config['weights'] = get_codes_frequency('./data/length7+_cleaned_cp_data_train_valid_test.json')

