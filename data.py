import json
from functools import partial
import csv
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F
from utils import get_txt_mask
from load_config import get_attribute
from model import Item2CPEncoder, CPEncoder, Generator, Discriminator, CPDecoder
from txt_encoder import return_txt_data
import pandas as pd
import numpy as np
import random



class SetDataset(Dataset):
    def __init__(self, data_path, basket_maxlen, key=None):

        with open(data_path, 'r') as file:
            data_dict = json.load(file)

        self.data_list = []
        self.length_list = []

        data = data_dict[key]
        for user in data:
            tmp = {}
            length_tmp = {}
            lengths = []
            value = data[user]
            new_value = []
            for basket in value:
                lengths.append(len(basket))
                new_basket = [int(item) for item in basket]
                new_basket = list(np.sort(new_basket))
                if len(new_basket) <= basket_maxlen:
                    new_basket.extend([0 for i in range(basket_maxlen - len(new_basket))])
                new_value.append(new_basket)
            tmp[user] = new_value
            length_tmp[user] = lengths
            self.data_list.append(tmp)
            self.length_list.append(length_tmp)

    def __getitem__(self, index):
        return self.data_list[index], self.length_list[index]

    def __len__(self):
        return len(self.data_list)


def collate_fn_set(data_list, return_sequence):

    user_list = []
    train_list = []
    truth_list = []
    lengths_list = []
    frequency_dic = {}
    for tup in data_list:
        dic = tup[0]
        length = tup[1]
        user = list(dic.keys())[0]
        baskets = list(dic.values())[0]
        lengths = list(length.values())[0]

        user_list.append(user)
        train = baskets[:-1]
        frequency_dic[user] = {}
        for pbask in train:
            for pitem in pbask:
                if pitem == 0:
                    break
                if pitem not in frequency_dic[user]:
                    frequency_dic[user][pitem] = 1
                if pitem in frequency_dic[user]:
                    frequency_dic[user][pitem] += 1


        truth = baskets[-1]
        new_truth = []
        for target in truth:
            if target != 0:
                new_truth.append(target)
        train_list.append(torch.tensor(train))
        truth_list.append(new_truth)
        lengths_list.append(lengths)

    return user_list, train_list, truth_list, lengths_list, frequency_dic



def get_data_loader(data_path, data_type, batch_size, return_sequence, split=1):

    if split == 1:
        collate_fn = partial(collate_fn_set, return_sequence=return_sequence)
        dataset = SetDataset(data_path, get_attribute('basket_maxlen'), key=data_type)
        print(f'{data_type} data length -> {len(dataset)}')
        data_loader = DataLoader(dataset=dataset,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 collate_fn=collate_fn)
    elif split == 2:
        collate_fn = partial(collate_fn_set_time, data_type=data_type, return_sequence=return_sequence)
        dataset = SetDataset(data_path)
        data_loader = DataLoader(dataset=dataset,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 collate_fn=collate_fn)
    else:
        raise NotImplementedError()
    return data_loader
















