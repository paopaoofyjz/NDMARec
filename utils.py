import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from load_config import get_attribute


class WeightMSELoss(nn.Module):

    def __init__(self, weights=None):
        super(WeightMSELoss, self).__init__()
        self.weights = weights
        if weights is not None:
            self.weights = torch.sqrt(torch.tensor(weights))
        self.mse_loss = nn.MSELoss(reduction='sum')

    def forward(self, truth, predict):
        # predict = torch.softmax(predict, dim=-1)
        # predict = torch.sigmoid(predict)
        truth = truth.float()
        if self.weights is not None:
            self.weights = self.weights.to(truth.device)
            predict = predict * self.weights
            truth = truth * self.weights

        loss = self.mse_loss(predict, truth)
        # loss = loss.requires_grad_()
        return loss


def get_attn_pad_mask(seq_q, seq_k):        #把pad与正常的词区分开

    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()

    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  
    return pad_attn_mask.expand(batch_size, len_q, len_k)  

def get_item_pad_mask(seq_q, seq_k):

    batch_size, qbasket_num, qitem_num = seq_q.size()
    batch_size, kbasket_num, kitem_num = seq_k.size()
    pad_mask = seq_k.data.eq(0).unsqueeze(2)
    return pad_mask.expand(batch_size, qbasket_num, qitem_num, kitem_num)




def get_attn_subsequence_mask(seq):

    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1) 
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask 


def save_model(model, model_path):
    torch.save(model.state_dict(), model_path)


def get_txt_mask(txt_data, user_data):
    user_txts = []
    user_txt_masks = []
    for user in user_data:
        txt = torch.tensor(txt_data[user[4:]][0])
        mask = torch.tensor(()).new_ones(len(txt))
        mask = txt * mask
        for idx, x in enumerate(mask):
            if x > 0:
                mask[idx] = 1
        user_txts.append(txt)
        user_txt_masks.append(mask)
    user_txts = torch.stack(user_txts)
    user_txt_masks = torch.stack(user_txt_masks)
    return user_txts, user_txt_masks.unsqueeze(2)


def get_truth_data(truth_data):
    truth_list = []
    for basket in truth_data:
        one_hot_items = F.one_hot(basket, num_classes=get_attribute("items_total"))
        one_hot_basket, _ = torch.max(one_hot_items, dim=-2)
        truth_list.append(one_hot_basket)
    truth = torch.stack(truth_list)
    return truth





















