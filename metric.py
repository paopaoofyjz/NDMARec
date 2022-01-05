import torch
import numpy as np
import torch
import torch.nn.utils.rnn as rnn_utils
from tqdm import tqdm
from load_config import get_attribute
from utils import get_txt_mask


def dcg(truths, preds, top_k):
    if len(preds) > top_k:
        preds = preds[:top_k]
    dcg = 0
    for idx, pred in enumerate(preds):
        if np.isin(pred, truths):
            dcg += (1 / np.log2(2 + idx))
    return dcg


def ndcg_score(y_true, y_pred, top_k):
    scores = []
    for truth_value, predict_value in zip(y_true, y_pred):
        predict_order_value, predict_order_index = torch.sort(predict_value, descending=True)
        truth_index = torch.nonzero(truth_value).flatten().cpu().numpy()
        predict_index = predict_order_index[:top_k].cpu().numpy()

        dcg_score = dcg(truth_index, predict_index, top_k)
        idcg_score = dcg(truth_index, truth_index, top_k)
        scores.append(dcg_score / idcg_score)

    return np.mean(scores)


def f1_score(y_true, y_pred, top_k):

    TP, TN, FP = 0, 0, 0
    for truth_value, predict_value in zip(y_true, y_pred):
        predict_order_value, predict_order_index = torch.sort(predict_value, descending=True)
        truth_index = torch.nonzero(truth_value).flatten()
        predict_index = predict_order_index[:top_k]

        truth_set = set(truth_index.tolist())
        predict_set = set(predict_index.tolist())

        TP += len(truth_set & predict_set)  # 实际为真，预测也为真
        TN += len(truth_set - predict_set)  # 实际为真，预测为假
        FP += len(predict_set - truth_set)  # 实际为假，预测为真

    precision = TP / (TP + FP)  # 除以所有预测为真
    recall = TP / (TP + TN)  # 除以所有实际为真
    if precision == 0 and recall == 0:
        score = 0
    else:
        score = 2 * precision * recall / (precision + recall)
    return score


def f1_score_mean(y_true, y_pred, top_k):

    f1_score_list = []
    for truth_value, predict_value in zip(y_true, y_pred):
        predict_order_value, predict_order_index = torch.sort(predict_value, descending=True)
        truth_index = torch.nonzero(truth_value).flatten()
        predict_index = predict_order_index[:top_k]

        truth_set = set(truth_index.tolist())
        predict_set = set(predict_index.tolist())

        TP = len(truth_set & predict_set)  # 实际为真，预测也为真
        TN = len(truth_set - predict_set)  # 实际为真，预测为假
        FP = len(predict_set - truth_set)  # 实际为假，预测为真

        precision = TP / (TP + FP)  # 除以所有预测为真
        recall = TP / (TP + TN)  # 除以所有实际为真
        if precision == 0 and recall == 0:
            score = 0
        else:
            score = 2 * precision * recall / (precision + recall)
        f1_score_list.append(score)
    return np.mean(f1_score_list)


def recall_score(y_true, y_pred, top_k):

    recall_list = []
    for truth_value, predict_value in zip(y_true, y_pred):
        predict_order_value, predict_order_index = torch.sort(predict_value, descending=True)
        truth_index = torch.nonzero(truth_value).flatten()
        predict_index = predict_order_index[:top_k]

        truth_set = set(truth_index.tolist())
        predict_set = set(predict_index.tolist())

        TP = len(truth_set & predict_set)  # 实际为真，预测也为真
        TN = len(truth_set - predict_set)  # 实际为真，预测为假

        recall = TP / (TP + TN)  # 除以所有实际为真
        recall_list.append(recall)
    return np.mean(recall_list)

def PHR(y_true, y_pred, top_k):
    """
        Args:
            y_true: tensor (samples_num, items_total)
            y_pred: tensor (samples_num, items_total)
            top_k: int
        Returns:
            score: float
        """
    hit_number = 0
    data_length = len(y_true)
    for truth_value, predict_value in zip(y_true, y_pred):
        predict_order_value, predict_order_index = torch.sort(predict_value, descending=True)
        truth_index = torch.nonzero(truth_value).flatten()
        predict_index = predict_order_index[:top_k]

        truth_set = set(truth_index.tolist())
        predict_set = set(predict_index.tolist())

        if len(truth_set.intersection(predict_set)) > 0:
            hit_number += 1

    return hit_number / data_length


def get_metric(y_true, y_pred):
    result = {}
    for top_k in [5, 10, 15, 20, 25]:
        result.update({
            f'f1_score_@K{top_k}': format(f1_score(y_true, y_pred, top_k=top_k),'.4f'),
            f'f1_score_mean_@K{top_k}': format(f1_score_mean(y_true, y_pred, top_k=top_k),'.4f'),
            f'recall_@K{top_k}': format(recall_score(y_true, y_pred, top_k=top_k),'.4f'),
            f'ndcg_@K{top_k}': format(ndcg_score(y_true, y_pred, top_k=top_k),'.4f'),
            f'PHR_@K{top_k}': format(PHR(y_true, y_pred, top_k=top_k),'.4f')
        })

    return result
























