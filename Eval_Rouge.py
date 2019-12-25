import sys
import glob
import json
import os
import time
from Rouge import *

def rounder(num): #保留两位小数，四舍五入
    return round(num, 2)

def rouge_max_over_ground_truths(prediction, ground_truths):
    scores_for_rouge1 = []
    scores_for_rouge2 = []
    scores_for_rougel = []
    for ground_truth in ground_truths:
        score = cal_rouge([prediction], [ground_truth])
        scores_for_rouge1.append(score[0])
        scores_for_rouge2.append(score[1])
        scores_for_rougel.append(score[2])
    return max(scores_for_rouge1), max(scores_for_rouge2), max(scores_for_rougel)

def cal_rouge(infer, ref):
    x = rouge(infer, ref)
    return x['rouge_1/f_score'] * 100, x['rouge_2/f_score'] * 100, x['rouge_l/f_score'] * 100

def eval_rouge(systems, refs):
    rouge_1 = rouge_2 = rouge_l = total = 0
    assert len(systems) == len(refs), "the length of predicted span and ground_truths span should be same"

    for i, pre in enumerate(systems):
        rouge_result = rouge_max_over_ground_truths(pre, refs[i])
        rouge_1 += rouge_result[0]
        rouge_2 += rouge_result[1]
        rouge_l += rouge_result[2]
        total += 1

    rouge_1 = rouge_1 / total
    rouge_2 = rouge_2 / total
    rouge_l = rouge_l / total

    return {'ROUGE_1_F1':rouge_1, 'ROUGE_2_F1':rouge_2, 'ROUGE_L_F1':rouge_l}


