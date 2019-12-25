import sys
import glob
import json
import os
import time
from Bleu import *

def rounder(num): #保留两位小数，四舍五入
    return round(num, 2)

def bleu_max_over_ground_truths(prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = cal_bleu([prediction], [ground_truth])
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)

def cal_bleu(infer, ref):
    while True:  # 就是为了发生异常之后反复试一试的，如果执行成功会执行return，跳出方法，死循环自然会跳出
        try:
            bleu_score = moses_multi_bleu(infer, ref)
            return bleu_score
        except FileNotFoundError:
            print("Failed to test bleu_score. Sleeping for %i secs...", 3)
            time.sleep(3)

def eval_bleu(systems, refs):
    bleu = total = 0
    assert len(systems) == len(refs), "the length of predicted span and ground_truths span should be same"

    for i, pre in enumerate(systems):
        bleu += bleu_max_over_ground_truths(pre, refs[i])
        total += 1

    bleu = bleu / total

    return {'BLEU':bleu}


