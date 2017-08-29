#!/usr/bin/env python2

from __future__ import division

__version__ = "0.1"
__author__ = "Hao Fang"
__email__ = "hfang@uw.edu"

import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

def adjust_f1_score(ytrue, ypred, average=None):
    assert average in [None, 'macro', 'weighted']
    list_f1 = []
    for level in range(7):
        tmp_ytrue = ytrue.copy()
        tmp_ypred = ypred.copy()
        mask_pos = (ytrue > level)
        tmp_ytrue[mask_pos] = 1
        tmp_ytrue[~mask_pos] = 0
        mask_pos = (ypred > level)
        tmp_ypred[mask_pos] = 1
        tmp_ypred[~mask_pos] = 0

        fscore = f1_score(tmp_ytrue, tmp_ypred)
        list_f1.append(fscore)

    if average == None:
        return list_f1
    elif average == 'macro':
        return np.mean(np.array(list_f1))
    elif average == 'weighted':
        averaged_f1 = 0.0
        for level, f1 in enumerate(list_f1):
            averaged_f1 += f1 * (level + 1)
        averaged_f1 /= np.sum(np.arange(1, 8))
        return averaged_f1

def score_hypos(label_tsv, hypo_tsv):
    label_df = pd.read_csv(label_tsv, sep='\t', index_col='id')
    hypo_df = pd.read_csv(hypo_tsv, sep='\t', index_col='id')
    list_comment_id = label_df.index.tolist()

    ytrue = label_df.ix[:, 'label']
    ypred = hypo_df.loc[list_comment_id, 'hypo']

    accuracy = accuracy_score(ytrue, ypred)

    adjust_fscore = adjust_f1_score(
        ytrue,
        ypred,
        average=None
    )
    assert len(adjust_fscore) == 7

    adjust_fscore_weighted = adjust_f1_score(
        ytrue,
        ypred,
        average='weighted'
    )

    list_score_row = []

    list_score_row.append({'metric': 'accuracy', 'value': accuracy})
    list_score_row.append({
        'metric': 'adjust_f1_macro',
        'value': np.mean(np.array(adjust_fscore))
    })
    list_score_row.append({
        'metric': 'adjust_f1_weighted',
        'value': adjust_fscore_weighted
    })
    for level in range(7):
        list_score_row.append({
            'metric': 'level{0}.fscore'.format(level + 1),
            'value': adjust_fscore[level]
        })
    score_df = pd.DataFrame(list_score_row)

    return score_df


def main():
    import argparse

    pa = argparse.ArgumentParser(description='score hypos for hierachical labels')
    pa.add_argument('label_tsv', help='label tsv')
    pa.add_argument('hypo_tsv', help='hypo tsv')
    pa.add_argument('score_tsv', help='hypo tsv')
    args = pa.parse_args()

    score_df = score_hypos(args.label_tsv, args.hypo_tsv)
    score_df.to_csv(args.score_tsv, sep='\t', index=False)

if __name__ == '__main__':
    main()
