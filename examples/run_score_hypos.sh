#!/usr/bin/env bash

set -o nounset                              # Treat unset variables as an error
set -e

../scripts/score_hypos.py \
  ../labels/all/askwomen.comment_label.test.tsv \
  askwomen.test.hypos.tsv \
  askwomen.test.scores.tsv


