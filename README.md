A Factored Neural Network Model for Characterizing Online Discussions in Vector Space
=================

This repository is for the data and model used by the paper
[A Factored Neural Network Model for Characterizing Online Discussions in Vector Space](https://arxiv.org/).
```
@InProceedings{Cheng2017EMNLP,
  author    = {Hao Cheng and Hao Fang and Mari Ostendorf},
	title     = {A Factored Neural Network Model for Characterizing Online Discussions in Vector Space},
	booktitle = {Proc. Empirical Methods in Natural Language Processing (EMNLP)},
	year      = {2017}
}
```
## Labels ##
We provide two versions of dataset: *all* and *subsampled*.
The *submsampled* dataset is created to balance the data points for different labels.
Note only the train and validation sets are subsampled; the test set are identical for *all* and *subsampled*.

The folder `labels` stores the comment labels.
Each file is tab-separated.
* id: the comment id
* link_id: the link id of the comment
* author: the author name of the comment
* label: the quantized karma label of the comment
* score: the karma score of the comment
* normalized_score: a normalized karma score (not used)

## Data ##

To retrieve the comments, you can use one of the following methods:
* [Reddit Comment Dataset](https://redd.it/3bxlg7) (we use this dataset in the paper)
* [Google BigQuery](https://bigquery.cloud.google.com/table/fh-bigquery:reddit_comments.2016_05)
* [Reddit API](https://www.reddit.com/dev/api)

## Features ##

For convinience, we put the content-agnostic (graph) features under `features`.
Each file is tab-separated.
* id: the comment id
* subtree_height: the height of the comment subtree
* subtree_size: the size of the comment subtree
* author_comment_count: the number of comments made by the author within the post
* author_is_op: the author of the comment is also the author of the post
* comment_tree_depth: the depth the comment subtree
* num_children: the number of direct reply comments to the comment
* num_later_comments: the number of comments within the post made after the comment
* num_previous_comments: the number of comments within the post made before the comment
* num_siblings: the number of sibling comments of the comment
* time_since_parent: time elapsed since the parent comment
* time_since_root: time elapsed since the post
* nchildren_div_sqrtR: the num_children divided by the square root of the rank
	of the comment in terms of num_children within the post
* nchildren_minus_mean: the number_children divided by the mean of num_children
	for all comments within the post
* treesz_div_sqrtR: the subtree_size divided by the square root of the rank
	of the comment in terms of subtree_size within the post
* treesz_minus_mean: the subtree_size divided by the mean of subtree_size
	for all comments within the post

## Evaluation ##
	
We provide the scoring script `scripts/score_hypos.py`
for computing the F1 scores (level{1,2,..,7}.fscore) for individual subtasks, as well as 
F1 score averaged over all subtasks (adjust_f1_macro).

An example usage of the script can be found in `examples/run_score_hypos.sh`.
