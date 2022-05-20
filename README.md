# Project_2-CommonLit_Readability
This project is on the Kaggle competition "CommonLit Readability Prize" .Rate the complexity of literary passages for grades 3-12 classroom use.
In this competition, we're predicting the reading ease of excerpts from literature. We've provided excerpts from several time periods and a wide range of reading ease scores. 
Test set includes a slightly larger proportion of modern texts (the type of texts we want to generalize to) than the training set.
This is a NLP Problem where we have to try different NLP models. Then ensemble them using various ensemble techniques.
We will be using various NLP Models like (Roberta-Base, Roberta-Large, Deberta, Electra, etc.).We also be using various fine-tuning strategies like 
Concatenate last 4 layer, Attention Head, Layer-wise Learning Rate Decay (LLRD), Frequent Evaluation.


**Files:**

train.csv - the training set

test.csv - the test set

sample_submission.csv - a sample submission file in the correct format

**Columns-**

id - unique ID for excerpt

url_legal - URL of source - this is blank in the test set.

license - license of source material - this is blank in the test set.

excerpt - text to predict reading ease of

target - reading ease

standard_error - measure of spread of scores among multiple raters for each excerpt. Not included for test data.

The target value is the result of a Bradley-Terry analysis of more than 111,000 pairwise comparisons between excerpts. Teachers spanning grades 3-12 (a majority teaching between grades 6-10) served as the raters for these comparisons.

Standard error is included as an output of the Bradley-Terry analysis because individual raters saw only a fraction of the excerpts, while every excerpt was seen by numerous raters. The test and train sets were split after the target scores and standard error were computed.


![Project_2_Train_Screenshot](https://user-images.githubusercontent.com/53327139/169512769-9bf99097-90cf-4816-86c2-a2adab668c78.png)
