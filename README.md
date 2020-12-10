# Kaggle-Toxic-Comment-Classification
## Dataset
The dataset is available :- https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge <br>
This is multi-class classification problem <br>
Total classes 6: <br>
Toxic<br>
Severe_toxic<br>
Obscene<br>
Threat<br>
Insult<br>
Identity_hate<br>

## Idea of code<br>

Word embeddings are made for every word using word2vec .<br>
They are averaged for every sentence, so for every sentence we will get some encoding <br>
Then cmeans (fuzzy clustering or soft clustering so that it can be classified in to more than one class) is used to get the membership for clusters<br>

Word embeddings are made for every word using word2vec.<br>
They are sent to RNN to get the encoding for every sentence <br>
Then cmeans (fuzzy clustering or soft clustering so that it can be classified in to more than one class) is used to get the membership for clusters<br>


