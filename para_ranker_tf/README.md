This model uses RNN-LSTM with simple attention mechanism. It take lot of time to train compare to the bert model. 

Steps to run:

1. Create training file train.ids.question, train.ids.para, train.ids.label, valid.ids.question, valid.ids.para, valid.ids.label. *.ids.para and *.ids.question contains the token indices for each token in the question. Also you'll need to create a vocab list before that to map the tokens and the index. Also this uses trimmed glove embeddings which has word vectors for the woords which occurs in the train/valid set only. 

2. Edit the hyperparameters in main.py.

3. Run python main.py from same folder.