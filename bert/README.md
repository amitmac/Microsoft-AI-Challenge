This model uses bert model to generate embedding for each token in sentence and fine tune the network using another transformer layer and simple attention mechanism for both paragraph and question at the end. I also used simple feed forward at the end but found the simple attention mechanism better.

This model give test score of ~0.69. Using ensembles of the different models with little modified architectures, I was able to get 0.70.

**NOTE**: Most of the code is taken from Google BERT repository.

Steps to run the code.

1. Download the bert pre-trained embeddings in pretrained_models directory parallel to the main.py file and extract the file directly in that folder.

2. Create the directory models/ for storing the updated models.

3. Create directory data/ for keeping all the training and evaluation data. Each file has format -
training - qid     question    paragraph   label sequence_number
testing doesn't contain label

3. Run python main.py from the same folder. It will automatically create a tf_record files by calling functions from data_ops.

**NOTE**: Hyperparameters can be edited in main.py file. Also I have kept main.py and main_test.py different as to allow the model train in parallel and test while it is training. But you won't be able to use GPU in parallel and it would raise an error. This can be tackled by setting os.environ["CUDA_VISIBLE_DEVICES"]="-1". Model is very complex so testing might take more than 4-5 hours on CPU. So whenever you want to test, you can stop the training file and run the main_test.py which will take only 10-15 mins on GPU.