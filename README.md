# Repo for deploying BERT-based models;

## Saved models:

I have saved model checkpoints (model and optimiser state dict) in the `./saved_models` directory. The saved models currently are:

* `mlp_agnews.ckpt` - an MLP making predictions based on the cls embeddings of huggingface's `BertModel.from_pretrained('google-bert/bert-base-uncased')`. 
    * In this case I didn't train any params of BERT, so can directly download it from HF. 
    * There is a <a href="./bert-mlp/BERT_MLP_AGNEWS.ipynb">notebook</a> showing how I trained and saved the MLP.