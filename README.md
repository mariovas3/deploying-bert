# Repo for deploying BERT-based models;

## Saved models:

> All models are based on HuggingFace's `BertModel.from_pretrained('google-bert/bert-base-uncased')` and `BertTokenizerFast.from_pretrained('google-bert/bert-base-uncased')`.

I have saved model checkpoints (model and optimiser state dict) in the `./saved_models` directory (not committed to git). The saved models currently are:

* `mlp_agnews.ckpt` - an MLP making predictions based on the cls embeddings of huggingface's `BertModel.from_pretrained('google-bert/bert-base-uncased')`. 
    * In this case I didn't train any params of BERT, so can directly download it from HF. 
    * There is a <a href="./bert_mlp/BERT_MLP_AGNEWS.ipynb">notebook</a> showing how I trained and saved the MLP.

## Building the image and testing:
* The base image used is from <a href="https://hub.docker.com/r/amazon/aws-lambda-python">aws-lambda-python</a>.
* To build the cpu image, navigate to root of repo and run:

    ```bash
    docker build -t bert_mlp/aws_lambda_opit -f bert_mlp/api_serverless/Dockerfile .
    ```

* Then to run a container and test it, do:

    ```bash
    docker run -p 9000:8080 bert_mlp/aws_lambda_opit
    ```

* Then open a different terminal and run:

    ```bash
    curl --request POST --url http://localhost:9000/2015-03-31/functions/function/invocations --header 'Content-Type: application/json' --data '{"body": "{\"context\": \"Cristiano Ronaldo scored a goal against Argentina!\"}"}'
    ```

    which should return a response:

    ```
    {"statusCode": 200, "headers": {"Content-Type": "application/json"}, "body": "{\"article_types\": [\"Sports\"]}"}
    ```

    i.e., "Cristiano Ronaldo scored a goal against Argentina!" is predicted to belong to the "Sports" label.

    The first request may take longer to download the hf BERT model, 
    and the following requests should be significantly faster. Testing locally (8 gen intel i5-U cpu), the first request took `58812.12 ms` while the others took about `600 ms` (~100 times quicker than the first request).