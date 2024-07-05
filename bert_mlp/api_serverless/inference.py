import json

import torch
import utils
from transformers import BertModel, BertTokenizerFast

IDX_TO_STRINGS = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CPU_DEVICE = torch.device("cpu")


def get_model_and_tokenizer():
    """Return BertMlp and BertTokenizerFast instances."""
    bert = BertModel.from_pretrained(
        "google-bert/bert-base-uncased", cache_dir="."
    )
    # mlp takes cls token of dim 768 and maps to 4 labels;
    mlp = utils.MLP(768, 4)
    # load mlp state dict from checkpoint;
    # checkpoint expected to be in same as inference.py
    # so take care when copying in the docker image;
    mlp.load_state_dict(
        torch.load("mlp_agnews.ckpt", map_location=DEVICE)["mlp_state_dict"]
    )
    # combine bert and mlp;
    model = utils.BertMlp(bert=bert, mlp=mlp)
    model = model.to(DEVICE)
    model.requires_grad_(False)  # stop grads;
    # get tokenizer;
    tokenizer = BertTokenizerFast.from_pretrained(
        "google-bert/bert-base-uncased", cache_dir="."
    )
    return model, tokenizer


def predict(strings: list[str]) -> list[str]:
    model, tokenizer = get_model_and_tokenizer()
    x = tokenizer(
        strings,
        padding="longest",
        return_tensors="pt",
        truncation="longest_first",
        max_length=200,
    )
    x = x.to(DEVICE)
    # get predicted ids and move to cpu;
    ids = model(**x).argmax(-1)
    if DEVICE != CPU_DEVICE:
        ids = ids.to(CPU_DEVICE)
    # decode to string;
    return [IDX_TO_STRINGS[i.item()] for i in ids]


def handler(event, context):
    """This is the handler for the AWS Lambda"""
    try:
        body = json.loads(event["body"])
        context = body["context"]
        if isinstance(context, str):
            context = [context]
        labels = predict(context)
        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
            },
            "body": json.dumps({"article_types": labels}),
        }
    except Exception as e:
        print(repr(e))
        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json",
            },
            "body": json.dumps({"error": repr(e)}),
        }
