import os
import random
import argparse

import numpy as np
from sklearn.metrics import accuracy_score

import torch
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
)

from load_data import *

from transformers import Trainer, TrainingArguments
from transformers import EarlyStoppingCallback
from transformers.optimization import get_cosine_with_hard_restarts_schedule_with_warmup

# import wandb

# ------* Fix Seeds * -----------#
def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def compute_metrics(pred):
    """validation을 위한 metrics function"""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    # calculate accuracy using sklearn's function
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
    }


def label_to_num(label):
    dict_label_to_num = {
        "international": 0,
        "economy": 1,
        "society": 2,
        "sport": 3,
        "it": 4,
        "politics": 5,
        "entertain": 6,
        "culture": 7,
    }
    num_label = []

    for v in label:
        num_label.append(dict_label_to_num[v])

    return num_label


def train():
    parser = argparse.ArgumentParser
    # args = parser.parse_args()

    # fix a seed
    seed_everything(12)
    # load model and tokenizer
    MODEL_NAME = 'klue/roberta-large'
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # load dataset
    train_dataset = load_data("./Data/train.csv")
    test_dataset = load_data("./Data/test.csv")

    train_label = train_dataset["topic_idx"].values.tolist()
    test_label = test_dataset["topic_idx"].values.tolist()
    # print(train_label)

    # tokenizing dataset
    tokenized_train = tokenized_dataset(train_dataset, tokenizer, 20)
    tokenized_test = tokenized_dataset(test_dataset, tokenizer, 20)

    # make dataset for pytorch.
    news_train_dataset = news_dataset(tokenized_train, train_label)
    news_test_dataset = news_dataset(tokenized_test, test_label)
    print("@@@@@@@@@@@@@@@@@@@@@@@")
    print(news_train_dataset)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(device)

    # setting model hyperparameter
    model_config = AutoConfig.from_pretrained(MODEL_NAME)
    model_config.num_labels = 8

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, config=model_config
    )

    # print(model.config)
    model.parameters
    model.to(device)

    training_args = TrainingArguments(
        output_dir=args.save_path + "results",  # output directory
        save_total_limit=args.save_limit,  # number of total save model.
        save_steps=args.save_step,  # model saving step.
        num_train_epochs=args.epochs,  # total number of training epochs
        learning_rate=args.lr,  # learning_rate
        per_device_train_batch_size=args.batch_size,  # batch size per device during training
        per_device_eval_batch_size=8,  # batch size for evaluation
        warmup_steps=args.warmup_steps,  # number of warmup steps for learning rate scheduler
        weight_decay=args.weight_decay,  # strength of weight decay
        logging_dir=args.save_path + "logs",  # directory for storing logs
        logging_steps=100,  # log saving step.
        evaluation_strategy="steps",  # evaluation strategy to adopt during training
        # `no`: No evaluation during training.
        # `steps`: Evaluate every `eval_steps`.
        # `epoch`: Evaluate every end of epoch.
        eval_steps=500,  # evaluation step.
        load_best_model_at_end=True,
    )

    ### callback & optimizer & scheduler 추가
    MyCallback = EarlyStoppingCallback(
        early_stopping_patience=3, early_stopping_threshold=0.001
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=args.weight_decay,
        amsgrad=False,
    )

    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=news_train_dataset,  # training dataset
        eval_dataset=news_test_dataset,  # evaluation dataset
        compute_metrics=compute_metrics,  # define metrics function
        callbacks=[MyCallback],
        optimizers=(
            optimizer,
            get_cosine_with_hard_restarts_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_steps,
                num_training_steps=len(news_train_dataset) * args.epochs,
            ),
        ),
    )

    # train model
    trainer.train()
    model.save_pretrained("./best_model")


def main():
    train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type", default="roberta", type=str, help="model type(default=roberta)"
    )
    parser.add_argument(
        "--model_name",
        default="klue/roberta-large",
        type=str,
        help='model name(default="Huffon/klue-roberta-base-nli")',
    )
    parser.add_argument(
        "--save_path", default="C:/Users/HP/Desktop/workspace/", type=str, help="saved path(default=./)"
    )
    parser.add_argument(
        "--save_step", default=500, type=int, help="model saving step(default=500)"
    )
    parser.add_argument(
        "--save_limit", default=5, type=int, help="# of save model(default=5)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed (default: 42)"
    )
    parser.add_argument(
        "--epochs", type=int, default=3, help="number of epochs to train (default: 20)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="batch size per device during training (default: 16)",
    )
    parser.add_argument(
        "--max_len", type=int, default=384, help="max length (default: 256)"
    )
    parser.add_argument(
        "--lr", type=float, default=5e-5, help="learning rate (default: 5e-5)"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="strength of weight decay(default: 0.01)",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=300,
        help="number of warmup steps for learning rate scheduler(default: 500)",
    )
    parser.add_argument(
        "--scheduler", type=str, default="linear", help='scheduler(default: "linear")'
    )
    args = parser.parse_args()
    print(args)

    # fix a seed
    
    seed_everything(args.seed)
    main()
    