

# Load Data



import pickle as pickle
import os
import pandas as pd
import torch

class news_dataset(torch.utils.data.Dataset):
    """Dataset 구성을 위한 class."""

    def __init__(self, news_dataset, labels):
        self.news_dataset = news_dataset
        self.labels = labels


    def __getitem__(self, idx):
        item = {
            key: val[idx].clone().detach() for key, val in self.news_dataset.items()
        }
      
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
        
        
def load_data(dataset_dir):
    """csv 파일을 경로에 맡게 불러 옵니다."""
    pd_dataset = pd.read_csv(dataset_dir)
    return pd_dataset


def tokenized_dataset(dataset, tokenizer, max_length):
    """tokenizer에 따라 sentence를 tokenizing 합니다."""
    concat_entity = []
    for e01 in dataset["title"]:
        concat_entity.append(e01)

    tokenized_sentences = tokenizer(
        concat_entity,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
        return_token_type_ids=False,
    )
    return tokenized_sentences



    
# Train


import os
import random
#import argparse

import numpy as np
from sklearn.metrics import accuracy_score

import torch
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
)



from transformers import Trainer, TrainingArguments
from transformers import EarlyStoppingCallback
from transformers.optimization import get_cosine_with_hard_restarts_schedule_with_warmup




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

def train():
    #parser = argparse.ArgumentParser
    #args = parser.parse_args()

    # fix a seed
    seed_everything(12)
    # load model and tokenizer
    MODEL_NAME = 'klue/roberta-large'
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # load dataset
    train_dataset = load_data("/home/sgnamu3021/Data/train_final.csv")
    test_dataset = load_data("/home/sgnamu3021/Data/test_final.csv")
    
    train_label = train_dataset["topic_idx"].values.tolist()
    test_label = test_dataset["topic_idx"].values.tolist()
    
    # topic_idx 값을 1씩 빼기 (pandas.DataFrame)
#     train_dataset['topic_idx'] = train_dataset['topic_idx']-1
#     test_dataset['topic_idx'] = test_dataset['topic_idx']-1

    # topic_idx 값을 1씩 빼기 (list)    
#     train_label = [i-1 for i in train_dataset["topic_idx"].values.tolist()]
#     test_label = [i-1 for i in test_dataset["topic_idx"].values.tolist()]


    for i in range(len(train_label)):
        train_label[i] = train_label[i] - 1

    for i in range(len(test_label)):
        test_label[i] = test_label[i]-1
        

    # tokenizing dataset
    tokenized_train = tokenized_dataset(train_dataset, tokenizer, 64)
    tokenized_test = tokenized_dataset(test_dataset, tokenizer, 64)

    # make dataset for pytorch.
    news_train_dataset = news_dataset(tokenized_train, train_label)
    news_test_dataset = news_dataset(tokenized_test, test_label)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    print(device, end='\n')



    # setting model hyperparameter
    model_config = AutoConfig.from_pretrained(MODEL_NAME)
    model_config.num_labels = 5

    model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, config=model_config
    )

    print(model.config, end='\n')
    model.parameters
    #model.to(device)

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


import easydict
 
args = easydict.EasyDict({
    
        "model_type": "roberta",
    
        "model_name": "klue/roberta-large",
    
        "save_path": "/home/sgnamu3021/workspace/",
    
        "save_step": 500,
    
        "save_limit": 5,
    
        "seed": 12,
 
        "batch_size": 16,
 
        "epochs": 3,
    
        "max_len": 64,
 
        "lr": 5e-5,
 
        "weight_decay": 0.01,
 
        "warmup_steps": 300,
 
        "scheduler": "inear"
 
})

    
seed_everything(args.seed)
main()



# Inference


from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
#from load_data import *
import pandas as pd
import torch
import torch.nn.functional as Fa

import numpy as np
#import argparse
from tqdm import tqdm

# from train import label_to_num


def inference(model, tokenized_sent, device):
    """
    test dataset을 DataLoader로 만들어 준 후,
    batch_size로 나눠 model이 예측 합니다.
    """
    dataloader = DataLoader(tokenized_sent, batch_size=16, shuffle=False)
    model.eval()
    output_pred = []
    for i, data in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            outputs = model(
                input_ids=data["input_ids"].to(device),
                attention_mask=data["attention_mask"].to(device),
            )
        # print(outputs[0])
        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        result = np.argmax(logits, axis=-1)

        output_pred.append(result)
    return (np.concatenate(output_pred).tolist(),)
    
    

def load_test_dataset(dataset_dir, tokenizer):
    """
    test dataset을 불러온 후,
    tokenizing 합니다.
    """
    test_dataset = load_data(dataset_dir)
    # test_label = list(map(int, label_to_num(test_dataset["category"].values)))

    # tokenizing dataset
    tokenized_test = tokenized_dataset(test_dataset, tokenizer, 384)
    return tokenized_test
    

def main(args):
    """
    주어진 dataset csv 파일과 같은 형태일 경우 inference 가능한 코드입니다.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # load tokenizer
    Tokenizer_NAME = args.model
    tokenizer = AutoTokenizer.from_pretrained(Tokenizer_NAME)

    ## load my model
    MODEL_NAME = args.model_dir  # model dir.

    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    # model.parameters
    model.to(device)

    ## load test datset
    test_dataset_dir = "/home/sgnamu3021/Data/test_final.csv"
    test_dataset, test_label = load_test_dataset(test_dataset_dir, tokenizer)
    news_test_dataset = news_dataset(test_dataset, test_label)

    ## predict answer
    pred_answer = inference(model, news_test_dataset, device)  # model에서 class 추론
    #pred_answer = num_to_label(pred_answer)
    test_dataset = load_data(test_dataset_dir)

    ## make csv file with predicted answer
    #########################################################
    output = pd.DataFrame(
        {
            "title": test_dataset["title"],
            #"topic_idx": test_dataset["topic_idx"],
            "topic_idx": list(test_dataset["topic_idx"].values),
            "result": pred_answer,
        }
    )

    output.to_csv(
        "/home/sgnamu3021/Data/submission.csv", index=False
    )  # 최종적으로 완성된 예측한 라벨 csv 파일 형태로 저장.
    print("---- Finish! ----")





