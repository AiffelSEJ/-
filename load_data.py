import pickle as pickle
import os
import pandas as pd
import torch

class news_dataset(torch.utils.data.Dataset):
    """Dataset 구성을 위한 class."""

    def __init__(self, news_dataset, labels):
        self.news_dataset = news_dataset
        self.labels = labels


    def __getitem__(self, idx=177):
        # print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        
        # print(type(self.news_dataset))
        # print(idx)
        # for key, val in self.news_dataset.items():
            # print(val[178])
        # print(len(self.news_dataset))
        # print(self.news_dataset.keys())
        item = {
            key: val[idx].clone().detach() for key, val in self.news_dataset.items()
        }
        
      
        item["topic_idx"] = torch.tensor(self.labels[idx])
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