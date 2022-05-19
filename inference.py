from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from load_data import *
import pandas as pd
import torch
import torch.nn.functional as F

import numpy as np
import argparse
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


def num_to_label(label):
    """
    숫자로 되어 있던 class를 원본 문자열 라벨로 변환 합니다.
    """
    origin_label = []
    dict_num_to_label = {
        0: "international",
        1: "economy",
        2: "society",
        3: "sport",
        4: "it",
        5: "politics",
        6: "entertain",
        7: "culture",
    }

    for v in label[0]:
        origin_label.append(dict_num_to_label[v])

    return origin_label


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
    test_dataset_dir = "..data/newszum_test_data.csv"
    test_dataset, test_label = load_test_dataset(test_dataset_dir, tokenizer)
    news_test_dataset = news_dataset(test_dataset, test_label)

    ## predict answer
    pred_answer = inference(model, news_test_dataset, device)  # model에서 class 추론
    pred_answer = num_to_label(pred_answer)
    test_dataset = load_data(test_dataset_dir)

    ## make csv file with predicted answer
    #########################################################
    output = pd.DataFrame(
        {
            "title": test_dataset["title"],
            "cleanBody": test_dataset["cleanBody"],
            "category": list(test_dataset["category"].values),
            "result": pred_answer,
        }
    )

    output.to_csv(
        "./prediction/submission.csv", index=False
    )  # 최종적으로 완성된 예측한 라벨 csv 파일 형태로 저장.
    print("---- Finish! ----")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # model dir
    parser.add_argument("--model_dir", type=str, default="./best_model")
    parser.add_argument("--model", type=str, default="klue/roberta-large")
    args = parser.parse_args()
    print(args)
    main(args)