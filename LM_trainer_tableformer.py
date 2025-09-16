import torch
import pandas as pd
from transformers import TapasConfig, TapasForQuestionAnswering, TapasTokenizer, TapasForMaskedLM
import sys
import ast
from transformers import AdamW
import numpy as np
from table_relative_ids import get_relative_attention_ids
import argparse
import collections
import os
import warnings
import json
from utils import *

warnings.simplefilter(action='ignore', category=FutureWarning)


# config = TapasConfig.from_pretrained('model/config.json')
# model = TapasForMaskedLM.from_pretrained('model/pytorch_model.bin', config=config)
# tokenizer = TapasTokenizer.from_pretrained("model/", do_lower_case=False)

# optimizer = AdamW(model.parameters(), lr=5e-5)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)


def text_masking(tokens):
    rand = torch.rand(tokens.input_ids.shape)
    masked_arr = (rand < 0.15) * (tokens.input_ids != 101) * (tokens.input_ids != 102) * (tokens.input_ids != 0)
    selection = torch.flatten((masked_arr[0].nonzero())).tolist()
    tokens.input_ids[0, selection] = 103
    return tokens.input_ids


class TableDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer):
        self.df = df['data']
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        query = self.df[idx]['Description']
        table_text = self.df[idx]['TBL']
        table = pd.DataFrame(table_text[1:], columns=table_text[0])

        encoding = self.tokenizer(table=table,
                                  queries=query,
                                  padding="max_length",
                                  truncation=True,
                                  return_tensors="pt"
                                  )

        # remove the batch dimension which the tokenizer adds

        # print(encoding['labels'], encoding['input_ids'])
        encoding['labels'] = encoding['input_ids'].clone()
        encoding['input_ids'] = text_masking(encoding)
        encoding = {key: val.squeeze(0) for key, val in encoding.items()}
        encoding['rel_ids'] = get_relative_attention_ids(encoding)

        return encoding

    def __len__(self):
        return len(self.df)


def _parse_answer_coordinates(answer_coordinate_str):
    """Parses the answer_coordinates of a question.
    Args:
        answer_coordinate_str: A string representation of a Python list of tuple
        strings.
        For example: "['(1, 4)','(1, 3)', ...]"
    """

    try:
        answer_coordinates = []
        # make a list of strings
        coords = ast.literal_eval(answer_coordinate_str)

        # parse each string as a tuple
        for row_index, column_index in sorted(ast.literal_eval(coord) for coord in coords):
            answer_coordinates.append((row_index, column_index))
    except SyntaxError:
        raise ValueError('Unable to evaluate %s' % answer_coordinate_str)

    return answer_coordinates


def _parse_answer_text(answer_text):
    """Populates the answer_texts field of `answer` by parsing `answer_text`.
    Args:
        answer_text: A string representation of a Python list of strings.
        For example: "[u'test', u'hello', ...]"
        answer: an Answer object.
    """
    try:
        answer = []
        for value in ast.literal_eval(answer_text):
            answer.append(value)
    except SyntaxError:
        raise ValueError('Unable to evaluate %s' % answer_text)

    return answer


def get_sequence_id(example_id, annotator):
    if "-" in str(annotator):
        raise ValueError('"-" not allowed in annotator.')
    return f"{example_id}-{annotator}"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    # parser.add_argument("--tsv_path", type=str, default="orig_data/train.tsv")
    # parser.add_argument("--table_csv_path", type=str, default="orig_data/")

    parser.add_argument("--tsv_path", type=str, default="korquad_table_data/new_train_file.tsv")
    parser.add_argument("--table_csv_path", type=str, default="korquad_table_data/")

    parser.add_argument("--dev_path", type=str, default="korquad_table_data/dev_file.tsv")
    parser.add_argument("--dev_result_path", type=str, default="dev_tapas_kor_result/")

    parser.add_argument("--test_path", type=str, default="korquad_table_data/test_file.tsv")
    parser.add_argument("--test_result_path", type=str, default="test_tapas_kor_result/")

    parser.add_argument("--model_name", type=str, default="tapas")

    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--checkpoints", type=str, default="checkpoints/")
    arg = parser.parse_args()
    tsv_path = arg.tsv_path

    table_csv_path = arg.table_csv_path
    batch_size = arg.batch_size
    # epoch = arg.epoch
    dev_path = arg.dev_path
    dev_result_path = arg.dev_result_path

    test_path = arg.test_path
    test_result_path = arg.test_result_path
    checkpoints = arg.checkpoints

    model_name = arg.model_name

    config = TapasConfig.from_pretrained('model/config.json')
    model = TapasForMaskedLM.from_pretrained('model/pytorch_model.bin', config=config)
    tokenizer = TapasTokenizer.from_pretrained("model/", do_lower_case=False)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    model.to(device)

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    data = {}
    with open('MLM_data/lm_data.json') as f:
        json_file = json.load(f)

    train_dataset = TableDataset(df=json_file, tokenizer=tokenizer)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)

    dev_accuracy = 0.0
    max_accuracy = 0.0
    #
    for epoch in range(arg.epoch):
        # Training step
        print('now epoch : {}'.format(epoch + 1))
        model.train()
        for idx, batch in enumerate(train_dataloader):
            # forward + backward + optimize
            relative_attention_ids = batch["rel_ids"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                                labels=labels, relative_attention_ids=relative_attention_ids)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            if idx % 500 == 0:
                print("Loss:", loss.item())
                torch.save(model, "lm_tableformer_pretrained/lm_{}.pt".format(idx))




        torch.save(model, "lm_tableformer_pretrained/lm_{}.pt".format(epoch + 1))