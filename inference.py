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
from utils import *
from tabulate import tabulate

warnings.simplefilter(action='ignore')
#warnings.filterwarnings(action='ignore',category=UserWarning)
np.seterr(all='ignore')

# config = TapasConfig.from_pretrained('model/config.json')
# model = TapasForMaskedLM.from_pretrained('model/pytorch_model.bin', config=config)
# tokenizer = TapasTokenizer.from_pretrained("model/", do_lower_case=False)

# optimizer = AdamW(model.parameters(), lr=5e-5)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)


class TableDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer):
        self.df = df
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        table = pd.read_csv(table_csv_path + item.table_file).astype(
            str)  # TapasTokenizer expects the table data to be text only
        # trigger = -1

        # this means it's the first table-question pair in a sequence
        encoding = self.tokenizer(table=table,
                                  queries=item.question,
                                  answer_coordinates=item.answer_coordinates,
                                  answer_text=item.answer_text,
                                  padding="max_length",
                                  truncation=True,
                                  return_tensors="pt"
                                  )

        # remove the batch dimension which the tokenizer adds
        encoding = {key: val.squeeze(0) for key, val in encoding.items()}
        encoding['rel_ids'] = get_relative_attention_ids(encoding)
        # print(item.table_file)
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


def compute_prediction_sequence(model, data, device, model_name, table):
    """Computes predictions using model's answers to the previous questions."""

    # prepare data
    model.eval()
    input_ids = data["input_ids"].to(device)
    attention_mask = data["attention_mask"].to(device)
    token_type_ids = data["token_type_ids"].to(device)

    all_logits = []
    prev_answers = None

    num_batch = data["input_ids"].shape[0]

    coords_to_answer = {}
    predicted_coords = []
    predicted_answer = []

    for idx in range(num_batch):
        prev_answers = None
        if prev_answers is not None:
            coords_to_answer = prev_answers[idx]
            # Next, set the label ids predicted by the model
            prev_label_ids_example = token_type_ids_example[:, 3]  # shape (seq_len,)
            model_label_ids = np.zeros_like(prev_label_ids_example.cpu().numpy())  # shape (seq_len,)

            # for each token in the sequence:
            token_type_ids_example = token_type_ids[idx]  # shape (seq_len, 7)
            for i in range(model_label_ids.shape[0]):
                segment_id = token_type_ids_example[:, 0].tolist()[i]
                col_id = token_type_ids_example[:, 1].tolist()[i] - 1
                row_id = token_type_ids_example[:, 2].tolist()[i] - 1
                if row_id >= 0 and col_id >= 0 and segment_id == 1:
                    model_label_ids[i] = int(coords_to_answer[(col_id, row_id)])

            # set the prev label ids of the example (shape (1, seq_len) )
            token_type_ids_example[:, 3] = torch.from_numpy(model_label_ids).type(torch.long).to(device)

        prev_answers = {}
        # get the example

        data_dict = {}
        input_ids_example = input_ids[idx]  # shape (seq_len,)
        attention_mask_example = attention_mask[idx]  # shape (seq_len,)
        token_type_ids_example = token_type_ids[idx]  # shape (seq_len, 7)
        data_dict['input_ids'] = input_ids_example
        data_dict['token_type_ids'] = token_type_ids_example

        # forward pass to obtain the logits
        if model_name == 'tableformer':
            relative_attention_ids_example = get_relative_attention_ids(data_dict)
            outputs = model(input_ids=input_ids_example.unsqueeze(0),
                            attention_mask=attention_mask_example.unsqueeze(0),
                            token_type_ids=token_type_ids_example.unsqueeze(0),
                            relative_attention_ids=relative_attention_ids_example.unsqueeze(0))
        else:
            outputs = model(input_ids=input_ids_example.unsqueeze(0),
                            attention_mask=attention_mask_example.unsqueeze(0),
                            token_type_ids=token_type_ids_example.unsqueeze(0))
        logits = outputs.logits
        all_logits.append(logits)

        # convert logits to probabilities (which are of shape (1, seq_len))
        dist_per_token = torch.distributions.Bernoulli(logits=logits)
        probabilities = dist_per_token.probs * attention_mask_example.type(torch.float32).to(
            dist_per_token.probs.device)

        # Compute average probability per cell, aggregating over tokens.
        # Dictionary maps coordinates to a list of one or more probabilities
        coords_to_probs = collections.defaultdict(list)
        prev_answers = {}
        for i, p in enumerate(probabilities.squeeze().tolist()):
            segment_id = token_type_ids_example[:, 0].tolist()[i]
            col = token_type_ids_example[:, 1].tolist()[i] - 1
            row = token_type_ids_example[:, 2].tolist()[i] - 1
            if col >= 0 and row >= 0 and segment_id == 1:
                coords_to_probs[(col, row)].append(p)

        # Next, map cell coordinates to 1 or 0 (depending on whether the mean prob of all cell tokens is > 0.5)

        coord_tmp = []
        text_tmp = []
        for key in coords_to_probs:
            row = key[0]
            column = key[1]
            new_key = (column, row)
            coords_to_answer[key] = np.array(coords_to_probs[key]).mean() > 0.5
            if coords_to_answer[key] == True:
                coord_tmp.append("'" + str(new_key) + "'")
                text_tmp.append("'" + table.iat[int(column), int(row)] + "'")
        # prev_answers[idx + 1] = coords_to_answer
        predicted_coords.append([(", ").join(coord_tmp)])
        predicted_answer.append((", ").join(text_tmp))
    logits_batch = torch.cat(tuple(all_logits), 0)

    return logits_batch, predicted_coords, predicted_answer


def inference_phase(table, queries, tokenizer, model, device, model_name):
    inputs = tokenizer(table=table, queries=queries, padding='max_length', truncation=True, return_tensors='pt')
    logits, predicted_coords, predicted_answer = compute_prediction_sequence(model, inputs, device, model_name, table)

    return predicted_answer, predicted_coords


def file_inference(dev_path, dev_result_path, epoch, tokenizer, model, device, model_name):
    model.eval()
    devset = open(dev_path, 'r').readlines()
    queries = []
    pred_list, id_list, annotator_list, position_list, table_file_list, coord_list = [], [], [], [], [], []
    lineno = 0
    dev_result = open(dev_result_path + '{}.tsv'.format(epoch + 1), 'w')
    table = None
    for i, line in enumerate(devset):
        lineno += 1
        line = line.strip()
        line = line.split('\t')
        if i == 0:
            for k in range(len(line) - 1):
                dev_result.write(line[k] + '\t')
            dev_result.write(line[-1] + '\n')
            continue

        pred_list, id_list, annotator_list, position_list, table_file_list, coord_list, queries = [], [], [], [], [], [], []
        position = int(line[2])
        question = line[3]
#        print(f"질문 : {question}")
        table = pd.read_csv(table_csv_path + line[4]).astype(str)

        id_list.append(line[0])
        annotator_list.append(line[1])
        position_list.append(line[2])
        table_file_list.append(line[4])
        queries.append(question)
        pred_answers, pred_coord = inference_phase(table, queries, tokenizer, model, device, model_name)
        table=table.rename(columns={"Unnamed: 0": " "})
        print("질문 : {}".format(queries))
#        print("답변 :{0}, 표 데이터  좌표 :{1}".format(pred_answers, pred_coord)) 
        print("답변 :{0}".format(pred_answers)) 
        print("표 데이터\n {}\n".format(tabulate(table, showindex=False, headers=table.columns)))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--test_path", type=str, default="korquad_table_data/test.tsv")
    parser.add_argument("--test_result_path", type=str, default="test_tapas_kor_result/")

    parser.add_argument("--model_name", type=str, default="tapas")
    parser.add_argument("--checkpoints", type=str, default="checkpoints/")
    parser.add_argument("--table_csv_path", type=str, default="korquad_table_data/")
    arg = parser.parse_args()

    table_csv_path = arg.table_csv_path

    test_path = arg.test_path
    test_result_path = arg.test_result_path
    model_name = arg.model_name

    config = TapasConfig.from_pretrained('model/config.json')
    model = TapasForQuestionAnswering.from_pretrained('checkpoints/new_best.pt', config=config)
    tokenizer = TapasTokenizer.from_pretrained("model/", do_lower_case=False)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(device)


    # test 데이터 평가
    file_inference(test_path, test_result_path, 1, tokenizer, model, device, model_name)


