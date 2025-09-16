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
warnings.simplefilter(action='ignore', category=FutureWarning)

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
        #trigger = -1
        '''
        if item.position != 0:
        #if trigger == 0:
            # use the previous table-question pair to correctly set the prev_labels token type ids
            previous_item = self.df.iloc[idx-1]
            encoding = self.tokenizer(table=table,
                                      queries=[previous_item.question, item.question],
                                      answer_coordinates=[previous_item.answer_coordinates, item.answer_coordinates],
                                      answer_text=[previous_item.answer_text, item.answer_text],
                                      padding="max_length",
                                      truncation=True,
                                      return_tensors="pt"
                                      )
            # use encodings of second table-question pair in the batch
            encoding = {key: val[-1] for key, val in encoding.items()}
        
        else:
        '''
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
        #print(item.table_file)
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
        if model_name=='tableformer':
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
            coords_to_answer[key] = np.array(coords_to_probs[key]).mean() > 0.1
            if coords_to_answer[key]==True:
                coord_tmp.append("'"+str(new_key)+"'")
                text_tmp.append("'"+table.iat[int(column), int(row)]+"'")
        #prev_answers[idx + 1] = coords_to_answer
        predicted_coords.append([(", ").join(coord_tmp)])
        predicted_answer.append((", ").join(text_tmp))
    logits_batch = torch.cat(tuple(all_logits), 0)

    return logits_batch, predicted_coords, predicted_answer

def inference_phase(table, queries, tokenizer, model, device, model_name):

    inputs = tokenizer(table=table, queries=queries, padding='max_length', truncation=True, return_tensors='pt')
    logits, predicted_coords, predicted_answer = compute_prediction_sequence(model, inputs, device, model_name,table)
    #print(table)
    '''
    predicted_answer_coordinates, = tokenizer.convert_logits_to_predictions(inputs, logits.cpu().detach())
    answers = []
    ans_coord = []
    for coordinates in predicted_answer_coordinates:
        coord_list = []
        if len(coordinates) == 1:
            # only a single cell:
            answers.append("'"+table.iat[coordinates[0]]+"'")
            coord_list.append("'"+str(coordinates[0])+"'")
        else:
            # multiple cells
            cell_values = []
            tmp = []
            for coordinate in coordinates:
                tmp.append("'"+str(coordinate)+"'")
                cell_values.append("'"+table.iat[coordinate]+"'")
            coord_list.append(", ".join(tmp))
            answers.append(", ".join(cell_values))
        ans_coord.append(coord_list)
    print(ans_coord)
    print(answers)
    
    return answers, ans_coord
    '''
    return predicted_answer, predicted_coords


def file_inference(dev_path, dev_result_path, epoch, tokenizer, model, device, model_name):
    model.eval()
    devset = open(dev_path, 'r').readlines()
    queries = []
    pred_list, id_list, annotator_list, position_list, table_file_list, coord_list = [], [], [], [], [], []
    lineno = 0
    dev_result = open(dev_result_path + '{}.tsv'.format(epoch + 1), 'w')
    table=None
    for i, line in enumerate(devset):
        lineno += 1
        line = line.strip()
        line = line.split('\t')
        if i == 0:
            for k in range(len(line) - 1):
                dev_result.write(line[k] + '\t')
            dev_result.write(line[-1] + '\n')
            continue

        position = int(line[2])
        question = line[3]
        if position == 0 and i < 2:
            id_list.append(line[0])
            annotator_list.append(line[1])
            position_list.append(line[2])
            table_file_list.append(line[4])

            queries.append(question)
            table = pd.read_csv(table_csv_path + line[4]).astype(str)

        elif i == len(devset) - 1:
            id_list.append(line[0])
            annotator_list.append(line[1])
            position_list.append(line[2])
            table_file_list.append(line[4])
            queries.append(question)
            pred_answers, pred_coord = inference_phase(table, queries, tokenizer, model, device, model_name)

            coord_list.extend(pred_coord)
            pred_list.extend(pred_answers)

        elif position == 0 and i >= 2:
            pred_answers, pred_coord = inference_phase(table, queries, tokenizer, model, device, model_name)
            pred_list.extend(pred_answers)
            coord_list.extend(pred_coord)
            for id, anno, posit, query, t_name, coor, predicted_answer in zip(id_list, annotator_list, position_list,
                                                                              queries, table_file_list, coord_list,
                                                                              pred_list):
                dev_result.write(
                    id + '\t' + anno + '\t' + str(posit) + '\t' + query + '\t' + t_name + '\t' + str(coor).replace('"','') + '\t' + '[' + predicted_answer + ']' + '\n')

            pred_list, id_list, annotator_list, position_list, table_file_list, coord_list = [], [], [], [], [], []
            queries = []

            id_list.append(line[0])
            annotator_list.append(line[1])
            position_list.append(line[2])
            table_file_list.append(line[4])
            queries.append(question)
            table = pd.read_csv(table_csv_path + line[4]).astype(str)
        else:
            id_list.append(line[0])
            annotator_list.append(line[1])
            position_list.append(line[2])
            table_file_list.append(line[4])

            queries.append(question)
    for id, anno, posit, query, t_name, coor, predicted_answer in zip(id_list, annotator_list,
                                                                      position_list, queries,
                                                                      table_file_list, coord_list,
                                                                      pred_list):
        dev_result.write(id + '\t' + anno + '\t' + str(
            posit) + '\t' + query + '\t' + t_name + '\t' + str(coor).replace('"','') + '\t' + '[' + predicted_answer + ']' + '\n')
    fnGold = dev_path
    fnPred = dev_result_path + '{}.tsv'.format(epoch + 1)
    dev_result.close()

    seqCor, seqCnt, ansCor, ansCnt = evaluate(fnGold, fnPred)
    dev_result = open(dev_result_path + '{}.tsv'.format(epoch + 1), 'a')
    dev_result.write("Sequence Accuracy = %0.8f%% (%d/%d)\n" % (100.0 * seqCor / seqCnt, seqCor, seqCnt))
    dev_result.write("Answer Accuracy =   %0.8f%% (%d/%d)" % (100.0 * ansCor / ansCnt, ansCor, ansCnt))
    return seqCor, seqCnt, ansCor, ansCnt



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    #parser.add_argument("--tsv_path", type=str, default="orig_data/train.tsv")
    #parser.add_argument("--table_csv_path", type=str, default="orig_data/")

    parser.add_argument("--tsv_path", type=str, default="pump_table/pump_train_file.tsv")
    parser.add_argument("--table_csv_path", type=str, default="pump_table/")

    parser.add_argument("--dev_path", type=str, default="pump_table/pump_dev_file.tsv")
    parser.add_argument("--dev_result_path", type=str, default="pump_table/pump_dev_tableformer_result/test0.1/")

    parser.add_argument("--test_path", type=str, default="pump_table/pump_test_file.tsv")
    parser.add_argument("--test_result_path", type=str, default="pump_table/pump_test_tableformer_result/test0.1/")

    parser.add_argument("--model_name", type=str, default="tableformer")

    parser.add_argument("--epoch", type=int, default=30)
    parser.add_argument("--checkpoints", type=str, default="pump_table/tableformer_checkpoints/test0.1/")
    arg = parser.parse_args()
    tsv_path = arg.tsv_path

    table_csv_path = arg.table_csv_path
    batch_size = arg.batch_size
    #epoch = arg.epoch
    dev_path = arg.dev_path
    dev_result_path = arg.dev_result_path

    test_path = arg.test_path
    test_result_path = arg.test_result_path
    checkpoints = arg.checkpoints


    model_name = arg.model_name


    config = TapasConfig.from_pretrained('model/config.json')
    model = TapasForQuestionAnswering.from_pretrained('pretrained/lm.pt', config=config)
    tokenizer = TapasTokenizer.from_pretrained("model/", do_lower_case=False)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)


    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    data = pd.read_csv(tsv_path, sep='\t')

    data['answer_coordinates'] = data['answer_coordinates'].apply(lambda coords_str: _parse_answer_coordinates(coords_str))
    data['answer_text'] = data['answer_text'].apply(lambda txt: _parse_answer_text(txt))
    data['sequence_id'] = data.apply(lambda x: get_sequence_id(x.id, x.annotator), axis=1)

    grouped = data.groupby(by='sequence_id').agg(lambda x: x.tolist())
    grouped = grouped.drop(columns=['id', 'annotator', 'position'])
    grouped['table_file'] = grouped['table_file'].apply(lambda x: x[0])

    train_dataset = TableDataset(df=data, tokenizer=tokenizer)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    dev_dataset = TableDataset(df=data, tokenizer=tokenizer)
    dev_dataloader = torch.utils.data.DataLoader(dev_dataset, batch_size=1)

    dev_accuracy = 0.0
    max_accuracy = 0.0
    #
    for epoch in range(arg.epoch):
        count_table = 2
        model.train()
        # Training step
        model.train()
        correct = 0

        print('now epoch : {}'.format(epoch+1))
        for idx, batch in enumerate(train_dataloader):
            count_table += 1

            
            relative_attention_ids = batch["rel_ids"].to(device)

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad()
            if model_name=='tableformer':
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,labels=labels, relative_attention_ids = relative_attention_ids)
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, token_type_ids=token_type_ids)
            loss = outputs.loss
            if idx%30==0:
                print("Loss:", loss.item())
            loss.backward()
            optimizer.step()
            #print(count_table)

        print('now dev inference...')
        #validation 데이터 평가
        seqCor, seqCnt, ansCor, ansCnt = file_inference(dev_path, dev_result_path, epoch, tokenizer, model, device, model_name)
        dev_accuracy = ansCor / ansCnt
        print('validation result: {}'.format(dev_accuracy))
        if max_accuracy<dev_accuracy:
            max_accuracy = dev_accuracy
            print('now model save...')
            torch.save(model, checkpoints+"best_{}.pt".format(epoch + 1))

            #test 데이터 평가
            seqCor, seqCnt, ansCor, ansCnt = file_inference(test_path, test_result_path, epoch, tokenizer, model, device, model_name)
            test_accuracy = ansCor / ansCnt
            print('test result: {}'.format(test_accuracy))

