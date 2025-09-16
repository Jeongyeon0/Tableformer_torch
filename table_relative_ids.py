import torch
import sys


def get_relative_attention_ids(features):

    relative_attention_ids=[]
    input_mask = features['input_ids']
    type_ids = features['token_type_ids'].transpose(1,0)
    segment_ids = type_ids[0].unsqueeze(dim=0)
    column_ids = type_ids[1].unsqueeze(dim=0)
    row_ids = type_ids[2].unsqueeze(dim=0)
    '''
    row_ids = features['row_ids']
    column_ids = features['columns_ids']
    segment_ids = features['segment_ids']
    input_mask=features['input_mask']
    '''
    cell_mask = torch.logical_not(torch.eq(row_ids, 0)) & torch.eq(segment_ids, 1) & torch.eq(input_mask, 1)
    header_mask = torch.eq(row_ids, 0) & torch.eq(segment_ids, 1) & torch.eq(input_mask, 1)
    sent_mask = torch.eq(segment_ids, 0) & torch.eq(input_mask, 1)
    disabled_attention_bias = 0
    # "Same row" attention bias with type id = 1.
    relative_attention_ids.append((torch.eq(torch.unsqueeze(row_ids,1),torch.unsqueeze(row_ids,2)) & torch.unsqueeze(cell_mask, 1) & torch.unsqueeze(cell_mask, 2)).type(torch.int32) * 1)

    # "Same column" attention bias matrix with type id = 2.
    relative_attention_ids.append((torch.eq(torch.unsqueeze(column_ids, 1), torch.unsqueeze(column_ids, 2)) & torch.unsqueeze(cell_mask, 1) & torch.unsqueeze(cell_mask, 2)).type(torch.int32) * 2)

    # "Same cell" attention bias matrix with type id = 3.
    if disabled_attention_bias == 3:
        relative_attention_ids.append((torch.eq(torch.unsqueeze(column_ids, 1), torch.unsqueeze(column_ids, 2)) & torch.eq(torch.unsqueeze(row_ids, 1), torch.unsqueeze(row_ids, 2)) & torch.unsqueeze(cell_mask, 1) & torch.unsqueeze(cell_mask, 2)).type(torch.int32) * -3)

    # "Cell to its header" bias matrix with type id = 4.
    relative_attention_ids.append((torch.eq(torch.unsqueeze(column_ids, 1), torch.unsqueeze(column_ids, 2)) & torch.unsqueeze(header_mask, 1) & torch.unsqueeze(cell_mask, 2)).type(torch.int32) * 4)

    # "Cell to sentence" bias matrix with type id = 5.
    relative_attention_ids.append((torch.unsqueeze(sent_mask, 1) & torch.unsqueeze(cell_mask, 2)).type(torch.int32) * 5)

    # "Header to column cell" bias matrix with type id = 6.
    relative_attention_ids.append((torch.eq(torch.unsqueeze(column_ids, 1), torch.unsqueeze(column_ids, 2)) & torch.unsqueeze(cell_mask, 1) & torch.unsqueeze(header_mask, 2)).type(torch.int32) * 6)

    # "Header to other header" bias matrix with type id = 7.
    relative_attention_ids.append((torch.unsqueeze(header_mask, 1) & torch.unsqueeze(header_mask, 2)).type(torch.int32) * 7)
    # "Header to same header" bias matrix with type id = 8.
    relative_attention_ids.append((torch.eq(torch.unsqueeze(column_ids, 1), torch.unsqueeze(column_ids, 2)) & torch.eq(torch.unsqueeze(row_ids, 1), torch.unsqueeze(row_ids, 2)) & torch.unsqueeze(header_mask, 1) & torch.unsqueeze(header_mask, 2)).type(torch.int32) * 1)

    # "Header to sentence" bias matrix with type id = 9.
    relative_attention_ids.append((torch.unsqueeze(sent_mask, 1) & torch.unsqueeze(header_mask, 2)).type(torch.int32) * 9)

    # "Sentence to cell" bias matrix with type id = 10.
    relative_attention_ids.append((torch.unsqueeze(cell_mask, 1) & torch.unsqueeze(sent_mask, 2)).type(torch.int32) * 10)

    # "Sentence to header" bias matrix with type id = 11.
    relative_attention_ids.append((torch.unsqueeze(header_mask, 1) & torch.unsqueeze(sent_mask, 2)).type(torch.int32) * 11)

    # "Sentence to sentence" bias matrix with type id = 12.
    relative_attention_ids.append((torch.unsqueeze(sent_mask, 1) & torch.unsqueeze(sent_mask, 2)).type(torch.int32) * 12)

    return torch.sum(torch.stack(relative_attention_ids,0),0)
