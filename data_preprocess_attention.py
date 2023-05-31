import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import pickle


class DataPreprocess:
    def __init__(self, df):
        self.table_page_title = df['table_page_title']
        self.table_section_title = df['table_section_title']
        self.table_array = df['table_array']
        self.hc_id = df['highlighted_cell_ids']
        self.question = df['question']
        self.answer = df['answer']
        self.len = len(df)
        self.table_with_col_header = []
        for idx in range(self.len):
            self.table_with_col_header.append(self.get_table_with_col_header(idx))
        self.row = self.get_row()
        self.col = self.get_col()
        self.grid = self.get_grid()

    def get_table_with_col_header(self, idx):
        col_header = self.table_array[idx][0]
        temp_table = self.table_array[idx][1:]
        table_with_col_header = [col_header]
        for row_t in temp_table:
            row_with_col_head = []
            for i in range(len(row_t)):
                grid_with_col_header = col_header[i] + ": " + row_t[i]
                row_with_col_head.append(grid_with_col_header)
            table_with_col_header.append(row_with_col_head)

        return table_with_col_header

    def get_row(self):
        temp_table = self.table_with_col_header
        row_t = []
        for each_table in temp_table:
            row_in_each_table = []
            for i in range(len(each_table) - 1):
                row_in_each_table.append(each_table[i+1])
            row_t.append(row_in_each_table)
        return row_t

    def get_col(self):
        temp_table = self.table_with_col_header
        col_t = []
        for each_table in temp_table:
            col_in_table = []
            for i in range(len(each_table[0])):
                each_col = []
                for j in range(len(each_table)):
                    each_col.append(each_table[j][i])
                col_in_table.append(each_col)
            col_t.append(col_in_table)

        return col_t

    def get_grid(self):
        temp_table = self.table_with_col_header
        grid_t = []
        for each_table in temp_table:
            grid_in_table = []
            for i in range(len(each_table) - 1):
                each_row = []
                for j in range(len(each_table[0])):
                    each_row.append([each_table[i + 1][j]])
                grid_in_table.append(each_row)
            grid_t.append(grid_in_table)

        return grid_t

    def get_data(self):
        return self.row, self.col, self.grid, self.table_page_title, self.table_section_title, self.question, self.answer

    def get_raw_data(self):
        return self.table_page_title, self.table_section_title, self.table_array, self.question, self.answer


# Convert preprocessed table to tokenized inputs for BERT.
class RepresentationGenerationDataset(Dataset):
    def __init__(self, row_d, col_d, grid_d, tpt_d, tst_d, question_d, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.row_d = row_d
        self.col_d = col_d
        self.grid_d = grid_d
        self.table_page_title_d = tpt_d
        self.table_section_title_d = tst_d
        self.question_d = question_d
        self.max_len = max_len

    def __len__(self):
        return len(self.grid_d)

    def __getitem__(self, index):
        row_ids_list = []
        row_mask_list = []
        row_token_type_ids_list = []
        for each_row in self.row_d[index]:
            row_inputs = self.tokenizer(str(each_row), max_length=self.max_len, pad_to_max_length=True)
            row_ids_list.append(row_inputs['input_ids'])
            row_mask_list.append(row_inputs['attention_mask'])
            row_token_type_ids_list.append(row_inputs["token_type_ids"])

        col_ids_list = []
        col_mask_list = []
        col_token_type_ids_list = []
        for each_col in self.col_d[index]:
            col_inputs = self.tokenizer(str(each_col), max_length=self.max_len, pad_to_max_length=True)
            col_ids_list.append(col_inputs['input_ids'])
            col_mask_list.append(col_inputs['attention_mask'])
            col_token_type_ids_list.append(col_inputs["token_type_ids"])

        grid_ids_list = []
        grid_mask_list = []
        grid_token_type_ids_list = []
        for each_row in self.grid_d[index]:
            temp_ids_list = []
            temp_mask_list = []
            temp_token_type_ids_list = []
            for each_grid in each_row:
                grid_inputs = self.tokenizer(str(each_grid), max_length=self.max_len, pad_to_max_length=True)
                temp_ids_list.append(grid_inputs['input_ids'])
                temp_mask_list.append(grid_inputs['attention_mask'])
                temp_token_type_ids_list.append(grid_inputs["token_type_ids"])
            grid_ids_list.append(temp_ids_list)
            grid_mask_list.append(temp_mask_list)
            grid_token_type_ids_list.append(temp_token_type_ids_list)

        table_context_question = self.tokenizer(str(self.table_page_title_d[index])
                                                + str(self.table_section_title_d[index])
                                                + str(self.question_d[index]),
                                                max_length=self.max_len, pad_to_max_length=True)

        return {
            'row_ids_tensor': torch.tensor(row_ids_list),
            'row_mask_tensor': torch.tensor(row_mask_list),
            'row_token_type_ids_tensor': torch.tensor(row_token_type_ids_list)
        },     {
            'col_ids_tensor': torch.tensor(col_ids_list),
            'col_mask_tensor': torch.tensor(col_mask_list),
            'col_token_type_ids_tensor': torch.tensor(col_token_type_ids_list)
        },     {
            'grid_ids_tensor': torch.tensor(grid_ids_list),
            'grid_mask_tensor': torch.tensor(grid_mask_list),
            'grid_token_type_ids_tensor': torch.tensor(grid_token_type_ids_list)
        },     {
            'tcq_ids_tensor': torch.tensor(table_context_question['input_ids']),
            'tcq_mask_tensor': torch.tensor(table_context_question['attention_mask']),
            'tcq_token_type_ids_tensor': torch.tensor(table_context_question['token_type_ids'])
        }


class CustomDataset(Dataset):
    def __init__(self, row_c, col_c, grid_c, tpt_c, tst_c, question_c, answer_c, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.row = row_c
        self.col = col_c
        self.grid = grid_c
        self.table_page_title = tpt_c
        self.table_section_title = tst_c
        self.question = question_c
        self.answer = answer_c
        self.max_len = max_len

    def __len__(self):
        return len(self.grid)

    def __getitem__(self, index):
        grid_sentence = str(self.grid[index])
        inputs = self.tokenizer(
            grid_sentence,
            max_length=self.max_len,
            padding='max_length',
            truncation='longest_first'
        )
        answer_sentence = str(self.answer[index])
        answer_tokenized = self.tokenizer(answer_sentence, max_length=self.max_len, padding='max_length',
                                          truncation='longest_first', return_tensors="pt")

        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'labels': answer_tokenized.input_ids
        }


if __name__ == '__main__':
    df_val = pd.read_json('fetaQA-v1_dev.jsonl', lines=True)
    df = df_val.head(10)
    row, col, grid, tpt, tst, question, _ = DataPreprocess(df).get_data()

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    representation_generation_set = RepresentationGenerationDataset(row, col, grid, tpt, tst, question, tokenizer, 128)
    representation_generation_loader = DataLoader(representation_generation_set, batch_size=1, num_workers=0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    RepresentationGenerationModel = BertModel.from_pretrained('bert-base-uncased')
    RepresentationGenerationModel.to(device)
    Table_Content_Representation = []
    row_representation = []
    column_representation = []
    grid_representation = []
    tcq_representation = []

    for _, data in enumerate(representation_generation_loader):
        row_ids = data[0]['row_ids_tensor'].to(device).squeeze(dim=0)
        row_mask = data[0]['row_mask_tensor'].to(device).squeeze(dim=0)
        row_token_type_ids = data[0]['row_token_type_ids_tensor'].to(device).squeeze(dim=0)
        with torch.no_grad():
            row_CLS = RepresentationGenerationModel(row_ids, row_mask, row_token_type_ids).pooler_output
        row_representation.append(row_CLS)

    for _, data in enumerate(representation_generation_loader):
        col_ids = data[1]['col_ids_tensor'].to(device).squeeze(dim=0)
        col_mask = data[1]['col_mask_tensor'].to(device).squeeze(dim=0)
        col_token_type_ids = data[1]['col_token_type_ids_tensor'].to(device).squeeze(dim=0)
        with torch.no_grad():
            col_CLS = RepresentationGenerationModel(col_ids, col_mask, col_token_type_ids).pooler_output
        column_representation.append(col_CLS)

    '''for _, data in enumerate(training_loader):
        grid_ids = data[2]['grid_ids_tensor'].to(device).squeeze(dim=0)
        grid_mask = data[2]['grid_mask_tensor'].to(device).squeeze(dim=0)
        grid_token_type_ids = data[2]['grid_token_type_ids_tensor'].to(device).squeeze(dim=0)
        grid_rep_for_each_table = []
        for i in range(grid_ids.size(0)):
            grid_rep_for_each_row = []
            for j in range(grid_ids.size(1)):
                # ids = grid_ids[i:i+1, j:j+1, :].squeeze(0)
                # mask = grid_mask[i:i+1, j:j+1, :].squeeze(0)
                # token_type_ids = grid_token_type_ids[i:i+1, j:j+1, :].squeeze(0)
                with torch.no_grad():
                    grid_CLS = RepresentationGenerationModel(grid_ids[i:i+1, j:j+1, :].squeeze(0),
                                                             grid_mask[i:i+1, j:j+1, :].squeeze(0),
                                                             grid_token_type_ids[i:i+1, j:j+1, :].squeeze(0)
                                                             ).pooler_output
                grid_rep_for_each_row.append(grid_CLS)
            grid_rep_for_each_table.append(grid_rep_for_each_row)
        grid_representation.append(grid_rep_for_each_table)'''

    '''for _, data in enumerate(training_loader):
        tcq_ids = data[3]['tcq_ids_tensor'].to(device)
        tcq_mask = data[3]['tcq_mask_tensor'].to(device)
        tcq_token_type_ids = data[3]['tcq_token_type_ids_tensor'].to(device)
        with torch.no_grad():
            tcq_CLS = RepresentationGenerationModel(tcq_ids, tcq_mask, tcq_token_type_ids).pooler_output
        tcq_representation.append(tcq_CLS)'''


    with open('row_representation.pickle', 'wb') as handle:
        pickle.dump(row_representation, handle)

    with open('col_representation.pickle', 'wb') as handle:
        pickle.dump(column_representation, handle)

    # with open('grid_representation.pickle', 'wb') as handle:
    #     pickle.dump(grid_representation, handle)

    # with open('tcq_representation.pickle', 'wb') as handle:
    #     pickle.dump(tcq_representation, handle)

