import torch
from torch_geometric.data import Dataset, Data


class DataPreprocess:
    def __init__(self, df):
        self.tpt = df['table_page_title']
        self.tst = df['table_section_title']
        self.table_array = df['table_array']
        self.hc_id = df['highlighted_cell_ids']
        self.question = df['question']
        self.answer = df['answer']
        self.x = []
        self.edge_index = []
        self.len = len(df)

        for idx in range(self.len):
            x_temp, column, grid = self.get_grid(idx)
            edge_index_t = self.get_edge(column, grid)
            self.edge_index.append(torch.tensor(edge_index_t))
            self.x.append(x_temp)

    def get_grid(self, i):
        x_t = [self.tpt[i], self.tst[i], self.question[i]]
        col = []
        gd = []
        ci = 0
        j = 3

        for _ in self.table_array[i][0]:
            col.append({'col_header': _, 'global_id': j, 'col_id': ci})
            j = j + 1
            ci = ci + 1

        row = self.table_array[i][1:]
        for num in range(len(row[:])):
            col_id = 0
            for sth in row[:][num]:
                gd.append({'grid_content': sth, 'global_id': j, 'col_id': col_id})
                col_id += 1
                j = j + 1

        for _ in col:
            x_t.append(_['col_header'])
        for _ in gd:
            x_t.append((_['grid_content']))

        return x_t, col, gd

    def get_edge(self, col, gd):
        edge_index_temp = []
        for item in col:
            edge_index_temp.append([0, item['global_id']])
            edge_index_temp.append([item['global_id'], 0])
            edge_index_temp.append([1, item['global_id']])
            edge_index_temp.append([item['global_id'], 1])
            edge_index_temp.append([2, item['global_id']])
            edge_index_temp.append([item['global_id'], 2])
            for it in gd:
                if it['col_id'] == item['col_id']:
                    edge_index_temp.append([it['global_id'], item['global_id']])
                    edge_index_temp.append([item['global_id'], it['global_id']])

        return edge_index_temp

    def get_data(self):
        return self.x, self.edge_index, self.answer


class CustomDataset:
    def __init__(self, x, edge_index, answer):
        self.x = x
        self.edge_index = edge_index
        self.label = answer

    def len(self):
        return len(self.x)

    def get(self, idx):
        x_current = self.x[idx]
        edge_index_current = self.edge_index[idx].t().contiguous()

        return Data(x=x_current, edge_index=edge_index_current)
