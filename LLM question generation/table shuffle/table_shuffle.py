import random
import json
import numpy as np
import torch
import pandas as pd

df_train = pd.read_json('D:/ComputerScience/CODES/fetaqa-gnn/fetaQA-v1_train.jsonl', lines=True)
df = df_train.head(1000)

original_table = df['table_array'].to_numpy()
shuffled_table = []
for each_table in original_table:
    row_shuffled_table = np.random.permutation(each_table[1:])
    row_shuffled_table = np.append([each_table[0]], row_shuffled_table, axis=0)
    gather_index = [x for x in range(len(each_table[0]))]
    random.shuffle(gather_index)
    each_shuffled_table = row_shuffled_table.take(indices=gather_index, axis=1)
    shuffled_table.append(each_shuffled_table)

df['shuffled_table'] = shuffled_table
print(df)

out = df.to_json(orient='records', force_ascii=False)[1:-1].replace('\\/', '/').replace('},{', '}\n{')\
      .replace('assistant:', '')
json_object = json.dumps(out, ensure_ascii=False)

with open('fetaQA-v1_train_st.jsonl', 'w', encoding='utf8') as f:
    f.write(out)
