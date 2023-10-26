from sklearn.utils import shuffle
import json
from data_preprocess_attention import DataPreprocess
import pandas as pd

df_train = pd.read_json('D:/ComputerScience/CODES/fetaqa-gnn/fetaQA-v1_train.jsonl', lines=True)
# df_train = shuffle(df_train)
df = df_train.head(1000)
tpt, tst, table, question, answer = DataPreprocess(df).get_raw_data()
table_processed = []
for table_array in table:
    row_processed = []
    for row in table_array:
        row_with_delimiter = ' | '.join([''.join(grid) for grid in row])
        row_processed.append('[ROW] ' + row_with_delimiter)
    table_processed.append(row_processed)

dict_all = []
for i in range(1000):
    dictionary = {
        "prompt": "Given the following json file, generate a question based on the table context and the answer.",
        "table_page_title": tpt[i],
        "table_section_title": tst[i],
        "table": table_processed[i],
        "answer": answer[i],
        "id": i+1
    }
    dict_all.append(dictionary)

# Serializing json
json_object = json.dumps(dict_all, ensure_ascii=False)[1:-1].replace('}, {', '}\n{')

# Writing to sample.json
with open("random_52_for_chatgpt.txt", "w", encoding='utf8') as outfile:
    outfile.write(json_object)
