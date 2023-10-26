from sklearn.utils import shuffle
import json
from data_preprocess_attention import DataPreprocess
import pandas as pd

df_train = pd.read_json('D:/ComputerScience/CODES/fetaqa-gnn/fetaQA-v1_train.jsonl', lines=True)
# df_train = shuffle(df_train)
df = df_train.sample(n=10, ignore_index=True)
tpt, tst, table, question, answer = DataPreprocess(df).get_raw_data()
table_processed = []
for table_array in table:
    row_processed = []
    for row in table_array:
        row_with_delimiter = ' | '.join([''.join(grid) for grid in row])
        row_processed.append('[ROW] ' + row_with_delimiter)
    table_processed.append(row_processed)

for i in range(10):
    dictionary = {
        "table_page_title": tpt[i],
        "table_section_title": tst[i],
        "table": table_processed[i],
        "question": question[i],
        "answer": answer[i],
        "id": i + 1,
        "prompt": "In the given input above, you can see a question about the given table, and an answer to it."
                  " Now, you should output the content of cells that contain information crucial to answering"
                  " the question. Include the column name of these cells. Describe such information"
                  " in a natural form of language"
    }

    # Serializing json
    json_object = json.dumps(dictionary, indent=4, ensure_ascii=False) + '\n'

    # Writing to sample.json
    with open("./reasoning/train_10_for_claude_highlighted_cell.jsonl", "a", encoding='utf8') as outfile:
        outfile.write(json_object)
