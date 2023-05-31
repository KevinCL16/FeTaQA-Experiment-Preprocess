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

for i in range(20):
    dictionary = {
        "table_page_title": tpt[i],
        "table_section_title": tst[i],
        "table": table_processed[i],
        "question": question[i],
        "answer": answer[i],
        "id": i+1,
        "prompt": "Given the above table and a question-answer pair regarding information in such table, generate "
                  "a faithful step by step reasoning path explaining why the question should lead to such an answer."
                  " Ideally, the explanation should be less than 100 words."
    }

    # Serializing json
    json_object = json.dumps(dictionary, indent=4, ensure_ascii=False) + '\n'

    # Writing to sample.json
    with open("./reasoning/random_20_for_rationale.jsonl", "a", encoding='utf8') as outfile:
        outfile.write(json_object)
