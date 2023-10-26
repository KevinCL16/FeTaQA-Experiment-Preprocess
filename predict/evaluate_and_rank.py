import json

import evaluate
import pandas as pd

sacrebleu = evaluate.load("sacrebleu")
df_predict = pd.read_json("predict_flan-t5_1000_test_on_trainset.json", orient="records")
df_detailed_cot_base = pd.read_json("D:/ComputerScience/CODES/fetaqa-gnn/LLM question generation/fixed_1000_train_1-shot_detailed_CoT_by_gpt-3.5.json", orient="records")["detailed_cot"]
df_detailed_cot = pd.DataFrame()
df_detailed_cot["relevant_cells"] = df_detailed_cot_base
df_relevant_cells = pd.read_json("D:/ComputerScience/CODES/fetaqa-gnn/LLM question generation/fixed_1000_1-shot_relevant_cells_by_gpt-3.5.json", orient="records")["relevant_cells"]
df = pd.read_json('D:/ComputerScience/CODES/fetaqa-gnn/fetaQA-v1_train.jsonl', lines=True)
df_train = df.head(1000)
result_df = []

for idx, data in df_predict.iterrows():
    new_data = pd.Series()
    pred = data["prediction"]
    label = data["answer_text"]
    score = sacrebleu.compute(predictions=[pred], references=[[label]])
    if score["score"] <= 3:
        df_dcot = df_detailed_cot.loc[[idx]]
        new_data = pd.concat([df_train.loc[[idx]], df_dcot], axis=1).squeeze()
    elif score["score"] > 3:
        df_r = df_relevant_cells.loc[[idx]]
        new_data = pd.concat([df_train.loc[[idx]], df_r], axis=1).squeeze()
    '''new_data["sacrebleu"] = score["score"]
    new_data["question"] = data["question"]
    new_data["prediction"] = pred
    new_data["answer_text"] = label
    new_data["relevant_cells"] = data["relevant_cells"]
    new_data["table_header"] = data["table"]["header"]
    new_data["table_rows"] = data["table"]["rows"]'''
    result_df.append(new_data)

df = pd.DataFrame(result_df)
# df = df.sort_values(by="sacrebleu", ascending=True)

out = df.to_json(orient='records', force_ascii=False)[1:-1].replace('\\/', '/').replace('},{', '}\n{')
json_object = json.dumps(out, ensure_ascii=False)

with open('../LLM question generation/rationale/fetaQA-v1_train_partial_detailed_cot.jsonl', 'w', encoding='utf8') as f:
    f.write(out)
