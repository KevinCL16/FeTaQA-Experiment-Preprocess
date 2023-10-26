import json
import evaluate
import pandas as pd

sacrebleu = evaluate.load("sacrebleu")
df_predict_1 = pd.read_json("predict_flan-t5_full_rationale_mtl_llm_guided_31.46.json", orient="records")
# df_predict_2 = pd.read_json("predict_flan-t5_full_relevant_cells_mtl_34.28.json", orient="records")
df_predict_3 = pd.read_json("predict_flan-t5_full_relevant_cells_mtl_llm_guided_testing_41.32.json", orient="records")
result_df = []

for (idx_1, data_1), (idx_3, data_3) in zip(df_predict_1.iterrows(), df_predict_3.iterrows()):
    new_data = pd.Series()
    pred_1 = data_1["prediction"]

    pred_3 = data_3["prediction"]
    label = data_1["answer_text"]
    score_1 = sacrebleu.compute(predictions=[pred_1], references=[[label]])

    score_3 = sacrebleu.compute(predictions=[pred_3], references=[[label]])

    diff_3_1 = score_3["score"] - score_1["score"]

    new_data["sacrebleu_rationale_llm_guided"] = score_1["score"]

    new_data["sacrebleu_relevant_cells_llm_guided"] = score_3["score"]
    new_data["sacrebleu_difference"] = diff_3_1
    new_data["question"] = data_1["question"]
    new_data["answer_text"] = label
    new_data["rationale_llm_guided_prediction"] = pred_1

    new_data["relevant_cells_llm_guided_prediction"] = pred_3
    new_data["table_title"] = data_1['meta']
    table_row = []
    row_processed = []
    table = [data_1["table"]["header"]]
    col = True
    for item in data_1["table"]["rows"]:
        table.append(item)

    for row in table:
        row_with_delimiter = ' | '.join([''.join(grid) for grid in row])
        if col is True:
            row_processed.append('[COL] ' + row_with_delimiter)
            col = False
        else:
            row_processed.append('[ROW] ' + row_with_delimiter)
    table_row.append(row_processed)
    new_data["table_rows"] = table_row

    result_df.append(new_data)

df = pd.DataFrame(result_df)
df = df.sort_values(by="sacrebleu_difference", ascending=False)

out = df.to_json(orient='records', force_ascii=False, indent=4)[1:-1].replace('\\/', '/').replace('},{', '}\n{')
json_object = json.dumps(out, ensure_ascii=False)

with open('side_by_side_relevant_cells_vs_rationale.jsonl', 'w', encoding='utf8') as f:
    f.write(out)
