import random
import pandas as pd
import json
# import evaluate

# sacrebleu = evaluate.load('sacrebleu')

df_prediction = pd.read_json("human_eval_20_model_generated_knowledge.jsonl", orient="records")
df_prediction["question"] = df_prediction["Question"]
df_baseline = pd.read_json("../LLM question generation/rationale/fetaQA-v1_test_llm_guided_hc.jsonl", lines=True)
# df_final = []
'''relevant_cells_instructions = ["Produce relevant data in the table cells to make it easier to answer questions about the table.",
                               "Generate table cell information that is directly related to answering the table's questions.",
                               "Provide valuable data in the table cells to aid in addressing the queries about the table.",
                               "Create table content that assists in understanding and answering questions about the table.",
                               "Generate meaningful information within the table cells to help answer the table's inquiries.",
                               "Produce relevant data points within the table to facilitate the answering of questions.",
                               "Create table cell information that is pertinent to the questions asked about the table.",
                               "Provide relevant details within the table cells to assist in answering queries about the table.",
                               "Generate table content that enables a better understanding of the questions and their answers.",
                               "Produce data within the table cells that makes it easier to respond to the table's questions."]'''

'''question_answering_instructions = ["Provide an answer to the question related to the given table.",
                                   "Answer the question using the data presented in the table.",
                                   "Use the information in the table to respond to the question asked.",
                                   "Offer a response that pertains to the table's contents and addresses the question.",
                                   "Based on the given table, provide an answer to the specified question.",
                                   "Utilize the data in the table to give a suitable response to the question.",
                                   "Respond to the question by referencing the data presented in the table.",
                                   "Use the table's information to formulate an accurate answer to the question.",
                                   "Examine the table and give an appropriate response to the question asked.",
                                   "Consider the table's contents and provide a well-founded answer to the question."]'''

'''rc_instructions = []
ans_instructions = []
for i in range(7326):
    idx = random.randint(0, 4)
    rc_instructions.append(relevant_cells_instructions[idx])
    ans_instructions.append(question_answering_instructions[idx])

df_rc = pd.DataFrame(rc_instructions)
df_ans = pd.DataFrame(ans_instructions)
df_rc_ins = pd.DataFrame()
df_ans_ins = pd.DataFrame()
df_rc_ins["df_rc_ins"] = df_rc
df_ans_ins["df_ans_ins"] = df_ans

result = pd.concat([df_relevant_cells, df_rc_ins, df_ans_ins], axis=1)'''

result_df = df_baseline.merge(df_prediction, on='question', how='inner')
df_final = pd.DataFrame()
df_final["Question"] = result_df['question']
df_final["Answer"] = result_df["Answer"]
df_final["LLM Predicted Knowledge"] = result_df["relevant_cells"]
table_row = []
for (idx_1, data_1) in result_df.iterrows():
    row_processed = []
    new_data = pd.Series()
    table = data_1["table_array"]
    col = True
    for row in table:
        row_with_delimiter = ' | '.join([''.join(grid) for grid in row])
        if col is True:
            row_processed.append('[COL] ' + row_with_delimiter)
            col = False
        else:
            row_processed.append('[ROW] ' + row_with_delimiter)
    table_row.append(row_processed)
df_final["Table Title"] = result_df["table_page_title"] + "|" + result_df["table_section_title"]
df_final["Table Rows"] = table_row

'''for (idx_1, data_1) in df_prediction.iterrows():
    new_data = pd.Series()
    pred = data_1["prediction"]
    label = data_1["answer_text"]
    score = sacrebleu.compute(predictions=[pred], references=[[label]])
    new_data["sacrebleu"] = score["score"]
    new_data["Prediction"] = data_1["prediction"]
    new_data["Question"] = data_1["question"]
    new_data["Ground Truth"] = data_1["answer_text"]
    new_data["Table Title"] = data_1["meta"]
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
    new_data["Table Rows"] = row_processed
    df_final.append(new_data)'''

# df_final = pd.DataFrame(df_final)
# df_final = df_final.drop("Table Column", axis=1)
# df_final = df_final.sort_values(by="sacrebleu", ascending=True)
# df_final = df_final.head(1600)
# df_final = df_final.drop("sacrebleu", axis=1)
# df_final = df_final.sample(80)
idx = [x for x in range(1, 21)]
df_final["id"] = idx

out = df_final.to_json(orient='records', force_ascii=False, indent=4)[1:-1].replace('\\/', '/').replace('},', '}\n')
json_object = json.dumps(out, ensure_ascii=False)

with open('human_eval_20_on_LLM_generated_knowledge.jsonl', 'w', encoding='utf8') as f:
    f.write(out)

'''out = result.to_json(orient='records', force_ascii=False)[1:-1].replace('\\/', '/').replace('},{', '}\n{')\
      .replace('assistant:', '')
json_object = json.dumps(out, ensure_ascii=False)

with open('../LLM question generation/rationale/fetaQA-v1_train_hc_7326_5_diverse_instructions.jsonl', 'w', encoding='utf8') as f:
    f.write(out)'''