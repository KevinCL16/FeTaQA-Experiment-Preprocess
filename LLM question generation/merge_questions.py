import pandas as pd
import json

# df_extra_questions = pd.read_excel('Claude_generated_questions.xlsx')
# df_extra_questions = pd.read_csv("reasoning/extra_questions_by_gpt-3.5.txt", sep='\n', header=None, names=["question"])
# df_extra_questions = pd.read_json("fixed_1000_train_set_3-shot_Q2_by_gpt-3.5.json", orient="records")["question"]
# df_extra_answers = pd.read_json("fixed_1000_train_set_3-shot_Q2_by_gpt-3.5.json", orient="records")["answer"]
# df_rationale = pd.read_json("fixed_1000_3-shot_rationale.json", orient="records")["rationale"]
df_relevant_cells_1 = pd.read_json("../predict/predict_flan-t5_fully_trained_generated_relevant_cells.json", orient="records")["prediction"]
df_r = pd.DataFrame()
df_r["relevant_cells"] = df_relevant_cells_1
# df_relevant_cells_2 = pd.read_json("fixed_1000-3000_1-shot_relevant_cells_by_gpt-3.5.json", orient="records")["relevant_cells"]
# df_extra_blank_1 = pd.read_csv("0.txt", header=None, names=["extra_question"])
df_extra_blank_2 = pd.DataFrame({"df_ans_ins": {}})
df_extra_blank_3 = pd.DataFrame({"df_rc_ins": {}})

df = pd.read_json('../LLM question generation/rationale/fetaQA-v1_dev_hc.jsonl', lines=True)
# df = pd.read_json('../LLM question generation/rationale/fetaQA-v1_test_llm_guided_hc.jsonl', lines=True)
# df_shuffle_table_1 = pd.read_json("rationale/fetaQA-v1_train_st_r-v2.jsonl", lines=True)
# df_shuffle_table_2 = pd.read_json("rationale/fetaQA-v1_test_st_r.jsonl", lines=True)
# df = df.head(500)
# df_relevant_cells = df_relevant_cells_1.append(df_relevant_cells_2, ignore_index=True)
# df_shuffle_table_2["question"] = df_extra_questions
# df_shuffle_table_2["answer"] = df_extra_answers
# df_shuffle_table_1 = pd.concat([df_shuffle_table_1, df_relevant_cells], axis=1)
# df_shuffle_table_2 = pd.concat([df, df_extra_blank_2], axis=1)
# result = df_shuffle_table_1.append(df_shuffle_table_2, ignore_index=True)
result = pd.concat([df, df_extra_blank_2, df_extra_blank_3], axis=1)

# result = pd.concat([df_shuffle_table, df_extra_questions, df_rationale], axis=1)
# result = pd.concat([df, df_extra_blank_2], axis=1)

# out = result.to_json(orient='records', force_ascii=False)[1:-1].replace('\\/', '/').replace('},{', '}\n{')
out = result.to_json(orient='records', force_ascii=False)[1:-1].replace('\\/', '/').replace('},{', '}\n{')\
      .replace('assistant:', '')
json_object = json.dumps(out, ensure_ascii=False)

with open('rationale/fetaQA-v1_dev_hc_ans_rc_ins.jsonl', 'w', encoding='utf8') as f:
    f.write(out)
