import pandas as pd
import json

# df_extra_questions = pd.read_excel('Claude_generated_questions.xlsx')
df_extra_questions = pd.read_csv("reasoning/extra_questions_by_gpt-3.5.txt", sep='\n', header=None, names=["question"])
df_rationale = pd.read_csv("1000_rationale_by_gpt-3.5.txt", sep='\n', header=None, names=["rationale"])
df_extra_blank_1 = pd.read_csv("0.txt", header=None, names=["extra_question"])
df_extra_blank_2 = pd.DataFrame({"rationale": {}})

# df = pd.read_json('D:/ComputerScience/CODES/fetaqa-gnn/fetaQA-v1_train.jsonl', lines=True)
df_shuffle_table_1 = pd.read_json("table shuffle/fetaQA-v1_test_st.jsonl", lines=True)
# df_shuffle_table_2 = pd.read_json("table shuffle/fetaQA-v1_train_st.jsonl", lines=True)
# df_train = df.head(1000)
# df_shuffle_table_2["question"] = df_extra_questions
# df_shuffle_table_1 = pd.concat([df_shuffle_table_1, df_rationale], axis=1)
# df_shuffle_table_2 = pd.concat([df_shuffle_table_2, df_rationale], axis=1)
# result = df_shuffle_table_1.append(df_shuffle_table_2, ignore_index=True)
# result = df_shuffle_table_1

# result = pd.concat([df_shuffle_table, df_extra_questions, df_rationale], axis=1)
result = pd.concat([df_shuffle_table_1, df_extra_blank_2], axis=1)

# out = result.to_json(orient='records', force_ascii=False)[1:-1].replace('\\/', '/').replace('},{', '}\n{')
out = result.to_json(orient='records', force_ascii=False)[1:-1].replace('\\/', '/').replace('},{', '}\n{')\
      .replace('assistant:', '')
json_object = json.dumps(out, ensure_ascii=False)

with open('rationale/fetaQA-v1_test_st_r.jsonl', 'w', encoding='utf8') as f:
    f.write(out)
