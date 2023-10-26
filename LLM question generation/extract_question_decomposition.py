import json
import pandas as pd
import ast

df = pd.read_json('fixed_1000_train_1-shot_detailed_CoT_by_gpt-3.5.json', orient='records')

'''
with open('1000_train_1-shot_question_decomposition_by_gpt-3.5.jsonl', 'a', encoding='utf-8') as f1:
    for idx, data in df.iterrows():
        cot = data["detailed_cot"]
        idx_begin = cot.index("tion:") + 5
        idx_end = cot.index("Locating")
        generated_qd = cot[idx_begin: idx_end]
        result = {
            "question decomposition": generated_qd,
            "id": idx + 1
        }
        json_object = json.dumps(result, indent=4, ensure_ascii=False) + '\n'
        f1.write(json_object)'''
