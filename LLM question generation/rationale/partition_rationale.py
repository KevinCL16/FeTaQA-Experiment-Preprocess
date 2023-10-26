import pandas as pd
import json

df_rationale_7326 = pd.read_json("fetaQA-v1_train_rationale_7326.jsonl", lines=True)
df_rationale_20_percent = df_rationale_7326.head(1465)
df_rationale_40_percent = df_rationale_7326.head(2930)
df_rationale_60_percent = df_rationale_7326.head(4396)
df_rationale_80_percent = df_rationale_7326.head(5861)

out1 = df_rationale_20_percent.to_json(orient='records', force_ascii=False)[1:-1].replace('\\/', '/').replace('},{', '}\n{')
out2 = df_rationale_40_percent.to_json(orient='records', force_ascii=False)[1:-1].replace('\\/', '/').replace('},{', '}\n{')
out3 = df_rationale_60_percent.to_json(orient='records', force_ascii=False)[1:-1].replace('\\/', '/').replace('},{', '}\n{')
out4 = df_rationale_80_percent.to_json(orient='records', force_ascii=False)[1:-1].replace('\\/', '/').replace('},{', '}\n{')

with open('fetaQA-v1_train_rationale_20_percent.jsonl', 'w', encoding='utf8') as f:
    f.write(out1)
with open('fetaQA-v1_train_rationale_40_percent.jsonl', 'w', encoding='utf8') as f:
    f.write(out2)
with open('fetaQA-v1_train_rationale_60_percent.jsonl', 'w', encoding='utf8') as f:
    f.write(out3)
with open('fetaQA-v1_train_rationale_80_percent.jsonl', 'w', encoding='utf8') as f:
    f.write(out4)