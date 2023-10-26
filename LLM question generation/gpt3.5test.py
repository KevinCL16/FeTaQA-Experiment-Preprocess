import openai
import json
from data_preprocess_attention import DataPreprocess
import pandas as pd
import random

# 获取API Key
# openai.api_key = "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"
openai.api_key = "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"
openai.organization = "org-IfPzna5SLpzIsCow27fYuPda"

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential, stop_after_delay,
)  # for exponential backoff


@retry(wait=wait_random_exponential(min=0.02, max=1), stop=(stop_after_delay(60) | stop_after_attempt(500)))
def completion_with_backoff(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system", "content": "Given the above table and the question-answer pair, generate"
                                             " a faithful step by step reasoning path explaining how to answer"
                                             " said question. The explanation should be less than 128 tokens!"
                                             " Be concise and precise."
            },
            {"role": "user", "content": prompt}
        ]
    )
    result = response.choices[0]['message']
    return f"{result['content']}"
    # return f"{result['role']}:{result['content']}"


# Define a function that adds a delay to a Completion API call
# def delayed_completion(delay_in_seconds: float = 1, **request_params):
#     """Delay a completion by a specified amount of time."""
#     # Call the Completion API and return the result
#     response = openai.ChatCompletion.create(**request_params)
#     result = response.choices[0]['message']
#     # Sleep for the delay
#     time.sleep(delay_in_seconds)
#     return  (f"{result['role']}: {result['content']}")


# def completion_with_backoff(**kwargs):
#     completions = openai.ChatCompletion.create(**kwargs)
#     result = completions.choices[0]['message']
#     return  (f"{result['role']}: {result['content']}")


# def generate_answer(prompt):
#     response = openai.ChatCompletion.create(
#         model="gpt-3.5-turbo",
#         messages=[
#             {"role": "system", "content": "你是一个具备因果推理，常识知识的人。"},
#             {"role": "user", "content": "我给你一个句子，请生成10个可以填入该句子中<mask>位置中的候选词，并且按照合理性从高到低排序，不要换行。" + prompt}
#         ]
#     )
#     result = response.choices[0]['message']
#     return f"{result['role']}:{result['content']}"


def process_table(dataframe, table_processed):
    tpt, tst, table, question, answer = DataPreprocess(dataframe).get_raw_data()
    for table_array in table:
        row_processed = []
        for row in table_array:
            row_with_delimiter = ' | '.join([''.join(grid) for grid in row])
            row_processed.append('[ROW] ' + row_with_delimiter)
        table_processed.append(row_processed)
    return tpt, tst, question, answer


final_res = []
if __name__ == '__main__':
    # with open('extra_questions_by_gpt-3.5.txt', 'a', encoding='utf-8') as f1:
    # with open('random_52_for_chatgpt.txt', 'r', encoding='utf-8') as f:
    # for line in f.readlines():
    # prompt = line
    # res = completion_with_backoff(prompt) + '\n'
    # res = generate_answer(prompt) + '\n'
    # f1.write(res)
    # time.sleep(20)

    df_train = pd.read_json('D:/ComputerScience/CODES/fetaqa-gnn/fetaQA-v1_train.jsonl', lines=True)
    df_test = pd.read_json('D:/ComputerScience/CODES/fetaqa-gnn/fetaQA-v1_test.jsonl', lines=True)
    # df_train = shuffle(df_train)
    # df_train = df_train.head(1000)
    # df_test = df_test.head(100)
    df_exemplar = pd.read_json('D:/ComputerScience/CODES/fetaqa-gnn/LLM question generation/reasoning/train_10_for_claude_rationale.jsonl', orient="records")
    df_exemplar = df_exemplar.to_dict('records')

    train_table_processed = []
    test_table_processed = []

    train_tpt, train_tst, train_q, train_a = process_table(df_train, train_table_processed)
    test_tpt, test_tst, test_q, test_a = process_table(df_test, test_table_processed)

    with open('2003_test_set_1-shot_rationale_by_gpt-3.5.jsonl', 'a', encoding='utf-8') as f1:
        '''idx = []
        for i in range(0, 3):
            idx.append(random.randint(0, 9))

        few_shot_examplar = [df_exemplar[idx[0]], df_exemplar[idx[1]], df_exemplar[idx[2]]]'''

        for i in range(1978, 2003):
            index = random.randint(0, 8)
            few_shot_examplar = df_exemplar[index]

            dictionary = {
                "Demonstration:": "Here is an examples for you to learn: " + str(few_shot_examplar) +
                                  " Now, you will generate step-by-step reasoning for the following table",
                "table_page_title": test_tpt[i],
                "table_section_title": test_tst[i],
                "table": test_table_processed[i],
                "question": test_q[i],
                # "answer": train_a[i],
                "id": i + 1,
                "reminder:": " The information you generate should be less than 128 tokens."
            }
            prompt = str(dictionary)
            res = completion_with_backoff(prompt) + '\n'
            out = res.replace('\n', ' ').replace('},{', '}\n{')
            # idx_c = out.index("C:")
            # idx_a = out.index("A:")
            # generated_c = out[idx_c + 3:]
            # generated_a = out[idx_a + 3:]
            result = {
                "rationale": out,
                "id": i + 1
            }
            json_object = json.dumps(result, indent=4, ensure_ascii=False) + '\n'
            f1.write(json_object)
