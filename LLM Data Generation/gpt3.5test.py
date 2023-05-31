import openai
import json
from data_preprocess_attention import DataPreprocess
import pandas as pd
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
                {"role": "system", "content": "You will be presented with a table and a question-answer pair. "
                                              "You need to generate a faithful and nuanced step by step reasoning path "
                                              "explaining how to derive from the question to the answer. "
                                              "The explanation should be less than 100 words with no line changes"},
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

    with open('1000_rationale_by_gpt-3.5.txt', 'a', encoding='utf-8') as f1:
        for i in range(1000):
            dictionary = {
                "table_page_title": tpt[i],
                "table_section_title": tst[i],
                "table": table_processed[i],
                "question": question[i],
                "answer": answer[i],
                "id": i + 1,
                "prompt": "Given the above table and a question-answer pair regarding information in such table, generate "
                          "a faithful step by step reasoning path explaining why the question should lead to such an answer."
                          " The explanation should be less than 100 words with no line changes."
            }
            prompt = str(dictionary)
            res = completion_with_backoff(prompt) + '\n'
            f1.write(res)
