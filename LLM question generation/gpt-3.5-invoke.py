import openai
import json
from data_preprocess_attention import DataPreprocess
import pandas as pd
import random

# 个人API Key
openai.api_key = ""
# 企业API Key
openai.organization = ""

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential, stop_after_delay,
)  # for exponential backoff


# 防止调用频率超过每分钟上限的等待代码######
@retry(wait=wait_random_exponential(min=0.02, max=1), stop=(stop_after_delay(60) | stop_after_attempt(500)))
# 调用OpenAI API
def completion_with_backoff(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                # 系统prompt
                "role": "system", "content": "Generate an answer given the table and a question. Generate the answer"
                                             "using the style of the answer given in your learning example."
                                             " The answer style is concise and void of extra explanation"
            },
            {
                # 每次调用的输入
                "role": "user", "content": prompt
            }
        ]
    )
    # API返回回答
    result = response.choices[0]['message']
    return f"{result['content']}"
    # return f"{result['role']}:{result['content']}"


# 我之前的课题相关的处理，忽略
def process_table(dataframe, table_processed):
    tpt, tst, table, question, answer = DataPreprocess(dataframe).get_raw_data()
    for table_array in table:
        row_processed = []
        col = True
        for row in table_array:
            row_with_delimiter = ' | '.join([''.join(grid) for grid in row])
            if col is True:
                row_processed.append('[COL] ' + row_with_delimiter)
                col = False
            else:
                row_processed.append('[ROW] ' + row_with_delimiter)
        table_processed.append(row_processed)
    return tpt, tst, question, answer


if __name__ == '__main__':
    # 读数据
    df_train = pd.read_json('../fetaQA-v1_train.jsonl', lines=True)
    df_test = pd.read_json('../fetaQA-v1_test.jsonl', lines=True)
    # df_train = shuffle(df_train)
    # df_train = df_train.head(1000)
    # df_test = df_test.head(100)
    # df_exemplar = pd.read_json('/LLM question generation/reasoning/train_10_for_claude_highlighted_cell.jsonl', orient="records")
    # df_exemplar = df_exemplar.to_dict('records')

    train_table_processed = []
    test_table_processed = []

    train_tpt, train_tst, train_q, train_a = process_table(df_train, train_table_processed)
    test_tpt, test_tst, test_q, test_a = process_table(df_test, test_table_processed)

    # 将返回的结果写进json文件
    with open('2003_TEST_1-shot_by_gpt-3.5_with_specific_style_requirement.jsonl', 'a', encoding='utf-8') as f1:
        '''idx = []
        for i in range(0, 3):
            idx.append(random.randint(0, 9))'''

        idx = random.randint(0, 7325)
        # 我在这里准备的few-shot样例，但是是通过上面注释掉的函数实现的，可以自行修改
        few_shot_examplar = {
            "table_page_title": train_tpt[idx],
            "table_section_title": train_tst[idx],
            "table": train_table_processed[idx],
            "question": train_q[idx],
            "answer": train_a[idx],
        }
        # 循环（待调用数据的长度）次
        for i in range(0, 2003):
            # 准备输入
            chat_input = {
                "Example": "Here is an example for you to learn: " + str(few_shot_examplar) +
                           "Above is the example for you to understand task specifications. "
                           "Focus now, generate an answer for the table below, remember to output answers in the style"
                           " of the answer given in your learning example, which is concise and void of extra explantion: ",
                "table_page_title": test_tpt[i],
                "table_section_title": test_tst[i],
                "table": test_table_processed[i],
                # "answer": train_a[i],
                "question": test_q[i],
                # "reminder:": " The information you generate should be less than 128 tokens."
            }
            prompt = str(chat_input)
            # 调用上面那个函数
            res = completion_with_backoff(prompt) + '\n'
            out = res.replace('\n', ' ').replace('},{', '}\n{')
            # idx_c = out.index("C:")
            # idx_a = out.index("A:")
            # generated_c = out[idx_c + 3:]
            # generated_a = out[idx_a + 3:]

            # 输出json文件的内容格式，自定，想写什么些什么
            result = {
                "prediction": out,
                "seq_out": test_a[i],
                "id": i + 1
            }
            json_object = json.dumps(result, indent=4, ensure_ascii=False) + ',\n'
            f1.write(json_object)
