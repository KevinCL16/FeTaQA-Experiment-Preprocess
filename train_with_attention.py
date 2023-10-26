from transformers import T5Tokenizer
import evaluate
import pickle
import nltk
from data_preprocess_attention import CustomDataset, DataPreprocess
from model_with_attention import T5TableQA
from torch.utils.data import DataLoader
import pandas as pd
import torch

nltk.set_proxy('127.0.0.1:7890')
# nltk.download('punkt')
df_train = pd.read_json('fetaQA-v1_train.jsonl', lines=True)
df_val = pd.read_json('fetaQA-v1_dev.jsonl', lines=True)
df_test = pd.read_json('fetaQA-v1_test.jsonl', lines=True)
tokenizer = T5Tokenizer.from_pretrained("t5-small")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

with open('./weighted_representation/weighted_col_representation.pickle', 'rb') as handle:
    weighted_col_representation = pickle.load(handle)

with open('./weighted_representation/weighted_row_representation.pickle', 'rb') as handle:
    weighted_row_representation = pickle.load(handle)

with open('./weighted_representation/weighted_grid_representation.pickle', 'rb') as handle:
    weighted_grid_representation = pickle.load(handle)

row, col, grid, tpt, tst, question, answer = DataPreprocess(df_val.head(10)).get_data()
_, _, _, _, _, _, val_targets = DataPreprocess(df_val)
training_set = CustomDataset(row, col, grid, tpt, tst, question, answer, tokenizer, 128)

train_params = {'batch_size': 1,
                'shuffle': True,
                'num_workers': 0
                }

training_loader = DataLoader(training_set, **train_params)

model = T5TableQA()
model.to(device)

optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-4)


def train(epoch):
    model.train()
    for idx, data in enumerate(training_loader):
        optimizer.zero_grad()
        labels = data['labels'].to(device).squeeze(dim=1)
        w_r_rep = weighted_row_representation[idx]
        w_c_rep = weighted_col_representation[idx]
        w_g_rep = weighted_grid_representation[idx]
        input_embeds = torch.cat((w_g_rep, torch.cat((w_r_rep, w_c_rep), dim=1)), dim=1)

        loss = model(input_embeds, labels)

        if idx % 1 == 0:
            print(f'Epoch: {epoch}, Loss:  {loss.item()}')
        # 反向传播
        loss.backward()
        optimizer.step()

        return loss


def validate(loader, label):
    model.eval()
    predictions = []
    targets = []
    for data, _ in zip(loader, label):
        preds = model.generate(data['ids'].to(device, dtype=torch.long))
        predictions.append(preds)

    for each_label in label:
        each_label = each_label.squeeze(dim=0)
        target = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(each_label))
        targets.append(target)


    sacrebleu = evaluate.load("sacrebleu")
    rouge = evaluate.load('rouge')
    meteor = evaluate.load('meteor')
    bleurt = evaluate.load("bleurt", module_type="metric")
    bertscore = evaluate.load("bertscore")
    result = [sacrebleu.compute(predictions=predictions, references=targets),
              rouge.compute(predictions=predictions, references=targets),
              meteor.compute(predictions=predictions, references=targets),
              bleurt.compute(predictions=predictions, references=targets),
              bertscore.compute(predictions=predictions, references=targets, lang="en")]

    return result


for epoch in range(50):
    print('------------------------------------',epoch,'-----------------------------------------')
    train(epoch)
