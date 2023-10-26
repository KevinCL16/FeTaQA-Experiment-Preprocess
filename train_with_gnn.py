import json
import evaluate
import nltk
from data_preprocess import CustomDataset, DataPreprocess
from datasets import load_dataset, load_metric
import pandas as pd
import numpy as np
import os
from sentence_transformers import SentenceTransformer
import torch
from torch_geometric.loader import DataLoader
from model_with_gnn import GraphSeq2Seq
from transformers import T5Tokenizer

nltk.set_proxy('127.0.0.1:7890')
# nltk.download('punkt')
df_train = pd.read_json('fetaQA-v1_train.jsonl', lines=True)
df_val = pd.read_json('fetaQA-v1_dev.jsonl', lines=True)
df_test = pd.read_json('fetaQA-v1_test.jsonl', lines=True)
tokenizer = T5Tokenizer.from_pretrained("t5-small")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_dataset(df):
    labels = []
    df = df.head(128)
    x, edge_index, answers = DataPreprocess(df).get_data()
    answers = answers.to_list()
    for each_answer in answers:
        labels.append(tokenizer(each_answer, return_tensors="pt", padding=True).input_ids.to(device))

    progress = 0
    node_embedding = []
    node_encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    print('model loaded')

    for node in x:
        node_embedding.append(torch.tensor(node_encoder.encode(node)))
        print(progress)
        progress += 1

    raw_dataset = CustomDataset(node_embedding, edge_index, answers)
    dataset_len = raw_dataset.len()
    processed_dataset = []
    for idx in range(dataset_len):
        processed_dataset.append(raw_dataset.get(idx).to(device))

    return processed_dataset, labels


# Hyper-Parameter
BATCH_SIZE = 64

train_dataset, train_label = get_dataset(df_train)
validation_dataset, val_label = get_dataset(df_val)
# test_dataset, test_label = get_dataset(df_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True)

model = GraphSeq2Seq(384, 512)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
model.to(device)
# criterion = torch.nn.CrossEntropyLoss()


def train():
    model.train()
    batch_size = BATCH_SIZE
    i = 0
    for data, label in zip(train_loader, train_label):
        t_label = train_label[i:i+batch_size]
        loss = model(data.x, data.edge_index, t_label, data.batch, batch_size)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        i = i + batch_size

    return loss


def validate(loader, label):
    model.eval()
    i = 0
    batch_size = BATCH_SIZE
    predictions = []
    targets = []
    for data, _ in zip(loader, label):
        v_label = label[i:i + batch_size]
        probs_list = model(data.x, data.edge_index, v_label, data.batch, batch_size, train_or_not=False)
        for probs in probs_list:
            preds = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(probs.argmax(dim=1)))
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


for epoch in range(1, 6):
    loss = train()
    result = validate(val_loader, val_label)
    print(f'Epoch: {epoch:03d}, Loss: {loss}, sacrebleu: {result[0]}, rouge: {result[1]}, meteor: {result[2]}, '
          f'bleurt: {result[3]}, bertscore: {result[4]}')

    print(f'Epoch: {epoch:03d}, Loss: {loss}')

# result = validate(test_loader, test_label)
# print(f'sacrebleu: {result[0]}, rouge: {result[1]}, meteor: {result[2]}, bleurt: {result[3]}, bertscore: {result[4]}')