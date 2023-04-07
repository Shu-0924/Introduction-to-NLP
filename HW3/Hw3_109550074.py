from transformers import BertTokenizerFast, BertForMultipleChoice
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score
from jieba import analyse
from tqdm import tqdm
import numpy as np
import torch
import json
import csv

device = 'cuda'
MAX_LEN = 500
epoch_num = 5
batch_size = 1
tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
bert_model = BertForMultipleChoice.from_pretrained('bert-base-chinese', return_dict=False)


class BertForMultipleChoiceChinese(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = bert_model.bert

        self.dropout_high = torch.nn.Dropout(0.5)
        self.dropout_mid = torch.nn.Dropout(0.3)
        self.dropout_low = torch.nn.Dropout(0.2)

        self.dense_1 = torch.nn.Linear(768, 256)
        self.dense_2 = torch.nn.Linear(256, 256)
        self.dense_3 = torch.nn.Linear(256, 64)
        self.dense_4 = torch.nn.Linear(64, 64)
        self.dense_5 = torch.nn.Linear(64, 16)
        self.dense_6 = torch.nn.Linear(16, 16)
        self.dense_out = torch.nn.Linear(16, 1)

    def forward(self, input_ids, token_type_ids, attention_mask):
        input_ids = input_ids.view(-1, input_ids.size(-1))
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
        attention_mask = attention_mask.view(-1, attention_mask.size(-1))
        sequence_output, pooled_output = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )

        pooled_output = self.dropout_high(pooled_output)
        pooled_output = self.dense_1(pooled_output)
        pooled_output = self.dropout_low(pooled_output)
        pooled_output = self.dense_2(pooled_output)
        pooled_output = self.dropout_mid(pooled_output)
        pooled_output = self.dense_3(pooled_output)
        pooled_output = self.dropout_low(pooled_output)
        pooled_output = self.dense_4(pooled_output)
        pooled_output = self.dropout_mid(pooled_output)
        pooled_output = self.dense_5(pooled_output)
        pooled_output = self.dropout_low(pooled_output)
        pooled_output = self.dense_6(pooled_output)
        pooled_output = self.dropout_mid(pooled_output)

        predict = self.dense_out(pooled_output)
        predict_reshaped = predict.view(-1, 4)
        return predict_reshaped


def load_json(path):
    with open(path, 'r', encoding='utf-8') as jsonfile:
        content = json.load(jsonfile)
    return content


def extract_from_article(context, question, choices):
    sentences = context.split('。')
    keywords = question + '。' + '。'.join(choices) + '。'
    keywords_tfidf = analyse.extract_tags(keywords, withWeight=True)

    score_list = [0] * len(sentences)
    for i, sentence in enumerate(sentences):
        sentence_tfidf = dict(analyse.extract_tags(sentence, withWeight=True))
        for keyword, score in keywords_tfidf:
            if sentence_tfidf.get(keyword) is not None:
                score_list[i] += score * sentence_tfidf[keyword]

    extract = []
    for i, score in enumerate(score_list):
        if score > 0.25:
            extract.append(sentences[i])
    return '。'.join(extract)


def get_generator(raw_data, train=True):
    input_ids_list, token_type_ids_list, attention_mask_list, ans_list = [], [], [], []
    for context, QAs, _ in tqdm(raw_data):
        first_part = context[0]

        for QA in QAs:
            cnt, flag = 0, 0
            second_part, third_part = QA['question'], []
            for choice in QA['choice']:
                if train is True and flag == 0 and choice == QA['answer']:
                    ans_list.append(cnt)
                    flag += 1
                third_part.append(choice)
                cnt += 1

            extracted = extract_from_article(first_part, second_part, third_part)
            extracted = extracted if (len(extracted) > 1) else first_part

            while len(third_part) < 4:
                third_part.append('无此选项')

            # deal with imbalaced data
            if train is True:
                target_idx = len(ans_list) % 4
                tmp = third_part[target_idx]
                third_part[target_idx] = third_part[ans_list[-1]]
                third_part[ans_list[-1]] = tmp
                ans_list[-1] = target_idx

            text = [None]
            choice_max_len = -1
            for choice in third_part:
                if len(choice) > choice_max_len:
                    choice_max_len = len(choice)

            total_len = len(extracted) + len(second_part) + choice_max_len
            if total_len + 10 > MAX_LEN:
                end = MAX_LEN - 10 - len(second_part) - choice_max_len
                text[0] = extracted[:end] + '[SEP]' + second_part
            else:
                text[0] = extracted + '[SEP]' + second_part

            text_encode = tokenizer(
                text * 4,
                third_part,
                max_length=MAX_LEN,
                padding='max_length',
                return_tensors='pt'
            )

            input_ids_list.append(text_encode['input_ids'].unsqueeze(0))
            token_type_ids_list.append(text_encode['token_type_ids'].unsqueeze(0))
            attention_mask_list.append(text_encode['attention_mask'].unsqueeze(0))

    input_ids_list = torch.cat(input_ids_list)
    token_type_ids_list = torch.cat(token_type_ids_list)
    attention_mask_list = torch.cat(attention_mask_list)
    ans_list = torch.tensor(ans_list)
    if train is True:
        data_zip = TensorDataset(input_ids_list, token_type_ids_list, attention_mask_list, ans_list)
        return DataLoader(data_zip, batch_size=batch_size, shuffle=True)
    else:
        data_zip = TensorDataset(input_ids_list, token_type_ids_list, attention_mask_list)
        return DataLoader(data_zip, batch_size=batch_size, shuffle=False)


train_data = load_json('./train_HW3dataset.json')
train_data_generator = get_generator(train_data, train=True)

val_data = load_json('./dev_HW3dataset.json')
val_data_generator = get_generator(val_data, train=True)


model = BertForMultipleChoiceChinese()

for name, param in model.named_parameters():
    if name.startswith("bert.embeddings."):
        param.requires_grad = False
    if name.startswith("bert.encoder.layer.0."):
        param.requires_grad = False
    if name.startswith("bert.encoder.layer.1."):
        param.requires_grad = False
    if name.startswith("bert.encoder.layer.2."):
        param.requires_grad = False
    if name.startswith("bert.encoder.layer.3."):
        param.requires_grad = False
    if name.startswith("bert.encoder.layer.4."):
        param.requires_grad = False
    if name.startswith("bert.encoder.layer.5."):
        param.requires_grad = False
    if name.startswith("bert.encoder.layer.6."):
        param.requires_grad = False

for name, param in model.named_parameters():
    print(name, param.requires_grad)


model = model.to(device)
loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

best_score = 0
for epoch in range(epoch_num):
    train_loss, val_loss = 0.0, 0.0
    train_count, val_count = 0.0, 0.0
    train_true, val_true = [], []
    train_pred, val_pred = [], []

    model.train()
    model.zero_grad()
    for i, (input_ids, token_type_ids, attention_mask, y_true) in enumerate(train_data_generator):
        input_ids = input_ids.to(device)
        token_type_ids = token_type_ids.to(device)
        attention_mask = attention_mask.to(device)
        y_true = y_true.to(device)

        y_pred = model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )
        batch_loss = loss(y_pred, y_true)
        batch_loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        # scheduler.step()
        model.zero_grad()

        train_loss += batch_loss.item()
        train_count += input_ids.shape[0]
        train_true += y_true.tolist()
        train_pred += np.argmax(y_pred.cpu().detach().numpy(), axis=1).tolist()
        if (i + 1) % 50 == 0 or (i + 1) == len(train_data_generator):
            train_acc = round(accuracy_score(train_true, train_pred), 4)
            print(f'Epoch {epoch + 1}/{epoch_num} [{i + 1}/{len(train_data_generator)}]:'
                  + f' loss:{train_loss / train_count:.4f} - acc:{train_acc:.4f}')

    model.eval()
    for i, (input_ids, token_type_ids, attention_mask, y_true) in enumerate(val_data_generator):
        input_ids = input_ids.to(device)
        token_type_ids = token_type_ids.to(device)
        attention_mask = attention_mask.to(device)
        y_true = y_true.to(device)

        with torch.no_grad():
            y_pred = model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
            )

        batch_loss = loss(y_pred, y_true)

        val_loss += batch_loss.item()
        val_count += input_ids.shape[0]
        val_true += y_true.tolist()
        val_pred += np.argmax(y_pred.cpu().detach().numpy(), axis=1).tolist()
        if (i + 1) % 50 == 0 or (i + 1) == len(val_data_generator):
            val_acc = round(accuracy_score(val_true, val_pred), 4)
            print(f'Epoch {epoch + 1}/{epoch_num} [{i + 1}/{len(val_data_generator)}]:'
                  + f' val_loss:{val_loss / val_count:.4f} - val_acc:{val_acc:.4f}')

    if accuracy_score(val_true, val_pred) > best_score:
        best_score = accuracy_score(val_true, val_pred)
        print('Saving weight...')
        torch.save(model.state_dict(), 'bert_weight.pth')

print(f"Best Validation Accuracy: {best_score}")


test_data = load_json('./test_HW3dataset.json')
test_data_generator = get_generator(test_data, train=False)

model.eval()
test_pred = []
for input_ids, token_type_ids, attention_mask in tqdm(test_data_generator):
    input_ids = input_ids.to(device)
    token_type_ids = token_type_ids.to(device)
    attention_mask = attention_mask.to(device)

    with torch.no_grad():
        y_pred = model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )

    test_pred += np.argmax(y_pred.cpu().detach().numpy(), axis=1).tolist()


with open('Hw3_predict.csv', 'w', newline='') as fout:
    csv_writer = csv.writer(fout)
    csv_writer.writerow(['index', 'answer'])
    for idx, ans in enumerate(test_pred):
        csv_writer.writerow([idx, ans+1])


