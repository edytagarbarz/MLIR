from torch.optim import Adam
from transformers import BertTokenizer, BertModel
import pandas as pd
import numpy as np
import torch as tc

import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

if tc.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

import transformers
transformers.logging.set_verbosity_error()


class BertClassifier(nn.Module):

    def __init__(self, N_CLASSES, bert, dropout=0.1):

        super(BertClassifier, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, N_CLASSES)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, lang, input_ids, attention_mask):

        sentence_output, pooled_output = self.bert(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=False)[:2]
        output = sentence_output[:, 0, :]
        dropout_output = self.dropout(output)
        linear_output = self.linear(dropout_output)
        return linear_output


class TrainDataset(Dataset):
    def __init__(self, tokenizer, qrels, text_path, queries):
        self.qrels = qrels
        self.text_path = text_path
        self.queries = queries
        self.tokenizer = tokenizer

        self.positive = qrels[qrels['label'] == 1].reset_index(drop=True)
        self.negative = qrels[qrels['label'] == 0].reset_index(drop=True)

        print("More positive then negative:", len(
            self.positive) > len(self.negative))
        assert len(self.positive) > len(self.negative)

    def __len__(self):
        return len(self.positive)

    def __getitem__(self, idx):
        if idx == 0:
            self.positive = self.positive.sample(frac=1).reset_index(drop=True)
            self.negative = self.negative.sample(frac=1).reset_index(drop=True)

        positive = self.positive.iloc[idx]
        qid = positive['qid']
        qid_negative = self.negative[self.negative['qid'] == qid]
        negative = qid_negative.sample(1).iloc[0]

        def read(docno):
            p = self.text_path + docno + '.txt'

            with open(p) as f:
                text = f.read()
                return text
        p_text = read(positive['docno'])
        n_text = read(negative['docno'])
        texts = [p_text, n_text]

        q_text = [queries[qid]] * 2
        item = self.tokenizer(q_text, texts, truncation=True, padding='max_length',
                              max_length=512, return_tensors="pt").to(device)

        return item


class TestDataset(Dataset):
    def __init__(self, tokenizer, qrels, text_path, queries):
        self.qrels = qrels
        self.text_path = text_path
        self.queries = queries
        self.tokenizer = tokenizer
        self.docnos = qrels['docno'].tolist()
        self.qids = qrels['qid'].tolist()

    def __len__(self):
        return len(self.qrels)

    def __getitem__(self, idx):

        sample = self.qrels.iloc[idx]

        def read(docno):
            docno = docno+'.txt'

            p = self.text_path + docno

            with open(p) as f:
                text = f.read()
                return text
        text = read(sample['docno'])
        qid = sample['qid']
        q_text = queries[qid]
        p_item = self.tokenizer(q_text, text, truncation=True, padding='max_length',
                                max_length=512, return_tensors="pt").to(device)

        return p_item


def eval_results(scores, qrels):
    prec_at_10 = []
    prec = []
    for qid in np.unique(scores['qid']):
        positive = qrels[(qrels['qid'] == qid) & (
            qrels['label'] == 1)]['docno'].tolist()
        subscores = scores[scores['qid'] == qid]
        subscores = subscores.sort_values(by='score', ascending=False)
        docnos = subscores['docno'].tolist()
        at10, p = 0, 0
        for i, d in enumerate(docnos):
            if d in positive:
                if i < 10:
                    at10 += 1
                if i < 50:
                    p += 1
                else:
                    break
        prec_at_10.append(at10/10)
        prec.append(p/50)
    print(
        f'Precision at 10: {np.mean(prec_at_10): .3f} | Precision at 50: {np.mean(prec): .3f}')


def eval(model, dataset):
    with tc.no_grad():
        scores = []
        test_dataloader = DataLoader(
            test_dataset, batch_size=BATCH, shuffle=False)
        for input_data in tqdm(test_dataloader):
            output = model(input_ids=input_data['input_ids'].squeeze(
                1), attention_mask=input_data['attention_mask'].squeeze(1))
            output_scores = output.detach().cpu().tolist()
            scores += output_scores
        sim_scores = pd.DataFrame(data={"docno": dataset.docnos[:len(
            dataset)], "qid": dataset.qids[:len(dataset)], "score":  scores})

        eval_results(sim_scores, dataset.qrels)


def train(model, train_data, val_data, learning_rate, epochs):

    criterion = nn.HingeEmbeddingLoss().to(device)
    softmax = nn.Softmax(dim=1)
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)

    def encode_sample(train_input, lang):

        output = model(lang, input_ids=train_input['input_ids'].squeeze(
            0), attention_mask=train_input['attention_mask'].squeeze(0))
        return output
    for epoch_num in range(epochs):

        total_loss_train = 0

        for sample, lang in tqdm(train_data):

            output = encode_sample(sample, lang[0])
            output = output.reshape(-1, 2)

            target = tc.tensor([-1] * output.shape[0]).to(device)

            joined = softmax(output)
            diff = joined[0][0] - joined[0][1]

            batch_loss = criterion(diff, target)
            total_loss_train += batch_loss.item()

            model.zero_grad()
            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        total_loss_val = 0

        tc.save({
            'epoch': epoch_num,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, root + 'models/checkpoints_align/checkpoint' + str(epoch_num) + '.pt')
        print(
            f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} \
                | Val Loss: {total_loss_val / len(val_data): .3f}')
        eval(model, test_dataset)


root = ""
qrels_train = ""
topics_train = ""
text_path = ""


align_tokenizer = BertTokenizer.from_pretrained(
    root + 'bert-base-m-cased_align')
align_model = BertModel.from_pretrained(
    root + "bert-base-m-cased_align").to(device)

topics_train['query'] = topics_train.apply(
    lambda x: x['title'] + '. ' + x['description'], axis=1)
tuples = topics_train.apply(lambda x: (
    str(x['qid']), x['query']), axis=1).tolist()
queries = dict(tuples)

train = pd.read_csv(root + 'qrels/train_qrels.csv', dtype={'qid': str})
test = pd.read_csv(root + 'qrels/test_qrels.csv', dtype={'qid': str})

train_dataset = TrainDataset(align_tokenizer, train, text_path, queries)
test_dataset = TestDataset(align_tokenizer, test, text_path, queries)

EPOCHS = 10
N_CLASSES = 1
BATCH = 1
LR = 1e-5

model = BertClassifier(N_CLASSES, align_model).to(device)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True)

eval(model, test_dataset)
train(model, train_dataloader, test_dataset, LR, EPOCHS)

tc.save(model.state_dict(), root + 'models/aligned_sequence_classification.pt')
