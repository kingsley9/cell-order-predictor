# your_code.py

from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import sys
import numpy as np
from transformers import AutoModel, AutoTokenizer
import torch.nn as nn
import torch

from torch.utils.data import DataLoader
BERT_PATH = "./input/deberta-v3-base/"
BS = 32
NW = 8
# Add your imports for MarkdownDataset, read_notebook, and other necessary functions here

MAX_LEN = 128


class MarkdownModel(nn.Module):
    def __init__(self):
        super(MarkdownModel, self).__init__()
        self.bert = AutoModel.from_pretrained(BERT_PATH)
        self.top = nn.Linear(768, 1)

    def forward(self, ids, mask):
        x = self.bert(ids, mask)[0]
        x = self.top(x[:, 0, :])
        return x


class MarkdownDataset(Dataset):

    def __init__(self, df, max_len):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained(
            BERT_PATH, do_lower_case=True, use_fast=True)


    def __getitem__(self, index):
        row = self.df.iloc[index]

        inputs = self.tokenizer.encode_plus(
            row.source,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            return_token_type_ids=True,
            truncation=True
        )
        ids = torch.LongTensor(inputs['input_ids'])
        mask = torch.LongTensor(inputs['attention_mask'])

        return ids, mask, torch.FloatTensor([row.pct_rank])

    def __len__(self):
        return self.df.shape[0]


def read_data(data):
    return tuple(d.to(torch.device('cpu')) for d in data[:-1]), data[-1].to(torch.device('cpu'))


def validate(model, val_loader):
    model.eval()

    tbar = tqdm(val_loader, file=sys.stdout)

    preds = []
    labels = []

    with torch.no_grad():
        for idx, data in enumerate(tbar):
            inputs, target = read_data(data)

            pred = model(inputs[0], inputs[1])

            preds.append(pred.detach().cpu().numpy().ravel())
            labels.append(target.detach().cpu().numpy().ravel())

    return np.concatenate(labels), np.concatenate(preds)


def predict(model, input_df):
    test_ds = MarkdownDataset(
        input_df[input_df["cell_type"] == "markdown"].reset_index(drop=True), max_len=MAX_LEN)
    test_loader = DataLoader(test_ds, batch_size=BS, shuffle=False, num_workers=NW,
                             pin_memory=False, drop_last=False)

    _, y_test = validate(model, test_loader)
    input_df.loc[input_df["cell_type"] == "markdown", "pred"] = y_test
    sub_df = input_df.sort_values("pred").groupby(
        "id")["cell_id"].apply(lambda x: " ".join(x)).reset_index()
    sub_df.rename(columns={"cell_id": "cell_order"}, inplace=True)

    return sub_df["cell_order"]
