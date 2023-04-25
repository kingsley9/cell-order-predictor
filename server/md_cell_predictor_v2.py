import glob
import numpy as np
import pandas as pd
import os
import re
import gc
import time
import tqdm
from transformers import AutoTokenizer
import torch
from torch.utils.data import DataLoader
from models import MarkdownModel
from utils import *
from utils import Logger, get_model_path

import pandas as pd
import torch
from torch.utils.data.dataset import Dataset
# from utils import create_label
import numpy as np


os.environ['TOKENIZERS_PARALLELISM'] = 'false'

log_dir = 'inference_log.txt'
if os.path.exists(log_dir):
    os.remove(log_dir)

# Parameter class
log = Logger()
log.open(log_dir, mode='a')


class Parameter(object):
    def __init__(self):
        # data
        self.result_dir = './user_data/'
        self.data_dir = 'input/AI4Code/'
        self.k_folds = 5
        self.n_jobs = 4
        self.random_seed = 27
        self.seq_length = 512
        self.cell_count = 128
        self.cell_max_length = 128
        self.device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')
        # model
        self.use_cuda = torch.cuda.is_available()
        self.gpu = 0
        self.print_freq = 100
        self.lr = 0.003
        self.weight_decay = 0
        self.optim = 'Adam'
        self.base_epoch = 30

    def get(self, name):
        return getattr(self, name)

    def set(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __str__(self):
        return '\n'.join(['%s:%s' % item for item in self.__dict__.items()])


# Init parameters
parameter = Parameter()
parameter.set(**{'batch_size': 2, 'n_jobs': 2})
seed_everything(parameter.random_seed)
# Dataset


class MarkdownDatasetV2(Dataset):

    def __init__(self, meta_data: pd.DataFrame, tokenizer, parameter=None, max_length=4096):
        self.meta_data = meta_data.copy()
        self.meta_data.reset_index(drop=True, inplace=True)
        if tokenizer.sep_token != '[SEP]':
            self.meta_data['source'] = self.meta_data['source'].apply(
                lambda x: [
                    y.replace(tokenizer.sep_token, '').replace(
                        tokenizer.cls_token, '').replace(tokenizer.pad_token, '')
                    for y in x])
        self.batch_max_length = self.meta_data['batch_max_length'].values
        self.source = self.meta_data['source'].values
        self.parameter = parameter
        self.max_length = max_length
        self.cell_type = self.meta_data['cell_type'].values
        # self.cell_id = self.meta_data['cell_id'].values
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        source = self.source[index]
        cell_type = self.cell_type[index]
        batch_max_len = min(self.batch_max_length[index], self.max_length)

        cell_inputs = self.tokenizer.batch_encode_plus(
            source,
            add_special_tokens=False,
            max_length=self.parameter.cell_max_length,
            # padding="max_length",
            return_attention_mask=False,
            truncation=True,
        )
        seq, seq_mask, target_mask = self.max_length_rule_base(
            cell_inputs['input_ids'], cell_type, batch_max_len)
        return seq, seq_mask, target_mask

    def __len__(self):
        return len(self.meta_data)

    def max_length_rule_base(self, cell_inputs, cell_type, batch_max_len):
        init_length = [len(x) for x in cell_inputs]
        total_max_length = batch_max_len - len(init_length)
        min_length = total_max_length // len(init_length)
        cell_length = self.search_length(
            init_length, min_length, total_max_length, len(init_length))
        # print(init_code_length,code_length)

        seq = []
        for i in range(len(cell_length)):
            if cell_type[i] == 0:
                seq.append(self.tokenizer.cls_token_id)
            else:
                seq.append(self.tokenizer.sep_token_id)

            if cell_length[i] > 0:
                seq.extend(cell_inputs[i][:cell_length[i]])

        # print(len(seq),'1111', np.sum(init_length),np.sum(cell_length))
        if len(seq) < batch_max_len:
            seq_mask = [1] * len(seq) + [0] * (batch_max_len - len(seq))
            seq = seq + [self.tokenizer.pad_token_id] * \
                (batch_max_len - len(seq))
        else:
            seq_mask = [1] * batch_max_len
            seq = seq[:batch_max_len]
        seq, seq_mask = np.array(seq, dtype=int), np.array(
            seq_mask, dtype=int)
        target_mask = np.where((seq == self.tokenizer.cls_token_id) | (
            seq == self.tokenizer.sep_token_id), 1, 0)
        return seq, seq_mask, target_mask

    @staticmethod
    def search_length(init_length, min_length, total_max_length, cell_count, step=4, max_search_count=50):
        if np.sum(init_length) <= total_max_length:
            return init_length

        res = [min(init_length[i], min_length) for i in range(cell_count)]
        for s_i in range(max_search_count):
            tmp = [min(init_length[i], res[i] + step)
                   for i in range(cell_count)]
            if np.sum(tmp) < total_max_length:
                res = tmp
            else:
                break
        for s_i in range(cell_count):
            tmp = [i for i in res]
            tmp[s_i] = min(init_length[s_i], res[s_i] + step)
            if np.sum(tmp) < total_max_length:
                res = tmp
            else:
                break
        return res


def read_json_data(mode='train'):
    paths_train = sorted(
        list(glob.glob(parameter.data_dir + '{}/*.json'.format(mode))))  # [:100]
    res = pd.concat([
        pd.read_json(path, dtype={'cell_type': 'category', 'source': 'str'}).assign(
            id=path.split('/')[-1].split('.')[0]).rename_axis('cell_id')
        for path in tqdm(paths_train)]).reset_index(drop=False)
    res = res[['id', 'cell_id', 'cell_type', 'source']]
    return res


def preprocess_df(df):
    df['cell_count'] = df.groupby(by=['id'])['cell_id'].transform('count')
    # df['source'] = df['cell_type'] + ' ' + df['source']
    df['cell_type'] = df['cell_type'].map(
        {'code': 0, 'markdown': 1}).fillna(0).astype(int)
    # df.loc[df['cell_type']==0, 'source'] = df.loc[df['cell_type']==0, 'rank'] + ' ' + df.loc[df['cell_type']==0, 'source']
    df['markdown_count'] = df.groupby(by=['id'])['cell_type'].transform('sum')
    df['code_count'] = df['cell_count'] - df['markdown_count']
    df['rank'] = df['rank'] / df['cell_count']
    df['source'] = df['source'].apply(lambda x: x.lower().strip())
    df['source'] = df['source'].apply(lambda x: preprocess_text(x))

    df['source'] = df['source'].str.replace("[SEP]", "")
    df['source'] = df['source'].str.replace("[CLS]", "")

    df['source'] = df['source'].apply(lambda x: re.sub(' +', ' ', x))
    return df


# from https://www.kaggle.com/code/ilyaryabov/fastttext-sorting-with-cosine-distance-algo

# stemmer = WordNetLemmatizer()


def preprocess_text(document):
    # Remove all the special characters
    document = re.sub(r'\W', ' ', str(document))
    document = document.replace('_', ' ')

    # remove all single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

    document = re.sub(r'\s+', ' ', document, flags=re.I)

    document = document.lower()

    return document


def get_truncated_df(df, cell_count=128, id_col='id2', group_col='id', max_random_cnt=100, expand_ratio=5):
    tmp1 = df[df['cell_count'] <= cell_count].reset_index(drop=True)
    tmp1.loc[:, id_col] = 1
    tmp2 = df[df['cell_count'] > cell_count].reset_index(drop=True)
    # print(tmp1.shape,tmp2.shape)
    res = [tmp1]
    for _, df_g in tmp2.groupby(by=group_col):
        # print(df_g.columns)
        df_g = df_g.sample(frac=1.0).reset_index(drop=True)
        step = min(cell_count // 2, len(df_g) - cell_count)
        step = max(step, 1)
        id_col_count = 1
        for i in range(0, len(df_g), step):
            res_tmp = df_g.iloc[i:i + cell_count]  # .copy()
            if len(res_tmp) != cell_count:
                res_tmp = df_g.iloc[-cell_count:]
            # if len(res_tmp) == cell_count:
            res_tmp.loc[:, id_col] = id_col_count
            id_col_count += 1
            res.append(res_tmp)
            if i + cell_count >= len(df_g):
                break

        if len(df_g) // cell_count > 1.3:
            random_cnt = int(len(df_g) // cell_count * expand_ratio)
            random_cnt = min(random_cnt, max_random_cnt)  # todo

            for i in range(random_cnt):
                res_tmp = df_g.sample(n=cell_count).reset_index(drop=True)
                res_tmp.loc[:, id_col] = id_col_count
                id_col_count += 1
                res.append(res_tmp)

    res = pd.concat(res).reset_index(drop=True)
    res = res.sort_values(
        by=['id', id_col, 'cell_type', 'rank2'], ascending=True)
    res = res.groupby(by=['id', id_col, 'fold_flag', 'cell_count', 'markdown_count', 'code_count'], as_index=False, sort=False)[
        ['cell_id', 'cell_type', 'source', 'rank', 'rank2']].agg(list)
    return res


def get_preds(my_df, my_loader, my_model, model_path, max_length=4096):
    if my_df.shape[0] > 0:
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        my_model.load_state_dict(checkpoint['state_dict'])
    if torch.cuda.is_available():
        my_model = my_model.cuda()
    with torch.no_grad():
        y_pred, mask = predict(my_model, my_loader, max_length)
    return y_pred, mask


def predict(model, data_loader, max_length):
    # switch to evaluate mode
    model.eval()
    y_pred = []
    mask = []
    for i, batch_data in enumerate(data_loader):
        if torch.cuda.is_available():
            batch_data = (t.cuda() for t in batch_data)
        seq, seq_mask, target_mask = batch_data
        outputs = model(seq, seq_mask).detach().cpu().numpy()
        target_mask = target_mask.detach().cpu(
        ).numpy().reshape((outputs.shape[0], -1))
        tmp1 = np.zeros((outputs.shape[0], max_length))
        tmp1[:, :outputs.shape[1]] = outputs
        tmp2 = np.zeros((outputs.shape[0], max_length))
        tmp2[:, :outputs.shape[1]] = target_mask

        y_pred.append(tmp1)
        mask.append(tmp2)

    y_pred = np.concatenate(y_pred)
    mask = np.concatenate(mask)
    return y_pred, mask


def get_results(df, masks, rank_pred, code_df_valid):
    df['cell_id2'] = df['cell_id']
    df = df[['id', 'cell_id2']].explode('cell_id2')
    df = df[~pd.isnull(df['cell_id2'])]
    preds = rank_pred.flatten()[np.where(masks.flatten() == 1)]
    df['rank2'] = preds
    df = df.groupby(by=['id', 'cell_id2'], as_index=False)['rank2'].agg('mean')

    df.rename(columns={'cell_id2': 'cell_id'}, inplace=True)
    code_df_valid_tmp = code_df_valid[code_df_valid['id'].isin(df['id'])]
    code_df_valid_tmp['rank3'] = code_df_valid_tmp.groupby(
        by=['id'])['rank2'].rank(ascending=True, method='first')
    tmp = code_df_valid_tmp[['id', 'cell_id', 'rank3']].merge(
        df, how='inner', on=['id', 'cell_id'])
    tmp['rank4'] = tmp.groupby(by=['id'])['rank2'].rank(
        ascending=True, method='first')
    tmp = tmp[['id', 'cell_id', 'rank3']].merge(tmp[['id', 'rank4', 'rank2']].rename(
        columns={'rank4': 'rank3'}), how='inner', on=['id', 'rank3'])
    tmp = tmp[['id', 'cell_id', 'rank2']]

    df = df.merge(tmp[['id', 'cell_id', 'rank2']].rename(
        columns={'rank2': 'rank3'}), how='left', on=['id', 'cell_id'])
    df['rank2'] = np.where(pd.isnull(df['rank3']), df['rank2'], df['rank3'])

    # df = pd.concat([df[['id', 'cell_id', 'rank2']], code_df_valid_tmp]).reset_index(drop=True)
    df = df.sort_values(by=['id', 'rank2'], ascending=True)
    return df


def get_results2(df, masks, rank_pred, code_df_valid):
    df = df[['id', 'id2', 'cell_id']].explode('cell_id')
    df = df[~pd.isnull(df['cell_id'])]
    preds = rank_pred.flatten()[np.where(masks.flatten() == 1)]
    df['rank2'] = preds

    df = df.sort_values(by=['id', 'id2', 'rank2'], ascending=True)
    return df


def predict_df(input_df):
    log.write('>> reading input df\n')
    test_df = pd.concat([input_df])
    test_df['rank'], test_df['fold_flag'] = 1, -1
    test_df = preprocess_df(test_df)

    test_df = pd.concat(
        [test_df[test_df['cell_type'] == 0], test_df[test_df['cell_type'] == 1].sample(frac=1.0)]).reset_index(
        drop=True)
    test_df['rank2'] = (test_df.groupby(by=['id', 'cell_type']).cumcount() + 1) / \
        test_df.groupby(by=['id', 'cell_type'])['cell_id'].transform('count')
    test_df.loc[test_df['cell_type'] == 1, 'rank2'] = -1
    code_df_sub = test_df[test_df['cell_type']
                          == 0][['id', 'cell_id', 'rank2']].copy()

    # test_df2 = test_df[test_df['cell_count']>=96]
    test_df = get_truncated_df(test_df, cell_count=parameter.cell_count)

    log.write('>> predicting...\n')
    start = time.time()
    # --------------------
    model_name = 'deberta-v3-large'
    tokenizer_path = get_model_path(model_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    sort_df, length_sorted_idx = get_sorted_test_df(
        test_df, 'source', tokenizer, batch_size=parameter.batch_size, cell_max_length=parameter.cell_max_length)
    del test_df
    gc.collect()

    sort_df1 = sort_df[sort_df['batch_max_length'] <= 4096]
    sort_df2 = sort_df[sort_df['batch_max_length'] > 4096]

    # -------------------- part1
    test_dataset = MarkdownDatasetV2(
        sort_df1, tokenizer, parameter=parameter, max_length=4096)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=2,
                             num_workers=parameter.n_jobs, drop_last=False, pin_memory=True)
    model = MarkdownModel(get_model_path(model_name), pretrained=False)
    model_path = 'input/ai4code-model/deberta-v3-large_fold0.pth.tar'
    y_preds, masks = get_preds(sort_df1, test_loader,
                               model, model_path, max_length=4096+1024)
    del model
    gc.collect()
    torch.cuda.empty_cache()

    if len(sort_df2) > 0:
        # -------------------- part2
        test_dataset = MarkdownDatasetV2(
            sort_df2, tokenizer, parameter=parameter, max_length=4096+1024)
        test_loader = DataLoader(test_dataset, shuffle=False, batch_size=1,
                                 num_workers=parameter.n_jobs, drop_last=False, pin_memory=True)
        model = MarkdownModel(get_model_path(model_name), pretrained=False)
        model_path = 'input/ai4code-model/deberta-v3-large_fold0.pth.tar'
        y_preds2, masks2 = get_preds(
            sort_df2, test_loader, model, model_path, max_length=4096+1024)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        y_preds = np.concatenate([y_preds2, y_preds])
        masks = np.concatenate([masks2, masks])

    res = get_results(sort_df, masks, y_preds, code_df_sub)

    sub_df = res.groupby(by=['id'], sort=False)['cell_id'].apply(
        lambda x: ' '.join(x)).reset_index()
    sub_df.rename(columns={'cell_id': 'cell_order'}, inplace=True)

    return sub_df['cell_order']
