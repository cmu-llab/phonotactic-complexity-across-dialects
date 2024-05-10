# Source: Pimentel et al 2020 https://github.com/tpimentelms/phonotactic-complexity/blob/master/learn_layer/train_base.py
import numpy as np
import pickle
import math
import csv
from tqdm import tqdm
import io

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, TensorDataset, DataLoader

import sys
sys.path.append('./')
from model.lstm import IpaLM
from model.syllable_const_lstm import SyllableConstituentLM
import argparse
import pathlib

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

results_per_word = [['lang', 'concept_id', 'phoneme_id', 'phoneme', 'phoneme_len', 'phoneme_loss']]
results_per_position = [['lang'] + list(range(30))]
results_per_position_per_word = \
    [['lang', 'concept_id', 'phoneme_id', 'phoneme', 'phoneme_len', 'phoneme_loss'] +
     list(range(30))]

def get_data_loaders(dataset, lang, syllabify):
    train_loader = get_data_loader(dataset, lang, 'train', syllabify)
    val_loader = get_data_loader(dataset, lang, 'val', syllabify)
    test_loader = get_data_loader(dataset, lang, 'test', syllabify)

    return train_loader, val_loader, test_loader


def get_data_loader(dataset, lang, mode, syllabify):
    if syllabify:
        data = read_data(dataset, lang, mode)
        syllable_constituent_data = read_data(dataset, lang, mode, syllabify)
        return convert_to_loader(data, mode, syllable_constituent_data) # reads both
    else:
        data = read_data(dataset, lang, mode)
        return convert_to_loader(data, mode)


def read_data(dataset, lang, mode, syllabify=False):
    with open(f"data/{dataset}/preprocess/data{'-syllabified' if syllabify else ''}-{lang}-{mode}.npy", 'rb') as f:
        data = np.load(f)

    return data


def write_csv(results, filename):
    with io.open(filename, 'w', encoding='utf8') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerows(results)


def read_info(dataset):
    with open(f"data/{dataset}/preprocess/info.pckl", 'rb') as f:
        info = pickle.load(f)
    languages = info['languages']
    token_map = info['token_map']
    data_split = info['data_split']
    concept_ids = info['concepts_ids']
    ipa_to_concept = info['IPA_to_concept']

    return languages, token_map, data_split, concept_ids, ipa_to_concept


def convert_to_loader(data, mode, syllable_constituent_data=None, batch_size=64):
    x_phonemes = torch.from_numpy(data[:, :-2]).long().to(device=device)
    y = torch.from_numpy(data[:, 1:-1]).long().to(device=device)
    idx = torch.from_numpy(data[:, -1]).long().to(device=device)

    shuffle = True if mode == 'train' else False

    if syllable_constituent_data is not None:
        x_syl_const = torch.from_numpy(syllable_constituent_data[:, :-2]).long().to(device=device)
        y_syl_const = torch.from_numpy(syllable_constituent_data[:, 1:-1]).long().to(device=device)
        # note that each language was already padded with the longest word in that language's data, so no collate_fn is needed
        dataset = TensorDataset(x_phonemes, x_syl_const, y, y_syl_const, idx)
    else:
        dataset = TensorDataset(x_phonemes, y, idx)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

class MultiTaskLoss(torch.nn.Module):
  '''https://arxiv.org/abs/1705.07115'''
  def __init__(self, is_regression, reduction='none'):
    super(MultiTaskLoss, self).__init__()
    self.is_regression = is_regression
    self.n_tasks = len(is_regression)
    self.log_vars = torch.nn.Parameter(torch.zeros(self.n_tasks))
    self.reduction = reduction

  def forward(self, losses):
    dtype = losses.dtype
    device = losses.device
    stds = (torch.exp(self.log_vars)**(1/2)).to(device).to(dtype)
    self.is_regression = self.is_regression.to(device).to(dtype)
    coeffs = 1 / ( (self.is_regression+1)*(stds**2) )
    multi_task_losses = coeffs*losses + torch.log(stds)

    if self.reduction == 'sum':
      multi_task_losses = multi_task_losses.sum()
    if self.reduction == 'mean':
      multi_task_losses = multi_task_losses.mean()

    return multi_task_losses

def train_epoch(train_loader, model, loss, optimizer, syllabify):
    model.train()
    total_loss = 0.0
    for batches, data in enumerate(train_loader):
        optimizer.zero_grad()

        if syllabify:
            (batch_x_phonemes, batch_x_syl_const, batch_y, syl_batch_y, _) = data
            phon_y_hat, _, const_y_hat, _ = model(batch_x_phonemes, batch_x_syl_const)
        else:
            (batch_x_phonemes, batch_y, _) = data
            phon_y_hat, _ = model(batch_x_phonemes)
        
        if syllabify: 
            multitaskloss_instance = MultiTaskLoss(torch.tensor([False, False]), reduction='sum')
            phon_l = loss(phon_y_hat.view(-1, phon_y_hat.size(-1)), batch_y.view(-1)) / math.log(2) 
            const_l = loss(const_y_hat.view(-1, const_y_hat.size(-1)), syl_batch_y.view(-1)) / math.log(2)
            # during training, backprop both losses
            # only use the phone loss for calculating the correlation though
            losses = torch.stack((phon_l, const_l))
            multitaskloss = multitaskloss_instance(losses)
            l = multitaskloss
        else:
            l = loss(phon_y_hat.view(-1, phon_y_hat.size(-1)), batch_y.view(-1)) / math.log(2)

        l.backward()
        optimizer.step()

        total_loss += l.item()

    if syllabify:
        return total_loss / (batches + 1), phon_l.item(), const_l.item(), l.item()
    else:
        return total_loss / (batches + 1)


def eval(data_loader, model, loss, syllabify):
    model.eval()
    val_loss, val_acc, total_sent = 0.0, 0.0, 0
    for batches, data in enumerate(data_loader):
        if syllabify:
            (batch_x_phonemes, batch_x_syl_const, batch_y, syl_batch_y, _) = data
            phon_y_hat, _, const_y_hat, _ = model(batch_x_phonemes, batch_x_syl_const)
        else:
            (batch_x_phonemes, batch_y, _) = data
            y_hat, _ = model(batch_x_phonemes)
        
        if syllabify:
            multitaskloss_instance = MultiTaskLoss(torch.tensor([False, False]), reduction='sum')
            phon_l = loss(phon_y_hat.view(-1, phon_y_hat.size(-1)), batch_y.view(-1)) / math.log(2)
            const_l = loss(const_y_hat.view(-1, const_y_hat.size(-1)), syl_batch_y.view(-1)) / math.log(2)
            losses = torch.stack((phon_l, const_l))
            multitaskloss = multitaskloss_instance(losses)
            l = multitaskloss
        else:
            l = loss(y_hat.view(-1, y_hat.size(-1)), batch_y.view(-1)) / math.log(2)
        val_loss += l.item() * batch_y.size(0)

        non_pad = batch_y != 0
        if syllabify:
            val_acc += (phon_y_hat.argmax(-1)[non_pad] == batch_y[non_pad]).float().mean().item() * batch_y.size(0)
        else:
            val_acc += (y_hat.argmax(-1)[non_pad] == batch_y[non_pad]).float().mean().item() * batch_y.size(0)

        total_sent += batch_y.size(0)

    val_loss = val_loss / total_sent
    val_acc = val_acc / total_sent

    return val_loss, val_acc


def run_model(model, batch_x):
    return model(batch_x)


def eval_per_word(lang, data_loader, model, token_map, ipa_to_concept, model_name, args, model_func=run_model):
    global results_per_word, results_per_position, results_per_position_per_word
    model.eval()
    token_map_inv = {x: k for k, x in token_map.items()}
    ignored_tokens = [token_map['PAD'], token_map['SOW'], token_map['EOW']]
    loss = nn.CrossEntropyLoss(ignore_index=0, reduction='none').to(device=device)
    val_loss, val_acc, total_sent = 0.0, 0.0, 0
    loss_per_position, count_per_position = None, None

    for batches, data in enumerate(data_loader):
        if args.syllabify:
            (batch_x_phonemes, batch_x_syl_const, batch_y, syl_batch_y, batch_idx) = data
            phon_y_hat, _, const_y_hat, _ = model(batch_x_phonemes, batch_x_syl_const)
        else:
            (batch_x_phonemes, batch_y, batch_idx) = data
            y_hat, _ = model_func(model, batch_x_phonemes)

        if args.syllabify:
            phon_l = loss(phon_y_hat.view(-1, phon_y_hat.size(-1)), batch_y.view(-1)).reshape_as(batch_y).detach() / math.log(2)
            # const_l = loss(const_y_hat.view(-1, const_y_hat.size(-1)), syl_batch_y.view(-1)).reshape_as(batch_y).detach() / math.log(2)

            # important - the entropy comes from the loss
            # while we need to backprop both losses during training, we can only factor the phone loss for phonotactic complexity
            l = phon_l
        else:
            l = loss(y_hat.view(-1, y_hat.size(-1)), batch_y.view(-1)).reshape_as(batch_y).detach() / math.log(2)
        loss_per_position = loss_per_position + l.sum(0).data if loss_per_position is not None else l.sum(0).data
        count_per_position = count_per_position + (l != 0).sum(0).data if count_per_position is not None \
            else (l != 0).sum(0).data
        words = torch.cat([batch_x_phonemes, batch_y[:, -1:]], -1).detach()

        words_ent = l.sum(-1)
        words_len = (batch_y != 0).sum(-1)

        words_ent_avg = words_ent / words_len.float()
        val_loss += words_ent_avg.sum().item()

        non_pad = batch_y != 0
        if args.syllabify:
            val_acc += (phon_y_hat.argmax(-1)[non_pad] == batch_y[non_pad]).float().mean().item() * batch_y.size(0)
        else:
            val_acc += (y_hat.argmax(-1)[non_pad] == batch_y[non_pad]).float().mean().item() * batch_y.size(0)

        total_sent += batch_y.size(0)

        for i, w in enumerate(words):
            _w = idx_to_word(w, token_map_inv, ignored_tokens)
            idx = batch_idx[i].item()
            results_per_word += [[lang, ipa_to_concept[idx], idx, _w, words_len[i].item(), words_ent_avg[i].item()]]
            results_per_position_per_word += [[
                lang, ipa_to_concept[idx], idx, _w, words_len[i].item(),
                words_ent_avg[i].item()] + l[i].float().cpu().numpy().tolist()]

    results_per_position += [[lang] + list((loss_per_position / count_per_position.float()).cpu().numpy())]
    val_loss = val_loss / total_sent
    val_acc = val_acc / total_sent

    write_csv(results_per_position, '%s/%s__results-per-position.csv' % (args.rfolder, model_name))
    write_csv(results_per_position_per_word, '%s/%s__results-per-position-per-word.csv' % (args.rfolder, model_name))
    write_csv(results_per_word, '%s/%s__results-per-word.csv' % (args.rfolder, model_name))

    return val_loss, val_acc, results_per_word


def word_to_tensors(word, token_map):
    w = word_to_idx(word, token_map)

    x = torch.from_numpy(w[:, :-1]).long().to(device=device)
    y = torch.from_numpy(w[:, 1:]).long().to(device=device)
    return x, y


def word_to_idx(word, token_map):
    w = [[token_map['SOW']] + [token_map[x] for x in word] + [token_map['EOW']]]
    return np.array(w)


def idx_to_word(word, token_map_inv, ignored_tokens):
    _w = [token_map_inv[x] for x in word.tolist() if x not in ignored_tokens]
    return ' '.join(_w)


def _idx_to_word(word, token_map, ignored_tokens):
    token_map_inv = {x: k for k, x in token_map.items()}
    return idx_to_word(word, token_map_inv, ignored_tokens)


def train(train_loader, val_loader, test_loader, model, loss, optimizer, syllabify, wait_epochs=50):
    epoch, best_epoch, best_loss, best_acc = 0, 0, float('inf'), 0.0

    pbar = tqdm(total=wait_epochs)
    phon_losses, const_losses, losses = [], [], []
    while True:
        epoch += 1
        if syllabify:
            total_loss, phon_loss, const_loss, l = train_epoch(train_loader, model, loss, optimizer, syllabify)
            phon_losses.append(phon_loss)
            const_losses.append(const_loss)
            losses.append(l)
        else:
            total_loss = train_epoch(train_loader, model, loss, optimizer, syllabify)
        val_loss, val_acc = eval(val_loader, model, loss, syllabify)

        if val_loss < best_loss:
            best_epoch = epoch
            best_loss = val_loss
            best_acc = val_acc
            model.set_best()

        pbar.total = best_epoch + wait_epochs
        pbar.update(1)
        pbar.set_description('%d/%d: loss %.4f  val: %.4f  acc: %.4f  best: %.4f  acc: %.4f' %
                             (epoch, best_epoch, total_loss, val_loss, val_acc, best_loss, best_acc))

        if epoch - best_epoch >= wait_epochs:
            break

    pbar.close()
    model.recover_best()

    return best_epoch, best_loss, best_acc


def get_avg_len(data_loader, syllabify):
    total_phon, total_sent = 0.0, 0.0
    for batches, data in enumerate(data_loader):
        if syllabify:
            (batch_x_phonemes, _, batch_y, _, _) = data
        else:
            (batch_x_phonemes, batch_y, _) = data
        
        batch = torch.cat([batch_x_phonemes, batch_y[:, -1:]], dim=-1)
        total_phon += (batch != 0).sum().item()
        total_sent += batch.size(0)

    avg_len = (total_phon * 1.0 / total_sent) - 2  # Remove SOW and EOW tag in every sentence

    return avg_len


def get_avg_shannon_entropy(train_loader, test_loader, token_map, syllabify):
    counts = [0] * len(token_map)
    for batches, data in enumerate(train_loader):
        if syllabify:
            (_, _, batch_y, _, _) = data
        else:
            (_, batch_y, _) = data

        for token, index in token_map.items():
            counts[index] += (batch_y == index).sum().item()

    counts = counts[1:]  # Remove PAD
    total = sum(counts)

    probs = [x * 1.0 / total for x in counts]
    shannon = - sum([x * math.log2(x) if x != 0 else 0 for x in probs])

    return shannon


def init_model(model_name, hidden_size, token_map, embedding_size, nlayers, dropout, syllabify):
    vocab_size = len(token_map)
    if model_name == 'lstm':
        if syllabify:
            model = SyllableConstituentLM(
                vocab_size, hidden_size, embedding_size=embedding_size, nlayers=nlayers, dropout=dropout)
        else:
            model = IpaLM(
                vocab_size, hidden_size, embedding_size=embedding_size, nlayers=nlayers, dropout=dropout)
    else:
        raise ValueError("Model not implemented: %s" % model_name)

    return model.to(device=device)


def get_model_entropy(
        lang, model_name, train_loader, val_loader, test_loader, token_map, ipa_to_concept,
        embedding_size, hidden_size, nlayers, dropout, args, wait_epochs=50, per_word=True):
    model = init_model(model_name, hidden_size, token_map, embedding_size, nlayers, dropout, args.syllabify)

    loss = nn.CrossEntropyLoss(ignore_index=0).to(device=device)
    optimizer = optim.Adam(model.parameters())

    best_epoch, val_loss, val_acc = train(
        train_loader, val_loader, test_loader, model, loss, optimizer, args.syllabify, wait_epochs=wait_epochs)
    if per_word:
        test_loss, test_acc, _ = eval_per_word(lang, test_loader, model, token_map, ipa_to_concept, model_name, args)
    else:
        test_loss, test_acc = eval(test_loader, model, loss)

    return test_loss, test_acc, best_epoch, val_loss, val_acc


def _run_language(
        lang, train_loader, val_loader, test_loader, token_map, ipa_to_concept, args,
        embedding_size=None, hidden_size=256, nlayers=1, dropout=0.2, per_word=True):
    avg_len = get_avg_len(train_loader, args.syllabify)
    shannon = get_avg_shannon_entropy(train_loader, test_loader, token_map, args.syllabify)
    test_shannon = get_avg_shannon_entropy(test_loader, test_loader, token_map, args.syllabify)

    print('Language %s Avg len: %.4f Shanon entropy: %.4f Test shannon: %.4f' % (lang, avg_len, shannon, test_shannon))

    test_loss, test_acc, best_epoch, val_loss, val_acc = get_model_entropy(
        lang, args.model, train_loader, val_loader, test_loader, token_map, ipa_to_concept,
        embedding_size, hidden_size, nlayers, dropout, args, per_word=per_word)
    avg_len = get_avg_len(test_loader, args.syllabify)
    print('Test loss: %.4f  acc: %.4f    Test avg len: %.4f  Shannon: %.4f  Test: %.4f' %
          (test_loss, test_acc, avg_len, shannon, test_shannon))

    return avg_len, shannon, test_shannon, test_loss, test_acc, best_epoch, val_loss, val_acc


def run_language(dataset, lang, token_map, ipa_to_concept, args, embedding_size=None, hidden_size=256, nlayers=1, dropout=0.2):
    train_loader, val_loader, test_loader = get_data_loaders(dataset, lang, args.syllabify)

    return _run_language(lang, train_loader, val_loader, test_loader, token_map, ipa_to_concept,
                         args, embedding_size=embedding_size, hidden_size=hidden_size,
                         nlayers=nlayers, dropout=dropout)


def run_language_enveloper(dataset, lang, token_map, ipa_to_concept, args):
    return run_language(dataset, lang, token_map, ipa_to_concept, args)


def run_languages(args):
    dataset = args.dataset

    languages, token_map, data_split, _, ipa_to_concept = read_info(dataset)
    print('Train %d, Val %d, Test %d' % (len(data_split[0]), len(data_split[1]), len(data_split[2])))

    results = [['lang', 'avg_len', 'shannon', 'test_shannon', 'test_loss',
                'test_acc', 'best_epoch', 'val_loss', 'val_acc']]
    for i, lang in enumerate(languages):
        print()
        print(i, end=' ')

        avg_len, shannon, test_shannon, test_loss, \
            test_acc, best_epoch, val_loss, val_acc = run_language_enveloper(dataset, lang, token_map, ipa_to_concept, args)
        results += [[lang, avg_len, shannon, test_shannon, test_loss, test_acc, best_epoch, val_loss, val_acc]]

        write_csv(results, '%s/%s__results.csv' % (args.rfolder, args.model))
    write_csv(results, '%s/%s__results-final.csv' % (args.rfolder, args.model))



def mkdir(folder):
    pathlib.Path(folder).mkdir(parents=True, exist_ok=True)

def add_argument(*args, **kwargs):
    return parser.add_argument(*args, **kwargs)

def set_defaults(*args, **kwargs):
    return parser.set_defaults(*args, **kwargs)

def get_default(*args, **kwargs):
    return parser.get_default(*args, **kwargs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Phoneme LM')

    # Data
    parser.add_argument('--dataset', type=str, default='dutch_nodiacritics',
                    help='Dataset used. (default: dutch_nodiacritics)')
    parser.add_argument('--data-path', type=str, default='data',
                        help='Path where data is stored.')
    parser.add_argument('--model', type=str, default='lstm',
                        help='lstm (Pimentel et al 2020)')
    parser.add_argument('--syllabify', type=lambda x: (str(x).lower() == 'true'), help='whether or not to supervise the model with syllable structure')


    # Others
    parser.add_argument('--results-path', type=str, default='results',
                        help='Path where results should be stored.')
    parser.add_argument('--seed', type=int, default=7,
                        help='Seed for random algorithms repeatability (default: 7)')

    args = parser.parse_args()
    print("Syllabify!" if args.syllabify else "Do not syllabify")

    args.ffolder = '%s/%s' % (args.data_path, args.dataset)  # Data folder
    args.rfolder_base = '%s/%s' % (args.results_path, args.dataset)   # Results base folder
    args.rfolder = '%s/%s/orig' % (args.rfolder_base, 'syllabified' if args.syllabify else 'non-syllabified')  # Results folder
    pathlib.Path(args.rfolder).mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    assert args.dataset in ['dutch_nodiacritics', 'min', 'northeuralex'], 'this script should only be run with dutch_nodiacritics data'
    
    run_languages(args)
