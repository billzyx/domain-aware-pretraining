import transformers
import torch
import numpy as np
import re
from torch.nn import CrossEntropyLoss

import dataloaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

icu_list = []
with open('icu.txt', 'r') as f:
    icu_list = f.read().splitlines()
print(icu_list)


class DummyPPL:
    def ppl_score(self, bert_model, tokenizer):
        return 0


class PPL:
    def __init__(self, ppl_dataset):
        self.ppl_method = 'pseudo'
        if '-icu' in ppl_dataset:
            ppl_dataset = ppl_dataset.replace('-icu', '')
            self.ppl_method = 'icu'
        self.diff = False
        self.r_diff = False
        if '-HC' in ppl_dataset:
            ppl_dataset = ppl_dataset.replace('-HC', '')
            train_dataset = [data for data in dataloaders.load_dataset_easy(ppl_dataset) if data['label'] == 0]
        elif '-AD' in ppl_dataset:
            ppl_dataset = ppl_dataset.replace('-AD', '')
            train_dataset = [data for data in dataloaders.load_dataset_easy(ppl_dataset) if data['label'] == 1]
        elif '-diff' in ppl_dataset:
            ppl_dataset = ppl_dataset.replace('-diff', '')
            self.diff = True
            train_dataset = [data for data in dataloaders.load_dataset_easy(ppl_dataset)]
        elif '-rdiff' in ppl_dataset:
            ppl_dataset = ppl_dataset.replace('-rdiff', '')
            self.diff = True
            self.r_diff = True
            train_dataset = [data for data in dataloaders.load_dataset_easy(ppl_dataset)]
        else:
            train_dataset = [data for data in dataloaders.load_dataset_easy(ppl_dataset)]

        if not self.diff:
            self.text_list = []
            for idx, data in enumerate(train_dataset):
                self.text_list.append(data['text'])
        else:
            self.text_list_hc = []
            self.text_list_ad = []
            for idx, data in enumerate(train_dataset):
                if data['label'] == 0:
                    self.text_list_hc.append(data['text'])
                else:
                    self.text_list_ad.append(data['text'])

    def ppl_score(self, bert_model, tokenizer):
        if not self.diff:
            return cal_ppl_score(bert_model, tokenizer, self.text_list, self.ppl_method)
        else:
            if self.r_diff:
                return cal_ppl_score(bert_model, tokenizer, self.text_list_ad, self.ppl_method) - \
                       cal_ppl_score(bert_model, tokenizer, self.text_list_hc, self.ppl_method)
            else:
                return - cal_ppl_score(bert_model, tokenizer, self.text_list_ad, self.ppl_method) + \
                       cal_ppl_score(bert_model, tokenizer, self.text_list_hc, self.ppl_method)


def cal_ppl_score(bert_model, tokenizer, text_list, method='pseudo'):
    ppl_score_list = []
    for text in text_list:
        if method == 'pseudo':
            ppl_score = score_pseudo(bert_model, tokenizer, text)
        elif method == 'icu':
            ppl_score = score_icu(bert_model, tokenizer, text)
        else:
            raise ValueError('method should be pseudo or icu')
        ppl_score_list.append(ppl_score)
    return np.exp(np.mean(ppl_score_list))


def score_pseudo(model, tokenizer, sentence):
    batch_size = 32
    tensor_input = tokenizer.encode(sentence, return_tensors='pt', truncation=True, max_length=512).to(device)
    repeat_input = tensor_input.repeat(tensor_input.size(-1) - 2, 1)
    mask = torch.ones(tensor_input.size(-1) - 1).diag(1)[:-2].to(device)
    masked_input = repeat_input.masked_fill(mask == 1, tokenizer.mask_token_id)
    labels = repeat_input.masked_fill(masked_input != tokenizer.mask_token_id, -100)
    loss_list = []
    loss_fct = CrossEntropyLoss(reduce=False)
    model.to(device)
    with torch.inference_mode():
        for i in range(0, masked_input.size(0), batch_size):
            masked_input_batch = masked_input[i:i + batch_size]
            labels_batch = labels[i:i + batch_size]
            prediction_scores = model(masked_input_batch, labels=labels_batch).logits
            loss = loss_fct(prediction_scores.view(-1, model.config.vocab_size), labels_batch.view(-1))
            loss_list.append(loss)
    loss_list = torch.cat(loss_list)
    loss = loss_list[labels.view(-1) != -100].mean()
    # loss = model(tensor_input, labels=tensor_input.clone()).loss
    return loss.item()


def get_token_length(tokenizer):
    token_length_list = []
    for icu in icu_list:
        token_code = tokenizer.encode(icu)
        token_length_list.append(len(token_code) - 2)
    return token_length_list


def replace_nth(string, sub, wanted, n):
    where = [m.start() for m in re.finditer(sub, string)][n - 1]
    before = string[:where]
    after = string[where:]
    after = after.replace(sub, wanted, 1)
    new_string = before + after
    return new_string


def score_icu(model, tokenizer, sentence):
    sentence = sentence.lower()
    sentence_list = []
    label_list = []
    token_length_list = get_token_length(tokenizer)
    for icu, token_length in zip(icu_list, token_length_list):
        occurrences_count = ' {} '.format(sentence).count(' {} '.format(icu))
        for i in range(occurrences_count):
            sentence_list.append(replace_nth(' {} '.format(sentence), ' {} '.format(icu), ' [MASK] ' * token_length, i))
            label_list.append(sentence)
    if len(sentence_list) == 0:
        return 6.
    masked_input = tokenizer(sentence_list, return_tensors='pt', truncation=True, max_length=512, padding=True).to(
        device)
    labels = tokenizer(label_list, return_tensors='pt', truncation=True, max_length=512, padding=True).to(device)
    labels.input_ids = labels.input_ids.masked_fill(masked_input.input_ids != tokenizer.mask_token_id, -100)
    model.to(device)
    with torch.inference_mode():
        loss = model(**masked_input, labels=labels.input_ids).loss
    # loss = model(tensor_input, labels=tensor_input.clone()).loss
    return loss.item()


def main():
    model_name = 'bert-base-uncased'
    bert_model = transformers.AutoModelForMaskedLM.from_pretrained(model_name, return_dict=True).to(device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    ppl = PPL('ADReSS20-train-level-2-HC')
    ppl_hc = ppl.ppl_score(bert_model, tokenizer)
    ppl = PPL('ADReSS20-train-level-2-AD')
    ppl_ad = ppl.ppl_score(bert_model, tokenizer)

    print(ppl_hc, ppl_ad)

    dataset_hc = [data for data in dataloaders.load_dataset_easy('ADReSS20-train-level-2') if data['label'] == 0]
    dataset_ad = [data for data in dataloaders.load_dataset_easy('ADReSS20-train-level-2') if data['label'] == 1]

    for data in dataset_hc:
        ppl_score = score_pseudo(bert_model, tokenizer, data['text'])
        print(data['file_idx'], np.exp(ppl_score))

    print()

    for data in dataset_ad:
        ppl_score = score_pseudo(bert_model, tokenizer, data['text'])
        print(data['file_idx'], np.exp(ppl_score))


if __name__ == '__main__':
    main()
