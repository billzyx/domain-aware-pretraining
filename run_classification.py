import transformers
from sklearn.cluster import KMeans
from sklearn import svm
import numpy as np
from sklearn.metrics import classification_report
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import itertools
from torch.nn import CrossEntropyLoss

import dataloaders
import ppl_comp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_data_id = 'ADReSS20-train-transcript'
test_data_id = 'ADReSS20-test-transcript'
# train_data_id = 'ADReSS20-train-level-3'
# test_data_id = 'ADReSS20-test-level-3'
# train_data_id = 'ADReSSo21-train-level-1'
# test_data_id = 'ADReSSo21-test-level-1'


def get_dataset_feature(dataset, tokenizer, text_model):
    for idx, data in enumerate(dataset):
        # print(data)
        text = data['text']
        inputs = tokenizer([text], padding=True, truncation=True, return_tensors="pt", max_length=512).to(device)
        # outputs = text_model(**inputs).pooler_output
        # dataset[idx]['text_feature'] = outputs.squeeze().detach().numpy()
        outputs = text_model(**inputs).last_hidden_state
        text_feature = np.mean(outputs.squeeze().detach().cpu().numpy(), axis=0)
        dataset[idx].setdefault('text_feature', []).append(text_feature)

    return dataset


def get_dataset_ppl(dataset, model_name_list):
    # text_list_ad = []
    # text_list_hc = []
    # for idx, data in enumerate(dataset):
    #     if data['label'] == 1:
    #         text_list_ad.append(data['text'])
    #     else:
    #         text_list_hc.append(data['text'])
    # model_list = get_model_list_for_ppl(model_name_list)
    # ppl_hc = ppl_score(model_name_list, model_list, text_list_hc)
    # ppl_ad = ppl_score(model_name_list, model_list, text_list_ad)
    # ppl = np.abs(ppl_ad - ppl_hc)

    text_list = []
    for idx, data in enumerate(dataset):
        text_list.append(data['text'])
    model_list = get_model_list_for_ppl(model_name_list)
    ppl = ppl_score(model_name_list, model_list, text_list)
    return ppl


def get_model_list_for_ppl(model_name_list):
    model_list = []
    for model_name in model_name_list:
        text_ppl_model_base = transformers.AutoModelForMaskedLM.from_pretrained('bert-base-uncased', return_dict=True)
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        config = transformers.AutoConfig.from_pretrained(model_name)
        text_ppl_model = transformers.AutoModelForMaskedLM.from_pretrained(
            model_name, return_dict=True, output_hidden_states=True)
        # text_ppl_model = train_model(
        #     text_ppl_model, tokenizer, config,
        #     {'train': dataloaders.load_train_dataset(train_data_id),
        #      'test': dataloaders.load_test_dataset(test_data_id)})
        if not text_ppl_model.cls:
            text_ppl_model.cls = text_ppl_model_base.cls
        text_ppl_model.eval()
        text_ppl_model.to(device)
        model_list.append(text_ppl_model)
    return model_list


def ppl_score(model_name_list, model_list, text_list):
    config = transformers.AutoConfig.from_pretrained(model_name_list[0])
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_list[0])
    encodings = tokenizer("\n\n".join(text_list), padding=True, truncation=True, return_tensors="pt",
                          max_length=512).to(device)

    max_length = 512
    stride = 512

    neg_log_likelihood_list = []
    for model in model_list:
        nlls = []
        for i in range(0, encodings.input_ids.size(1), stride):
            begin_loc = max(i + stride - max_length, 0)
            end_loc = min(i + stride, encodings.input_ids.size(1))
            trg_len = end_loc - i  # may be different from stride on last loop
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)
                prediction_scores = outputs.logits
                loss_fct = CrossEntropyLoss(reduce=False)  # -100 index = padding token
                masked_lm_loss = loss_fct(prediction_scores.view(-1, config.vocab_size), target_ids.view(-1))
                nlls.append(masked_lm_loss)
        neg_log_likelihood_list.append(torch.stack(nlls))

    neg_log_likelihood = neg_log_likelihood_list[0]
    for idx in range(1, len(neg_log_likelihood_list)):
        neg_log_likelihood = torch.minimum(neg_log_likelihood, neg_log_likelihood_list[idx])
    neg_log_likelihood = torch.mean(neg_log_likelihood)
    neg_log_likelihood = neg_log_likelihood * trg_len


    ppl = torch.exp(neg_log_likelihood.sum() / end_loc)
    return ppl.detach().cpu().numpy()


def run_model(model_name):
    train_dataset = [data for data in dataloaders.load_dataset_easy(train_data_id)]
    test_dataset = [data for data in dataloaders.load_dataset_easy(test_data_id)]

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    text_model = transformers.AutoModel.from_pretrained(model_name, return_dict=True)
    text_model.eval()
    text_model.to(device)

    train_dataset = get_dataset_feature(train_dataset, tokenizer, text_model)
    test_dataset = get_dataset_feature(test_dataset, tokenizer, text_model)

    # train_hc_dataset = [data for data in train_dataset if data['label'] == 0]
    # text_model_lm = transformers.AutoModelForMaskedLM.from_pretrained(model_name, return_dict=True)
    text_model_lm = transformers.AutoModelForMaskedLM.from_pretrained('bert-base-uncased', return_dict=True)
    text_model_lm.bert = text_model
    # text_model.to(device)
    ppl = ppl_comp.cal_ppl_score(text_model_lm, tokenizer, [data['text'] for data in train_dataset], method='icu')
    ppl_hc = ppl_comp.cal_ppl_score(text_model_lm, tokenizer, [data['text'] for data in train_dataset if data['label'] == 0], method='icu')
    ppl_ad = ppl_comp.cal_ppl_score(text_model_lm, tokenizer, [data['text'] for data in train_dataset if data['label'] == 1], method='icu')

    train_feature = []
    train_label = []

    for idx, data in enumerate(train_dataset):
        train_feature.append(np.concatenate(data['text_feature'], axis=0))
        train_label.append(data['label'])

    train_feature = np.array(train_feature)
    train_label = np.array(train_label)

    test_feature = []
    test_label = []
    for idx, data in enumerate(test_dataset):
        test_feature.append(np.concatenate(data['text_feature'], axis=0))
        test_label.append(data['label'])

    test_feature = np.array(test_feature)
    test_label = np.array(test_label)

    # clf = KMeans(n_clusters=2, random_state=0).fit(train_feature)
    clf = svm.SVC().fit(train_feature, train_label)

    train_label_predict = clf.predict(train_feature)
    # print(classification_report(train_label, train_label_predict))
    # print(classification_report(1 - train_label, train_label_predict))

    test_label_predict = clf.predict(test_feature)
    # print(classification_report(test_label, test_label_predict))
    # print(classification_report(1 - test_label, test_label_predict))
    report_dict_a = classification_report(test_label, test_label_predict, digits=4, output_dict=True)
    report_dict_b = classification_report(1 - test_label, test_label_predict, digits=4, output_dict=True)
    report_dict = report_dict_a if report_dict_a['accuracy'] > report_dict_b['accuracy'] else report_dict_b
    report_dict['ppl'] = float(ppl)
    report_dict['ppl_hc'] = float(ppl_hc)
    report_dict['ppl_ad'] = float(ppl_ad)
    # print(report_dict)
    return report_dict


def main():
    # model_name = 'bert-base-uncased'
    # model_name = 'distilbert-base-uncased'
    # model_name = 'facebook/bart-large-cnn'
    # model_name = 'sentence-transformers/bert-base-nli-mean-tokens'
    # model_name = 'textattack/bert-base-uncased-RTE'
    # model_name = 'textattack/bert-base-uncased-MNLI'
    # model_name = 'textattack/bert-base-uncased-QQP'
    # model_name = 'textattack/bert-base-uncased-WNLI'
    # model_name = 'textattack/bert-base-uncased-STS-B'
    with open('nlp_models.txt', 'r') as f:
        model_name_list = f.read().splitlines()
    model_report_dict_list = []

    # for model_name in model_name_list:
    #     report_dict = run_model([model_name])
    #     report_dict['model_name'] = model_name
    #     model_report_dict_list.append(report_dict)

    # print(model_name_list_combinations)
    for model_name in model_name_list:
        report_dict = run_model(model_name)
        report_dict['model_name'] = model_name
        model_report_dict_list.append(report_dict)

    accuracy_list = []
    ppl_list = []
    model_name_list = []
    for report_dict in model_report_dict_list:
        print(report_dict['model_name'], report_dict['ppl'], report_dict['ppl_hc'], report_dict['ppl_ad'], report_dict['accuracy'], sep='\t')
        # print(report_dict['model_name'], report_dict['accuracy'], sep='\t')
        accuracy_list.append(report_dict['accuracy'])
        ppl_list.append(report_dict['ppl'])
        model_name_list.append(report_dict['model_name'].split('/')[-1])

    ppl_list = np.array(ppl_list)
    accuracy_list = np.array(accuracy_list)
    # plt.plot(ppl_list, accuracy_list, 'ro')
    colors = cm.rainbow(np.linspace(0, 1, len(ppl_list)))
    plt_point_list = []
    for ppl, accuracy, color in zip(ppl_list, accuracy_list, colors):
        plt_point = plt.scatter(ppl, accuracy, marker='o', color=color)
        plt_point_list.append(plt_point)
    m, b = np.polyfit(ppl_list, accuracy_list, 1)
    accuracy_list_fit = m * ppl_list + b
    print('MSE: ', np.sum((accuracy_list - accuracy_list_fit)**2))
    plt.plot(ppl_list, accuracy_list_fit)
    plt.xlabel('Perplexity')
    plt.ylabel('Accuracy')
    plt.legend(plt_point_list, model_name_list)
    plt.savefig('acc_ppl.pdf')


if __name__ == '__main__':
    main()
