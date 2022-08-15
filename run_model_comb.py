import transformers
from sklearn.cluster import KMeans
from sklearn import svm
import numpy as np
from sklearn.metrics import classification_report
import torch
import matplotlib.pyplot as plt
import itertools
from torch.nn import CrossEntropyLoss

import dataloaders
from run_classification import get_dataset_feature

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_data_id = 'ADReSS20-train-transcript'
test_data_id = 'ADReSS20-test-transcript'


def run_model_comb(model_name_list):
    train_dataset = [data for data in dataloaders.load_dataset_easy(train_data_id)]
    test_dataset = [data for data in dataloaders.load_dataset_easy(test_data_id)]

    for model_name in model_name_list:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        text_model = transformers.AutoModel.from_pretrained(model_name, return_dict=True)
        text_model.eval()
        text_model.to(device)

        train_dataset = get_dataset_feature(train_dataset, tokenizer, text_model)
        test_dataset = get_dataset_feature(test_dataset, tokenizer, text_model)

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
    print(classification_report(train_label, train_label_predict))
    print(classification_report(1 - train_label, train_label_predict))

    test_label_predict = clf.predict(test_feature)
    print(classification_report(test_label, test_label_predict))
    print(classification_report(1 - test_label, test_label_predict))
    report_dict_a = classification_report(test_label, test_label_predict, digits=4, output_dict=True)
    report_dict_b = classification_report(1 - test_label, test_label_predict, digits=4, output_dict=True)
    report_dict = report_dict_a if report_dict_a['accuracy'] > report_dict_b['accuracy'] else report_dict_b

    return report_dict


def main():
    model_name_list = ['../glue/checkpoints_ppl/None/3/wnli', '../glue/checkpoints_ppl/None/3/sst2',
                       '../glue/checkpoints_ppl/None/3/rte']
    model_name_list = ['../glue/checkpoints_ppl/None/3/rte', '../glue/checkpoints_ppl/None/3/cola',
                       '../glue/checkpoints_ppl/None/3/mrpc']
    model_report_dict_list = []

    report_dict = run_model_comb(model_name_list)
    report_dict['model_name'] = model_name_list
    model_report_dict_list.append(report_dict)

    accuracy_list = []
    for report_dict in model_report_dict_list:
        print(report_dict['model_name'], report_dict['accuracy'], sep='\t')
        accuracy_list.append(report_dict['accuracy'])


if __name__ == '__main__':
    main()
