import torch
import random
import pandas as pd
from tqdm import tqdm
from collections import Counter

def get_labels(dataset):
    dir_label = 'Datasets/' + dataset + '/label.dat'
    labels = pd.read_csv(dir_label, sep="\t", header=None)
    node_list = []
    label_dict = {}
    for i in range(len(labels)):
        node_list.append(labels.iloc[i, 0])
        label_dict[labels.iloc[i, 0]] = labels.iloc[i, 3]
    return node_list, label_dict

def label_prediction_difsim(dataset, meta_path, idxs, node_list, label_dict):
    dir_out = 'Results/' + dataset + '/label_prediction.txt'
    for k in range(len(meta_path)):
        dir_mat = 'Results/' + dataset + '/dif_simmat_' + meta_path[k] + '.pt'
        sim_mat = torch.load(dir_mat)
        correct_count = 0
        for i in tqdm(range(len(idxs))):
            label_list = []
            idx = idxs[i]
            top_10_indices = torch.topk(sim_mat[idx], k=10, largest=True)[1]
            for j in range(len(top_10_indices)):
                if top_10_indices[j] in node_list:
                    label_list.append(label_dict[top_10_indices[j].item()])
            if len(label_list):
                count = Counter(label_list)
                most_count = max(count.keys(), key=count.get)
                if most_count == label_dict[idx]:
                    correct_count = correct_count + 1
        accuracy_rate = correct_count / len(idxs)
        with open(dir_out, 'a') as f:
            f.write(dataset + '\t' + "DifSim" + '\t' + meta_path[k] + '\n')
            f.write("accuracy_rate: " + str(accuracy_rate) + '\n')
            f.write('\n')
            f.write('\n')

def label_prediction_metapath(dataset, baseline, meta_path, idxs, node_list, label_dict):
    dir_out = 'Results/' + dataset + '/label_prediction.txt'
    for k in range(len(meta_path)):
        dir_mat = 'Results/' + dataset + '/' + baseline + '_' + meta_path[k] + '.pt'
        sim_mat = torch.load(dir_mat)
        correct_count = 0
        for i in tqdm(range(len(idxs))):
        # for i in range(len(idxs)):
            label_list = []
            idx = idxs[i]
            top_10_indices = torch.topk(sim_mat[idx], k=10, largest=True)[1]
            for j in range(len(top_10_indices)):
                if top_10_indices[j] in node_list:
                    label_list.append(label_dict[top_10_indices[j].item()])
            if len(label_list):
                count = Counter(label_list)
                # print(label_list)
                # print(count)
                most_count = max(count.keys(), key=count.get)
                if most_count == label_dict[idx]:
                    correct_count = correct_count + 1
        accuracy_rate = correct_count / len(idxs)
        with open(dir_out, 'a') as f:
            f.write(dataset + '\t' + baseline + '\t' + meta_path[k] + '\n')
            f.write("accuracy_rate: " + str(accuracy_rate) + '\n')
            f.write('\n')
            f.write('\n')

def label_prediction(dataset, meta_path, times):
    node_list, label_dict = get_labels(dataset)
    idxs = []
    while(len(idxs) < times):
        idx = node_list[random.randint(0, len(node_list) - 1)]
        if idx not in idxs:
            idxs.append(idx)
    label_prediction_difsim(dataset, meta_path, idxs, node_list, label_dict)
    label_prediction_metapath(dataset, 'pathsim', meta_path, idxs, node_list, label_dict)
    label_prediction_metapath(dataset, 'hetesim', meta_path, idxs, node_list, label_dict)
    


dataset_dict = {0: ('ACM', ['0_1_0', '0_2_0', '0_3_0']), 1: ('DBLP', ['0_1_0', '0_1_2_1_0', '0_1_3_1_0']),}
times = 500
for i in range(len(dataset_dict)):
    dataset = dataset_dict[i][0]
    meta_path = dataset_dict[i][1]
    label_prediction(dataset, meta_path, times)