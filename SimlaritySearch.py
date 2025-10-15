import torch

def get_difsim_top_10(dataset, idx, meta_path):
    dir_out = 'Results/' + dataset + '/difsim_top_10.txt'
    for k in range(len(meta_path)):
        dir_name = 'Results/' + dataset + '/dif_simmat_' + meta_path[k] + '.pt'
        sim_mat = torch.load(dir_name)
        top_10 = torch.topk(sim_mat[idx], k=10, largest=True)
        values = top_10[0]
        indices = top_10[1]
        with open(dir_out, 'a') as f:
            f.write(meta_path[k] + '\n')
            f.write("Idx:" + str(idx) + '\n')
            f.write("Similarity Scores of Top-10 Lists" + '\n')
            for i in range(len(values)):
                f.write(str(values[i].item()) + '\n')
            f.write("Top-10 Similarity Search Results" + '\n')
            for i in range(len(indices)):
                f.write(str(indices[i].item()) + '\n')
            f.write('\n')

def get_metapath_top_10(baseline, dataset, idx, meta_path):
    dir_out = 'Results/' + dataset + '/' + baseline + '_top_10.txt'
    for k in range(len(meta_path)):
        dir_name = 'Results/' + dataset + '/' + baseline + '_' + meta_path[k] + '.pt'
        sim_mat = torch.load(dir_name)
        top_10 = torch.topk(sim_mat[idx], k=10, largest=True)
        values = top_10[0]
        indices = top_10[1]
        with open(dir_out, 'a') as f:
            f.write(meta_path[k] + '\n')
            f.write("Idx:" + str(idx) + '\n')
            f.write("Similarity Scores of Top-10 Lists" + '\n')
            for i in range(len(values)):
                f.write(str(values[i].item()) + '\n')
            f.write("Top-10 Similarity Search Results" + '\n')
            for i in range(len(indices)):
                f.write(str(indices[i].item()) + '\n')
            f.write('\n')
            f.write("-----------------------------------------" + '\n')

dataset_dict = {0: ('ACM', ['0_1_0', '0_2_0', '0_3_0']), 1: ('DBLP', ['0_1_0', '0_1_2_1_0', '0_1_3_1_0']),
                2: ('Data1', ['1_0_1', '1_2_1']), 3: ('Data4', ['0_5_0', '0_6_0'])}
for i in range(len(dataset_dict)):
    if i == 0 or i == 1:
        continue
    dataset = dataset_dict[i][0]
    meta_path = dataset_dict[i][1]
    if i == 0:
        idx = 377
    elif i == 1:
        idx = 1015
    else:
        idx = 0
    get_difsim_top_10(dataset, idx, meta_path)
    get_metapath_top_10('pathsim', dataset, idx, meta_path)
    get_metapath_top_10('hetesim', dataset, idx, meta_path)