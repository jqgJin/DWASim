import torch
from tqdm import tqdm

def load_mat(dataset, meta_path):
    matrix_dir = 'Matrixs/' + dataset + '/'
    mat_list = []
    for i in range(len(meta_path)):
        path_list = meta_path[i].split('_')
        filename = path_list[0] + path_list[1]
        mat = torch.load(matrix_dir + filename + '.pt').cuda()
        path_list.pop(0)
        while (1):
            if len(path_list) < 2:
                break
            filename = path_list[0] + path_list[1]
            mat = mat @ torch.load(matrix_dir + filename + '.pt').cuda()
            path_list.pop(0)
        mat_list.append(mat)
    return mat_list

def getdifsimmat(dataset, meta_path, t):
    mat_list = load_mat(dataset, meta_path)
    for k in range(len(mat_list)):
        mat = mat_list[k]
        # if 'Data' in dataset:
        #     # 若是连续数据集，则归一化
        #     mat = mat
        dif_mat = torch.zeros((mat.shape[0], mat.shape[0]), device='cuda')
        Max_non_zero = -1
        for i in tqdm(range(mat.shape[0])):
            for j in range(i + 1, mat.shape[0]):
                if len(torch.where(torch.abs(mat[i] - mat[j]) != 0)[0]) > Max_non_zero:
                    Max_non_zero = len(torch.where(torch.abs(mat[i] - mat[j]) != 0)[0])
        for i in tqdm(range(mat.shape[0])):
            max = torch.max(mat[i])
            min = torch.min(mat[i])
            for j in range(i + 1, mat.shape[0]):
                if torch.max(mat[j]) > max:
                    max = torch.max(mat[j])
                if torch.min(mat[j]) < min:
                    min = torch.min(mat[j])
                dif_mat[i][j] = (t * len(torch.where(torch.abs(mat[i] - mat[j]) != 0)[0]) + (1 - t) * torch.sum(torch.abs(mat[i] - mat[j]))) / (t * Max_non_zero + (1 - t) * Max_non_zero * (max - min))
                dif_mat[j][i] = dif_mat[i][j]
        dif_mat = dif_mat / torch.max(dif_mat)
        dif_mat = 1 - dif_mat
        torch.save(dif_mat, 'Results/' + dataset + '/dif_simmat_' + meta_path[k] + '.pt')
    return

# dataset_dict = {0: ('ACM', ['0_1_0', '0_2_0', '0_3_0']), 1: ('DBLP', ['0_1_0', '0_1_2_1_0', '0_1_3_1_0']),
#                 2: ('Data1', ['1_0_1', '1_2_1']), 3: ('Data4', ['0_5_0', '0_6_0'])}
# for i in range(len(dataset_dict)):
#     if i == 0 or i == 1:
#         continue
#     dataset = dataset_dict[i][0]
#     meta_path = dataset_dict[i][1]
#     getdifsimmat(dataset, meta_path, 0.5)
