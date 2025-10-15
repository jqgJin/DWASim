import torch
from tqdm import tqdm

def calsim(mat):
    sim_mat = torch.zeros((mat.shape[0], mat.shape[0]), device='cuda')
    for j in tqdm(range(mat.shape[0])):
        for k in range(j, mat.shape[0]):
            if j == k:
                sim_mat[j][k] = 1
                continue
            up = 2 * torch.sum(mat[j] * mat[k])
            down = torch.sum(torch.pow(mat[j], 2) + torch.pow(mat[k], 2))
            if down != 0:
                sim = up / down
            else:
                sim = 0
            sim_mat[j][k] = sim
            sim_mat[k][j] = sim
    return sim_mat

def pathsim(dataset, meta_path):
    matrix_dir = 'Matrixs/' + dataset + '/'
    for i in range(len(meta_path)):
        print('Processing ' + meta_path[i])
        path_list = meta_path[i].split('_')
        if len(path_list) == 3:
            filename = path_list[0] + path_list[1]
            mat = torch.load(matrix_dir + filename + '.pt', map_location=lambda storage, loc: storage.cuda(0))
            sim_mat = calsim(mat)
            print(meta_path[i] + ' finished!')
            torch.save(sim_mat, 'Results/' + dataset + '/pathsim_' +  meta_path[i] + '.pt')
            continue
        else:
            length = len(path_list) / 2
            while(1):
                if length < 1:
                    break
                path_list.pop(-1)
                length = length - 1
            filename = path_list[0] + path_list[1]
            mat = torch.load(matrix_dir + filename + '.pt', map_location=lambda storage, loc: storage.cuda(0))
            path_list.pop(0)
        while(1):
            if len(path_list) < 2:
                break
            filename = path_list[0] + path_list[1]
            mat_t = torch.load(matrix_dir + filename + '.pt', map_location=lambda storage, loc: storage.cuda(0))
            path_list.pop(0)
            mat = mat @ mat_t
        sim_mat = calsim(mat)
        print(meta_path[i] + ' finished!')
        torch.save(sim_mat, 'Results/' + dataset + '/pathsim_' +  meta_path[i] + '.pt')


# dataset_dict = {0: ('ACM', ['0_1_0', '0_2_0', '0_3_0']), 1: ('DBLP', ['0_1_0', '0_1_2_1_0', '0_1_3_1_0']),
#                 2: ('Data1', ['1_0_1', '1_2_1']), 3: ('Data4', ['0_5_0', '0_6_0'])}
# for i in range(len(dataset_dict)):
#     if i == 0 or i == 1:
#         continue
#     dataset = dataset_dict[i][0]
#     meta_path = dataset_dict[i][1]
#     pathsim(dataset, meta_path)