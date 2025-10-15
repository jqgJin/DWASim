import torch
from tqdm import tqdm

def get_U(dir_name, filename):
    W = torch.load(dir_name + filename + '.pt', map_location=lambda storage, loc: storage.cuda(0))
    U = torch.zeros((W.shape[0], W.shape[1]), device='cuda')
    for i in range(U.shape[0]):
        if torch.sum(W[i]) != 0:
            U[i] = W[i] / torch.sum(W[i])
        else:
            U[i] = W[i]
    return U

def decomposit(dir_name, filename):
    W = torch.load(dir_name + filename + '.pt', map_location=lambda storage, loc: storage.cuda(0))
    W_L = torch.zeros((W.shape[0], int(torch.sum(W).item())), device='cuda')
    W_R = torch.zeros((int(torch.sum(W).item()), W.shape[1]), device='cuda')
    sum_L = torch.sum(W, dim=1)
    where_R = torch.where(W == 1)[1]
    x = 0
    y = 0
    for i in range(len(sum_L)):
        step = sum_L[i].item()
        while(step):
            W_L[x][y] = 1
            y = y + 1
            step = step - 1
        x = x + 1
    x = 0
    for i in range(len(where_R)):
        y = where_R[i].item()
        W_R[x][y] = 1
        x = x + 1
    if torch.equal(W_L @ W_R, W):
        U_L = torch.zeros((W_L.shape[0], W_L.shape[1]), device='cuda')
        U_R = torch.zeros((W_R.shape[0], W_R.shape[1]), device='cuda')
        for i in range(U_L.shape[0]):
            if torch.sum(W_L[i]) != 0:
                U_L[i] = W_L[i] / torch.sum(W_L[i])
            else:
                U_L[i] = W_L[i]
        for i in range(U_R.shape[1]):
            if torch.sum(W_R[i]) != 0:
                U_R[i] = W_R[i] / torch.sum(W_R[i])
            else:
                U_R[i] = W_R[i]
        U_R = U_R.T
    else:
        print('Error!')
        return
    return U_L, U_R

def cal_sim(U_L, U_R, start_shape, end_shape):
    Sim = torch.zeros((start_shape, end_shape), device='cuda')
    for i in tqdm(range(Sim.shape[0])):
        for j in range(i, Sim.shape[1]):
            if i == j:
                Sim[i][j] = 1
                continue
            s = torch.cosine_similarity(U_L[i], U_R[j], dim=0)
            Sim[i][j] = s
            Sim[j][i] = s
    return Sim

def hetesim(dataset, meta_path):
    dir_name = 'Matrixs/' + dataset + '/'
    for i in range(len(meta_path)):
        print('Processing ' + meta_path[i])
        path_list = meta_path[i].split('_')
        path_length = len(path_list)
        if not path_length % 2:
            path_list.insert(int(path_length / 2), 'E')
        if path_list[1] == 'E':
            filename = path_list[0] + path_list[2]
            U_L, U_R = decomposit(dir_name, filename)
            start_shape = U_L.shape[0]
            end_shape = U_R.shape[0]
        else:
            filename1 = path_list[0] + path_list[1]
            filename2 = path_list[-1] + path_list[-2]
            U_L = get_U(dir_name, filename1)
            U_R = get_U(dir_name, filename2)
            start_shape = U_L.shape[0]
            end_shape = U_R.shape[0]
        path_list.pop(0)
        path_list.pop(-1)
        while(1):
            if len(path_list) == 1:
                break
            if path_list[1] == 'E':
                filename = path_list[0] + path_list[2]
                U_LL, U_RR = decomposit(dir_name, filename)
                U_L = U_L @ U_LL
                U_R = U_R @ U_RR
            else:
                filename1 = path_list[0] + path_list[1]
                filename2 = path_list[-1] + path_list[-2]
                U_L = U_L @ get_U(dir_name, filename1)
                U_R = U_R @ get_U(dir_name, filename2)
            path_list.pop(0)
            path_list.pop(-1)
        sim_mat = cal_sim(U_L, U_R, start_shape, end_shape)
        print(meta_path[i] + ' finished!')
        torch.save(sim_mat, 'Results/' + dataset + '/hetesim_' +  meta_path[i] + '.pt')
        # print(sim_mat)


# dataset_dict = {0: ('Data1_1', ['1_0_1', '1_2_1', '1_5_1']), 1: ('Data4_1', ['0_5_0', '0_6_0', '0_7_0']),
#                 2: ('Data1_2', ['1_0_1', '1_2_1', '1_5_1']), 3: ('Data4_2', ['0_5_0', '0_6_0', '0_7_0'])}
# for i in range(len(dataset_dict)):
#     dataset = dataset_dict[i][0]
#     meta_path = dataset_dict[i][1]
#     hetesim(dataset, meta_path)