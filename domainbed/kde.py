import torch as torch
import torch.nn as nn
import numpy as np
import time
import argparse


def shape_to_matrix(feature_num, env_list, label_num, max_data, data_len, data, device='cuda'):
    env_num = len(env_list)
    matrix = torch.zeros([env_num, label_num, max_data,
                       feature_num], device=device)
    for env in range(env_num):
        for label in range(label_num):
            matrix[env][label][0:data_len[env, label]
            ] = data[label][env_list[env]]
    return matrix


class opt_kde():
    def __init__(self, env_list, train_env, num_classes, feature_num,data,percent=0.5,
                 sample_size=1000,device='cuda'):
        self.sample_size = sample_size
        self.device = device
        self.envs = env_list
        self.train_env = train_env
        self.envs_num = len(self.envs)
        self.mask = None
        self.percent = percent

        # 准备初始化数据
        data_len = np.zeros(
            (len(env_list), num_classes), dtype=np.int32)
        for i in range(len(env_list)):
            for j in range(num_classes):
                data_len[i][j] = len(data[j][env_list[i]])
        #print('data:',data)
        matrix = shape_to_matrix(feature_num=feature_num, env_list=env_list, label_num=num_classes,
                                 max_data=int(
                                     max([max(w) for w in data_len])), data_len=data_len, data=data,
                                 device=device)

        # 确认参数匹配
        self.feature_num = matrix.shape[3]
        assert self.feature_num == feature_num, "Error when loading feature"
        self.label_num = matrix.shape[1]
        assert self.label_num == num_classes, "Error when dealing with labels"
        self.max_sample = matrix.shape[2]
        assert matrix.shape[0] == len(
            env_list), "length of envs in data does match provided envs"

        self.matrix = matrix
        #print('matrix', self.matrix)

        self.data_len = torch.tensor(data_len, dtype=torch.float32)
        self.data_mask = torch.ones(
            (self.envs_num, self.label_num, self.max_sample), dtype=torch.int32).to(self.device)
        for env in range(self.envs_num):
            for label in range(self.label_num):
                self.data_mask[env, label, data_len[env, label]:] -= 1
        self.len_unsqueeze = self.data_len.unsqueeze(2).to(self.device)

        self.bandwidth = 1.06 * \
                         self.max_sample ** (-1. / (1 + 4)) * \
                         torch.std(matrix, dim=2).mean().clone().detach()
        self.offset = torch.exp(-0.5 / (self.bandwidth ** 2)).to(self.device)
        # self.sample_size = int(sample_size * (torch.max(matrix) - torch.min(matrix)).cpu().item())

        self.batch_len = 1
        self.batch_size = (self.sample_size +
                           self.batch_len - 1) // self.batch_len

        self.params = torch.eye(
            self.feature_num, requires_grad=True).to(device)

    # def normalize(self):  # do normalization in params
    #     self.params = self.params / torch.sqrt(torch.sum(self.params ** 2, dim=0, keepdim=True)).detach().clamp_min_(
    #         1e-3)

    def forward(self, cal_info=False, verbose=False):
        # matmul matrix params, s.t. check the results in this linear combination
        matrix = self.matrix.detach().unsqueeze(dim=-1)
        left, right = torch.min(matrix).cpu(
        ).item(), torch.max(matrix).cpu().item()

        if verbose:
            print("sample message: from %.4f to %.4f, size is %d" %
                  (left, right, self.sample_size))
        delta = (right - left) / self.sample_size
        x_gird = torch.linspace(left, right, self.sample_size).to(self.device)
        divisor = np.sqrt(2 * np.pi) * self.bandwidth
        store_dis = torch.zeros(
            (self.envs_num * self.envs_num, self.label_num, self.feature_num)).to(self.device)
        if cal_info:
            store_info = torch.zeros((
                self.label_num * self.label_num, self.envs_num, self.feature_num
            )).to(self.device)
        reduce_zeros = torch.tensor(
            self.max_sample, dtype=torch.float32).to(self.device)

        index = 0
        train_index = []
        for envi in range(self.envs_num):
            for envj in range(self.envs_num):
                if self.envs[envi] in self.train_env and self.envs[envj] in self.train_env:
                    train_index.append(index)
                index += 1

        timing = 1000 // self.batch_len
        for batch in range(self.batch_size):
            if batch % timing == 0:
                start = time.time()
            points = x_gird[batch *
                            self.batch_len:min((batch + 1) * self.batch_len, self.sample_size)].reshape((1, -1))
            reducer = (torch.sum(torch.pow(self.offset, (matrix - points) ** 2), dim=2) -
                       ((reduce_zeros - self.len_unsqueeze) *
                        torch.pow(self.offset, points ** 2)).unsqueeze(dim=2)
                       ) / self.len_unsqueeze.unsqueeze(dim=3)

            dis_expand = reducer.expand(
                (self.envs_num, self.envs_num, self.label_num, self.feature_num, reducer.shape[-1]))
            store_dis += torch.sum(torch.abs(dis_expand - dis_expand.permute(1, 0, 2, 3, 4)), dim=-1).reshape(
                (-1, self.label_num, self.feature_num)) / divisor
            #print(store_dis)
            if cal_info:
                info_expand = reducer.permute(1, 0, 2, 3).expand(
                    (self.label_num, self.label_num, self.envs_num, self.feature_num, reducer.shape[-1]))
                store_info += torch.sum(torch.abs(info_expand - info_expand.permute(1, 0, 2, 3, 4)), dim=-1).reshape(
                    (-1, self.envs_num, self.feature_num)) / divisor

            if batch % timing == timing - 1 and verbose:
                print("epoch %d, avg time: %f" %
                      ((batch + 1) * self.batch_len, (time.time() - start) / timing / self.batch_len))
                # print("pure cal:" + str(cal_time / timing/self.batch_len))

        test_results = (store_dis * delta / 2).max(dim=0)[0]
        train_results = (store_dis[train_index] * delta / 2).max(dim=0)[0]
        if verbose:
            print("finish forward once.")


        if cal_info:
            # should consider min env s.t. this to feature is exhibit, and select the biggest label pair
            # train_info = (store_info * delta / 2).max(dim=0)[0]
            # return a (1, feature_num) dimension
            train_info = (store_info * delta /
                          2).min(dim=1)[0].max(dim=0)[0].reshape((1, -1))
            return {
                "train_results": train_results,
                "test_results": test_results,
                "train_info": train_info,
                "train_dis": torch.mean(train_results.max(dim=0)[0]),
                "test_dis": torch.mean(test_results.max(dim=0)[0])
            }
        return {
            "train_results": train_results,
            "test_results": test_results,
            "train_info": None,
            "train_dis": torch.mean(train_results.max(dim=0)[0]),
            "test_dis": torch.mean(test_results.max(dim=0)[0])
        }

    def eig_val(self):  # return sorted eig value, to check whether degenerate
        eigs = torch.eig(self.params)
        return np.sort(eigs[0].detach().cpu().numpy()[:, 0])

