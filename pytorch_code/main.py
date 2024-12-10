#!/usr/bin/env python36
# -*- coding: utf-8 -*-

import argparse
import pickle
import time
from utils import build_graph, Data, split_validation
from model import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='diginetica', help='dataset name: diginetica/yoochoose1_4/yoochoose1_64/sample')
parser.add_argument('--batchSize', type=int, default=50, help='input batch size')
parser.add_argument('--hiddenSize', type=int, default=120, help='hidden state size')
parser.add_argument('--epoch', type=int, default=30, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  # [0.001, 0.0005, 0.0001]
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
parser.add_argument('--step', type=int, default=1, help='gnn propogation steps')
parser.add_argument('--patience', type=int, default=10, help='the number of epoch to wait before early stop ')
parser.add_argument('--nonhybrid', action='store_true', help='only use the global preference to predict')
parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion of training set as validation set')
parser.add_argument('--dynamic', type=bool, default=False)
parser.add_argument('--last_k', type=int, default=7)
parser.add_argument('--heads', type=int, default=8)
parser.add_argument('--scale', default=True, help='scaling factor sigma')
parser.add_argument('--dot', default=True, action='store_true')
parser.add_argument('--l_p', type=int, default=4, help='Lipschitz norm for attention pooling')
parser.add_argument('--use_attn_conv', type=str, default="True", help='Whether to use attention convolution')

opt = parser.parse_args()
print(opt)

def main():
    train_data = pickle.load(open('/root/GC-SAN_master/datasets/' + opt.dataset + '/train.txt', 'rb'))
    if opt.validation:
        train_data, valid_data = split_validation(train_data, opt.valid_portion)
        test_data = valid_data
    else:
        test_data = pickle.load(open('../datasets/' + opt.dataset + '/test.txt', 'rb'))

    # Initialize the Data objects for both training and test datasets
    train_data = Data(train_data, shuffle=True, opt=opt)
    test_data = Data(test_data, shuffle=False, opt=opt)

    # Set the number of nodes (items) based on the dataset
    if opt.dataset == 'diginetica':
        n_node = 43098
    else:
        n_node = 310

    model = trans_to_cuda(SessionGraph(opt, n_node, max(train_data.len_max, test_data.len_max)))

    start = time.time()
    best_result = [0, 0]
    best_epoch = [0, 0]
    bad_counter = 0
    # Training loop over the specified number of epochs
    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        print(f'epoch:  {epoch}/{opt.epoch-1}')
        precision_at_k_mean, mrr_mean = train_test(model, train_data, test_data)
        flag = 0 # Flag to indicate if we got a new best result
        if precision_at_k_mean >= best_result[0]:
            best_result[0] = precision_at_k_mean
            best_epoch[0] = epoch
            flag = 1
        if mrr_mean >= best_result[1]:
            best_result[1] = mrr_mean
            best_epoch[1] = epoch
            flag = 1
        print('Best Result:')
        print(f'\tPrecision@{K}:\t{best_result[0]:.4f}\tMMR@{K}:\t{best_result[1]:.4f}\tEpoch:\t{best_epoch[0]},{best_epoch[1]}')
        bad_counter += 1 - flag
        if bad_counter >= opt.patience:
            break
    print('-------------------------------------------------------')
    end = time.time()
    print("Run time: %f s" % (end - start))

if __name__ == '__main__':
    main()