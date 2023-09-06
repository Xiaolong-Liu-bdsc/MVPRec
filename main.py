import numpy as np
from sklearn.utils import shuffle
import torch
import random
from models.Predictor import HeteroDotProductPredictor
from torch.utils.data import WeightedRandomSampler
import torch.nn.functional as F
import pandas as pd
from collections import defaultdict
import itertools
import logging
from models.model import our_model
from utils.util import Helper
from utils.dataset import GBdataset
import pdb
from torch.utils.data import TensorDataset, DataLoader
import torch as t
import dgl
import scipy.sparse as sp
import argparse
from xmlrpc.client import boolean
import torch.optim as optim
import os
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default = 'new_BeiBei', type = str,
                        help = 'Dataset to be used. (BeiBei or BeiDian)')
    parser.add_argument('--seed', default = 1008, type = int,
                        help = 'Random Seed')
    parser.add_argument('--n_layers', default = 1, type = int,
                        help = 'number of layers')
    parser.add_argument('--embed_size', default = 64, type = int,
                        help = 'embedding size for all layer')
    parser.add_argument('--lr', default = 0.01, type = float,
                        help = 'learning rate') 
    parser.add_argument('--epochs', default = 300, type = int,
                        help = 'epoch number')
    parser.add_argument('--early_stop', default = 10, type = int,
                        help = 'early_stop validation')
    parser.add_argument('--batch_size', default = 512, type = int,
                        help = 'batch size')     
    parser.add_argument('--cuda', default = 0, type = int,
                        help = 'cuda')
    parser.add_argument('--weight_decay', default = 1e-4, type = float,
                        help = 'weight_decay')
    parser.add_argument('--loss_reg', default = 0.1, type = float,
                        help = 'loss_reg')
    parser.add_argument('--head_num', default = 2, type = int,
                        help = 'head_num')
    args = parser.parse_args()
    return args

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
def eval(model, dataset, K, social_K):

    test_loader = dataset.test_loader
    max_k = max(K)
    max_social_k = max(social_K)
    metric_result,social_metric_result  = {},{}
    for k in K:
        metric_result[k] = [[],[]]  #0:Recall, 1:ndcg
    for k in social_K:
        social_metric_result[k] = [[],[]]
    for batch_id, [user, item, participant] in tqdm(enumerate(test_loader)):
        bprloss, prtc_loss, prtc_scores, consistent_loss  = model(user, item, participant)
        prediction = torch.topk(prtc_scores,max_social_k)[1]
        social_rating_K = torch.gather(model.friends[user], 1, prediction)
        # prediction = torch.matmul(query,torch.transpose(ua_embeddings_participant,0,1))
            
        for k in social_K:
            #social
            truth = participant.unsqueeze(-1)
            result = truth[:,:k] == social_rating_K[:,:k]

            recall = t.sum(result).item()

            dcg = t.sum(result/(t.log2(t.arange(2,k+2))),1)
            
            ndcg = t.sum(dcg).item()

            social_metric_result[k][0].append(recall)
            
            social_metric_result[k][1].append(ndcg)
    final_result,final_social_result = {},{}
    test_user_num = len(dataset.U_I_U_test)
    
    for k in social_K:
        final_social_result[k] = {}
        final_social_result[k]['recall'] = sum(social_metric_result[k][0])/test_user_num
        final_social_result[k]['ndcg'] = sum(social_metric_result[k][1])/test_user_num
    return final_social_result
# def eval(model, dataset, K, social_K):

#     test_loader = dataset.test_loader
#     max_k = max(K)
#     max_social_k = max(social_K)
#     metric_result,social_metric_result  = {},{}
#     item_embed_sharer,item_embed_participant, ua_embeddings_sharer, ua_embeddings_participant = model.fuse_views()
#     item_concat = torch.concat([item_embed_sharer, item_embed_participant], dim = 1)
#     ea_embeddings = torch.matmul(item_concat, model.linear_item)
#     for k in K:
#         metric_result[k] = [[],[]]  #0:Recall, 1:ndcg
#     for k in social_K:
#         social_metric_result[k] = [[],[]]
#     for batch_id, [user, item, participant] in tqdm(enumerate(test_loader)):
#         bprloss, prtc_loss, prtc_scores, consistent_loss  = model(user, item, participant)
#         prediction = torch.topk(prtc_scores,max_social_k)[1]
#         social_rating_K = torch.gather(model.friends[user], 1, prediction)
#         sharer_embeddings = ua_embeddings_sharer[user]

#         # bprloss, prtc_loss, prtc_scores = model(user, item, participant)
        
#         prediction = torch.mm(sharer_embeddings, torch.t(ea_embeddings))
#         # prediction = torch.matmul(query,torch.transpose(ua_embeddings_participant,0,1))
#         rating_K = torch.topk(prediction,max_k)[1]

#         for k in K:
#             truth = item.unsqueeze(-1)
#             result = truth[:,:k] == rating_K[:,:k]

#             recall = t.sum(result).item()

#             dcg = t.sum(result/(t.log2(t.arange(2,k+2))),1)
            
#             ndcg = t.sum(dcg).item()

#             metric_result[k][0].append(recall)

#             metric_result[k][1].append(ndcg)
            
#         for k in social_K:
#             #social
#             truth = participant.unsqueeze(-1)
#             result = truth[:,:k] == social_rating_K[:,:k]

#             recall = t.sum(result).item()

#             dcg = t.sum(result/(t.log2(t.arange(2,k+2))),1)
            
#             ndcg = t.sum(dcg).item()

#             social_metric_result[k][0].append(recall)
            
#             social_metric_result[k][1].append(ndcg)
#     final_result,final_social_result = {},{}
#     test_user_num = len(dataset.U_I_U_test)
    
#     for k in K:
#         final_result[k] = {}
#         final_result[k]['recall'] = sum(metric_result[k][0])/test_user_num
#         final_result[k]['ndcg'] = sum(metric_result[k][1])/test_user_num
#     for k in social_K:
#         final_social_result[k] = {}
#         final_social_result[k]['recall'] = sum(social_metric_result[k][0])/test_user_num
#         final_social_result[k]['ndcg'] = sum(social_metric_result[k][1])/test_user_num
#     return [final_result,final_social_result]

def train(model):
    stop_count = 0
    best_val_loss = 99999999
    val_length = len(dataset.val_loader)
    for e in range(args.epochs):
        print('Epoch: ', e)
        model.train()
        for batch_id, [user, item, participant] in enumerate(dataset.train_loader):
            
            # bprloss, prtc_loss, prtc_scores = model(user, item, participant)
            bprloss, prtc_loss, prtc_scores, consistent_loss = model(user, item, participant)
            # pdb.set_trace()

            # loss = prtc_loss
            loss = 1.0 * prtc_loss + args.loss_reg * consistent_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        val_loss = 0

        for batch_id, [user, item, participant] in enumerate(dataset.val_loader):
            bprloss, prtc_loss, prtc_scores, consistent_loss  = model(user, item, participant)
            loss = 1.0 * prtc_loss + args.loss_reg * consistent_loss
            loss = loss.detach()
            val_loss += loss
        print('val_loss: ', val_loss/val_length)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            stop_count = 0
            torch.save(model, 'models/saved_model/our/' + args.data + '_' + str(args.lr)+'_'+str(args.embed_size) +'_'+str(args.weight_decay)+'_'+str(args.n_layers) +'_'+str(args.loss_reg) +'_'+str(args.head_num) +  '.pt')
        else:
            stop_count += 1
            if stop_count > args.early_stop:
                break

    model_final = torch.load('models/saved_model/our/' + args.data + '_' + str(args.lr)+'_'+str(args.embed_size) +'_'+str(args.weight_decay)+'_'+str(args.n_layers) +'_'+str(args.loss_reg) +'_'+str(args.head_num) + '.pt')
    model_final.cpu()
    
    model_final.eval()
    result = eval(model_final, dataset, K,social_K)
    # os.remove('models/saved_model/our/' + args.data + '_' + str(args.lr)+'_'+str(args.embed_size) +'_'+str(args.weight_decay) +  '.pt')
    return result

if __name__ =='__main__':
    
    args = parse_args()
    print(args)
    setup_seed(args.seed)

    device = torch.device('cuda:'+str(args.cuda) if torch.cuda.is_available() else 'cpu')
    dataset = GBdataset(args)
    # pdb.set_trace()
    helper = Helper()
    K = [5,10,15,20]
    social_K = [1,2,3]
    model = our_model(args, dataset)
    model = model.to(device)
    opt = t.optim.Adam(model.parameters(), lr = args.lr, weight_decay=args.weight_decay)
    result = train(model)
    print(result)
    # for i in result:
    #     print(i)