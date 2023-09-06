from torch.autograd import Variable
import numpy as np
import math
import heapq
import torch as t
import pdb
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader

class Helper(object):
    """
        utils class: it can provide any function that we need
    """
    def __init__(self):
        self.timber = True

    # def MF_eval(self, model, graph, dataset, K, query_layer):
    def MF_eval(self, model, dataset, K):
        # self, model, graph, dataset, K, query_layer
        all_train = dataset.adj
        all_test = dataset.adj_test

        test_loader = dataset.test_loader
        user_embed, item_embed = model()
        max_k = max(K)
        dot_product = t.mm(user_embed, t.t(item_embed))
        # predictions =  t.t(t.topk(dot_product, k=max_k, dim=0)[1])
        # del dot_product
        

        metric_result = {}
        for k in K:
            metric_result[k] = [[],[]]  #0:Recall, 1:ndcg
        for batch_id, [user, item, participant] in tqdm(enumerate(test_loader)):
            # pdb.set_trace()
            predictions = [t.topk(dot_product[t.tensor([dataset.friendship[user[i].item()]])].squeeze(0)[:, item[i].item()],k=max_k)[1].tolist() for i in range(len(user))]
            rating_K = []
            for i in range(len(user)):
                interm = []
                for index in predictions[i]:
                    interm.append(dataset.friendship[user[i].item()][index])
                rating_K.append(interm)
            rating_K = t.tensor(rating_K)

            for k in K:
                truth = participant.unsqueeze(-1)
                result = truth[:,:k] == rating_K[:,:k]

                recall = t.sum(result).item()

                dcg = t.sum(result/(t.log2(t.arange(2,k+2))),1)
                
                ndcg = t.sum(dcg).item()

                metric_result[k][0].append(recall)
                # metric_result[k][1].append(recall) 
                metric_result[k][1].append(ndcg)
        final_result = {}
        test_user_num = len(dataset.U_I_U_test)
        
        for k in K:
            final_result[k] = {}
            final_result[k]['recall'] = sum(metric_result[k][0])/test_user_num
            final_result[k]['ndcg'] = sum(metric_result[k][1])/test_user_num
        return final_result

    def user_active_eval(self,dataset, K, sorted_friendship):

        test_loader = dataset.test_loader

        metric_result = {}
        max_k = max(K)
        for k in K:
            metric_result[k] = [[],[]]  #0: Recall, 1:ndcg
        for batch_id, [user, item, participant] in tqdm(enumerate(test_loader)): 
            
            rating_K = t.tensor([sorted_friendship[u.item()][:max_k] for u in user])

            for k in K:
                truth = participant.unsqueeze(-1)
                result = truth[:,:k] == rating_K[:,:k]
                recall = t.sum(result).item()
                dcg = t.sum(result/(t.log2(t.arange(2,k+2))),1)        
                ndcg = t.sum(dcg).item()
                metric_result[k][0].append(recall)
                metric_result[k][1].append(ndcg)
        final_result = {}
        test_user_num = len(dataset.U_I_U_test)
        
        for k in K:
            final_result[k] = {}
            final_result[k]['recall'] = sum(metric_result[k][0])/test_user_num
            final_result[k]['ndcg'] = sum(metric_result[k][1])/test_user_num
        return final_result


    def transE_eval(self, model, dataset, K):

        test_loader = dataset.test_loader
        max_k = max(K)

        metric_result = {}
        for k in K:
            metric_result[k] = [[],[]]  #0:Recall, 1:ndcg
        for batch_id, [user, item, participant] in tqdm(enumerate(test_loader)):
            rating_K = []
            for i in range(user.size()[0]):
                index = []

                num_friend = len(dataset.friendship[user[i].item()])
                heads = user[i].reshape(-1,1).repeat(num_friend,1)
                relations = item[i].reshape(-1,1).repeat(num_friend,1)
                tails = t.tensor([x for x in dataset.friendship[user[i].item()]]).reshape(-1,1)
                triplet = t.stack((heads, relations, tails),dim=1).reshape(-1,3)
                predictions = model.predict(triplet)
                topK = t.topk(predictions, k = max_k,largest = False)[1].tolist()
                for z in topK:
                    index.append(dataset.friendship[user[i].item()][z])
                rating_K.append(index)
            rating_K = t.tensor(rating_K)

            for k in K:
                truth = participant.unsqueeze(-1)
                result = truth[:,:k] == rating_K[:,:k]

                recall = t.sum(result).item()

                dcg = t.sum(result/(t.log2(t.arange(2,k+2))),1)
                
                ndcg = t.sum(dcg).item()

                metric_result[k][0].append(recall)
                # metric_result[k][1].append(recall) 
                metric_result[k][1].append(ndcg)
        final_result = {}
        test_user_num = len(dataset.U_I_U_test)
        
        for k in K:
            final_result[k] = {}
            final_result[k]['recall'] = sum(metric_result[k][0])/test_user_num
            final_result[k]['ndcg'] = sum(metric_result[k][1])/test_user_num
        return final_result

    def RGCN_eval(self, model, graph, dataset, K):

        test_loader = dataset.test_loader

        h = model(graph)
        user_embed = h['user']
        item_embed = h['item']
        max_k = max(K)
        dot_product = t.mm(user_embed, t.t(item_embed))
        # predictions =  t.t(t.topk(dot_product, k=max_k, dim=0)[1])
        # del dot_product
        

        metric_result = {}
        for k in K:
            metric_result[k] = [[],[]]  #0:Recall, 1:ndcg
        for batch_id, [user, item, participant] in tqdm(enumerate(test_loader)):
            # pdb.set_trace()
            predictions = [t.topk(dot_product[t.tensor([dataset.friendship[user[i].item()]])].squeeze(0)[:, item[i].item()],k=max_k)[1].tolist() for i in range(len(user))]
            rating_K = []
            for i in range(len(user)):
                interm = []
                for index in predictions[i]:
                    interm.append(dataset.friendship[user[i].item()][index])
                rating_K.append(interm)
            rating_K = t.tensor(rating_K)

            for k in K:
                truth = participant.unsqueeze(-1)
                result = truth[:,:k] == rating_K[:,:k]

                recall = t.sum(result).item()

                dcg = t.sum(result/(t.log2(t.arange(2,k+2))),1)
                
                ndcg = t.sum(dcg).item()

                metric_result[k][0].append(recall)
                # metric_result[k][1].append(recall) 
                metric_result[k][1].append(ndcg)
        final_result = {}
        test_user_num = len(dataset.U_I_U_test)
        
        for k in K:
            final_result[k] = {}
            final_result[k]['recall'] = sum(metric_result[k][0])/test_user_num
            final_result[k]['ndcg'] = sum(metric_result[k][1])/test_user_num
        return final_result

    def eval(self, model, graph, dataset, K, query_layer):

        test_loader = dataset.test_loader

        user_embed,item_embed = model(graph)
        max_k = max(K)
        # dot_product = t.mm(user_embed, t.t(item_embed))
        

        metric_result = {}
        for k in K:
            metric_result[k] = [[],[]]  #0:Recall, 1:ndcg
        for batch_id, [user, item, participant] in tqdm(enumerate(test_loader)):
            users = user_embed[user]
            items = item_embed[item]
            # participants = user_embed[participant]
            query_embed = query_layer(t.concat([users, items],1))
            rating_K = []
            for i in range(len(user)):
                interm = []
                friend_embed = user_embed[dataset.friendship[user[i].item()]]
                scores = (query_embed[i] * friend_embed).sum(1)
                prediction = t.topk(scores,k=max_k)[1].tolist()
                # pdb.set_trace()
                for index in prediction:
                    interm.append(dataset.friendship[user[i].item()][index])

                rating_K.append(interm)
            rating_K = t.tensor(rating_K)

            for k in K:
                truth = participant.unsqueeze(-1)
                result = truth[:,:k] == rating_K[:,:k]

                recall = t.sum(result).item()

                dcg = t.sum(result/(t.log2(t.arange(2,k+2))),1)
                
                ndcg = t.sum(dcg).item()

                metric_result[k][0].append(recall)
                # metric_result[k][1].append(recall) 
                metric_result[k][1].append(ndcg)
        final_result = {}
        test_user_num = len(dataset.U_I_U_test)
        
        for k in K:
            final_result[k] = {}
            final_result[k]['recall'] = sum(metric_result[k][0])/test_user_num
            final_result[k]['ndcg'] = sum(metric_result[k][1])/test_user_num
        return final_result