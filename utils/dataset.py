import pdb
import scipy.sparse as sp
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import random
from scipy.sparse import csr_matrix
import os.path
import dgl

class GBdataset(object):
    def __init__(self, args):
        self.path = './datasets/'+ args.data+'/'
        self.batch_size = args.batch_size
        # self.seed = args.seed
        self.numUsers, self.numItems = self.read_size()
        self.U_I_train,self.graph_U_I_train = self.read_U_I('train.txt')
        self.U_I_val,_ = self.read_U_I('val.txt')
        self.U_I_test,self.adj_test = self.read_U_I('test.txt')
        self.U_I_U_train, self.graph_U_I_train = self.read_U_I_U('train.txt')
        self.U_I_U_val,_ = self.read_U_I_U('val.txt')
        self.U_I_U_test,_ = self.read_U_I_U('test.txt')

        self.social, self.friendship, self.graph_social = self.read_social_relation()
        self.sharer_view, self.participant_view = self.build_view_graph()

        self.UI_train_loader = self.get_UI_loader(self.U_I_train)
        self.UI_val_loader = self.get_UI_loader(self.U_I_val)

        self.train_loader = self.get_loader(self.U_I_U_train)
        self.val_loader = self.get_loader(self.U_I_U_val)
        self.test_loader = self.get_loader(self.U_I_U_test)

    def read_size(self):
        with open(self.path + 'data_size.txt','r') as f:
            line = f.readline()
            elements = line.split()
            numUsers = int(elements[0])
            numItems = int(elements[1])
        return numUsers, numItems

    def read_U_I_U(self, file):
        # [user, item, user]
        U_I_U = []
        # sharer_list = []
        # item_list = []
        # participant_list = []
        users = []
        items = []
        with open(self.path + file, 'r') as f:
            lines = f.readlines()
            for i in lines:
                elements = i.split()
                sharer = int(elements[0])
                item = int(elements[1])
                
                users.append(sharer)
                items.append(item)
                if len(elements) > 3:
                    for e in elements[2:]:
                        participant = int(e)
                        U_I_U.append([sharer, item, participant])
                        users.append(participant)
                        items.append(item)
                else:
                    participant = int(elements[2])
                    U_I_U.append([sharer, item, participant])
                    users.append(participant)
                    items.append(item)
                # participant = int(elements[2])
                # U_I_U.append([sharer, item, participant])
                # sharer_list.append(sharer)
                # item_list.append(item)
                # participant_list.append(participant)
        num_nodes_dict = {'user': self.numUsers, 'item': self.numItems}
        graph_data = {
            ('user', 'rate', 'item'): (torch.tensor(users), torch.tensor(items)),
            ('item', 'rated by', 'user'): (torch.tensor(items), torch.tensor(users))
        }
        g = dgl.heterograph(graph_data, num_nodes_dict)
        return U_I_U,g

    def read_U_I(self, file):
        # [user, item]
        U_I = []
        with open(self.path + file, 'r') as f:
            lines = f.readlines()
            for i in lines:
                elements = i.split()
                sharer = int(elements[0])
                item = int(elements[1])

                if len(elements) > 3:
                    for e in elements[2:]:
                        participant = int(e)
                        U_I.append([participant, item])
                else:
                    participant = int(elements[2])
                    U_I.append([participant, item])
                U_I.append([sharer, item])
        # data = np.ones(len(U_I))
        users, items = list(zip(*U_I))
        
        num_nodes_dict = {'user': self.numUsers, 'item': self.numItems}
        graph_data = {
            ('user', 'rate', 'item'): (torch.tensor(users), torch.tensor(items)),
            ('item', 'rated by', 'user'): (torch.tensor(items), torch.tensor(users))
        }
        g = dgl.heterograph(graph_data, num_nodes_dict)

        return U_I, g

    def get_UI_loader(self, ui):
        users, items = list(zip(*ui))
        data = TensorDataset(torch.LongTensor(users), torch.LongTensor(items))
        dataloader = DataLoader(data, batch_size = self.batch_size, shuffle = True)
        return dataloader

    def read_social_relation(self):
        # [user, user]
        social = []
        friendship = {}
        with open(self.path + 'social_relation.txt','r') as f:
            lines = f.readlines()
            for i in lines:
                elements = i.split()
                u1 = int(elements[0])
                u2 = int(elements[1])
                social.append([u1, u2])
                if u1 not in friendship:
                    friendship[u1] = [u2]
                else:
                    friendship[u1].append(u2)
                if u2 not in friendship:
                    friendship[u2] = [u1]
                else:
                    friendship[u2].append(u1)
        for k,v in friendship.items():
            friendship[k] = list(set(v))
        u1, u2 = list(zip(*social))
        num_nodes_dict = {'user': self.numUsers}
        u_1 = u1 + u2
        u_2 = u2 + u1
        graph_data = {
            ('user', 'friend', 'user'): (torch.tensor(u_1), torch.tensor(u_2)),
        }
        g = dgl.heterograph(graph_data, num_nodes_dict)
        return social, friendship, g
    

    def get_loader(self, U_I_U):
        users, items, participant = list(zip(*U_I_U))
       
        data = TensorDataset(torch.LongTensor(users), torch.LongTensor(items), torch.LongTensor(participant))
        # data = TensorDataset(torch.LongTensor([users,items]), torch.LongTensor(participant))
        dataloader = DataLoader(data, batch_size = self.batch_size, shuffle = True)
        return dataloader


    def build_view_graph(self):
        sharer_list = []
        item_list = []
        participant_list = []
        for i in self.U_I_U_train:
            sharer_list.append(i[0])
            item_list.append(i[1])
            participant_list.append(i[2])
        num_nodes_dict_sharer = {'sharer': self.numUsers, 'item': self.numItems}
        graph_data_sharer = {
            ('sharer', 'rate', 'item'): (torch.tensor(sharer_list), torch.tensor(item_list)),
            ('item', 'rated by', 'sharer'): (torch.tensor(item_list), torch.tensor(sharer_list))
        }
        g_sharer = dgl.heterograph(graph_data_sharer, num_nodes_dict_sharer)

        num_nodes_dict_participant = {'participant': self.numUsers, 'item': self.numItems}
        graph_data_participant = {
            ('participant', 'rate', 'item'): (torch.tensor(participant_list), torch.tensor(item_list)),
            ('item', 'rated by', 'participant'): (torch.tensor(item_list), torch.tensor(participant_list))
        }
        g_participant = dgl.heterograph(graph_data_participant, num_nodes_dict_participant)
        return g_sharer, g_participant