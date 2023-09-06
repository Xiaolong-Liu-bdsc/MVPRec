from array import array
import dgl.function as fn
import torch.nn as nn
import torch
import torch.nn.functional as F
import pdb
import numpy as np
from dgl.nn import GATConv

# model definition
class MLP(nn.Module):
    # define model elements
    def __init__(self, n_inputs, dim):
        super(MLP, self).__init__()
        self.layer = nn.Linear(n_inputs, dim)
        self.activation = nn.Sigmoid()
 
    # forward propagate input
    def forward(self, X):
        X = self.layer(X)
        X = self.activation(X)
        return X
        

def message_func(edges):
    dic = {}
    dic['message'] = edges.src['h'] / torch.sqrt(edges.src['degree']).unsqueeze(1)
    return dic
 
def reduce_func(nodes):
    return {'h_agg': torch.sum(nodes.mailbox['message'], dim = 1) / torch.sqrt(nodes.data['degree'].unsqueeze(1))}

class LightGCNLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, graph, h, etype):
        with graph.local_scope():
            src, _ , dst = etype
            feat_src = h[src]
            feat_dst = h[dst]
            aggregate_fn = fn.copy_u('h', 'm')

            degs = graph.out_degrees(etype = etype).float().clamp(min = 1)
            norm = torch.pow(degs, -0.5)
            shp = norm.shape + (1,) * (feat_src.dim() - 1)
            norm = torch.reshape(norm, shp)
            feat_src = feat_src * norm

            graph.nodes[src].data['h'] = feat_src
            graph.update_all(aggregate_fn, fn.sum(msg = 'm', out = 'h'), etype = etype)

            # rst = graph.dstdata['h'][dst]
            rst = graph.nodes[dst].data['h']
            degs = graph.in_degrees(etype = etype).float().clamp(min = 1)
            norm = torch.pow(degs, -0.5)
            shp = norm.shape + (1,) * (feat_dst.dim() - 1)
            norm = torch.reshape(norm, shp)
            rst = rst * norm
            return rst

class HeteroDotProductPredictor(nn.Module):
    def forward(self, graph, h, etype):
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'), etype = etype)
            return graph.edges[etype].data['score']

class our_model(nn.Module):
    def __init__(self, args, dataset):
        super(our_model, self).__init__()
        self.embed_size = args.embed_size
        self.num_users = dataset.numUsers
        self.num_items = dataset.numItems
        self.user_embedding = torch.nn.init.xavier_uniform_(torch.nn.Parameter(torch.randn(dataset.numUsers, self.embed_size)))
        self.item_embedding = torch.nn.init.xavier_uniform_(torch.nn.Parameter(torch.randn(dataset.numItems, self.embed_size)))
        self.sharer_view = dataset.sharer_view
        self.participant_view = dataset.participant_view
        self.graph_U_I_train = dataset.graph_U_I_train
        self.graph_social = dataset.graph_social
        self.predictor = HeteroDotProductPredictor()
        self.n_heads = args.head_num
        # self.head_dim = 16
        # self.n_output = 1
        self.layer_num = args.n_layers
        self.query_ui1 = torch.nn.Parameter(torch.rand(self.embed_size* 2, self.embed_size))
        self.query_ui2 = torch.nn.Parameter(torch.rand(self.embed_size, self.n_heads*self.embed_size))
        self.mul_key_p1 = torch.nn.Parameter(torch.rand(self.embed_size, self.n_heads*self.embed_size))
        self.init_transform = torch.nn.Parameter(torch.randn(self.embed_size))
        self.part_transform = torch.nn.Parameter(torch.randn(self.embed_size))
        self.item_transform = torch.nn.Parameter(torch.randn(self.embed_size))
        
        self.friends, self.friends_mask = self.build_friends(dataset.friendship)
        self.build_model()
        self.build_social_model()


    def build_social_model(self):
        self.social_layer = nn.ModuleList()
        for idx in range(self.layer_num):
            h2h = LightGCNLayer()
            self.social_layer.append(h2h)


    def build_model(self):
        self.layers = nn.ModuleList()
        for idx in range(self.layer_num):
            h2h = LightGCNLayer()
            self.layers.append(h2h)


    def build_friends(self, friendship):
        
        max_friends = 0
        for u in friendship.keys():
            max_friends = max(max_friends, len(friendship[u]))
        friends = -torch.ones((self.num_users, max_friends),dtype = torch.long)
        for u in friendship.keys():
            friend = friendship[u]
            friends[u,:len(friend)] = torch.tensor(friend)
        friends_mask = torch.zeros(friends.shape)
        friends_mask[friends == -1] = -1000.0
        return friends, friends_mask

    def fuse_views(self):
        user_embed_sharer, item_embed_sharer, user_embed_participant, item_embed_participant, user_embed_social = self.get_embedding()
        # temp_emb = torch.stack([item_embed_sharer, item_embed_participant],dim=0)
        # temp = (temp_emb * self.item_transform).sum(-1)
        # weight = torch.softmax(temp,dim=0)
        # ea_embeddings = (weight.unsqueeze(-1) * temp_emb).sum(0)

        temp_emb = torch.stack([user_embed_sharer, user_embed_social],dim=0)
        temp = (temp_emb * self.init_transform).sum(-1)
        weight = torch.softmax(temp,dim=0)
        ua_embeddings_sharer = (weight.unsqueeze(-1) * temp_emb).sum(0)  

        temp_emb = torch.stack([user_embed_participant, user_embed_social],dim=0)
        temp = (temp_emb * self.part_transform).sum(-1)
        weight = torch.softmax(temp,dim=0)
        ua_embeddings_participant = (weight.unsqueeze(-1) * temp_emb).sum(0)       

        # u_concat_share = torch.concat([user_embed_sharer, user_embed_social], dim = 1)
        # ua_embeddings_sharer = torch.matmul(u_concat_share, self.linear_sharer)
        # u_concat_participant = torch.concat([user_embed_participant, user_embed_social], dim = 1)
        # ua_embeddings_participant = torch.matmul(u_concat_participant,self.linear_part)

        return item_embed_sharer,item_embed_participant , ua_embeddings_sharer, ua_embeddings_participant

    def forward(self, sharer, item, participant):
        device = self.user_embedding.device
        sharer = sharer.to(device)
        item = item.to(device)
        participant = participant.to(device)

        item_embed_sharer,item_embed_participant, ua_embeddings_sharer, ua_embeddings_participant = self.fuse_views()
        
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        consistent_loss = (1- cos(item_embed_sharer, item_embed_participant)).mean()
        # consistent_loss = 0
        temp_emb = torch.stack([item_embed_sharer, item_embed_participant],dim=0)
        temp = (temp_emb * self.item_transform).sum(-1)
        weight = torch.softmax(temp,dim=0)
        ea_embeddings = (weight.unsqueeze(-1) * temp_emb).sum(0)
        # item_concat = torch.concat([item_embed_sharer, item_embed_participant], dim = 1)
        # ea_embeddings = torch.matmul(item_concat, self.linear_item)
        
        sharer_embeddings = ua_embeddings_sharer[sharer]
        item_embeddings = ea_embeddings[item]
        participant_embeddings = ua_embeddings_participant[participant]
        
        # calculate recommendation loss
        neg_candidate = torch.randint(self.num_items, size = item.shape)
        neg_candidates  = ea_embeddings[neg_candidate]

        score = (sharer_embeddings * item_embeddings).sum(1)
        score_neg = (sharer_embeddings * neg_candidates).sum(1)
        bprloss = -(score - score_neg).sigmoid().log().sum()
        
        Sigmoid = torch.nn.Sigmoid()
        u_conc_i = torch.concat([sharer_embeddings, item_embeddings], -1)
        query = Sigmoid(torch.matmul(u_conc_i, self.query_ui1))
        
        query = torch.unsqueeze(query, 1)
        query = torch.matmul(query, self.query_ui2)

        friend_embedding = ua_embeddings_participant[self.friends[sharer.cpu()].to(device)]
        
        LogSoftmax = torch.nn.LogSoftmax(dim=-1)
        # key vector
        key = torch.matmul(friend_embedding, self.mul_key_p1)
        
        query_split = torch.split(query, self.embed_size, dim=2)
        key_split = torch.split(key, self.embed_size, dim=2)
        
        score = []
        for i in range(self.n_heads):
            prtc_scores = (query_split[i] * key_split[i]).sum(-1)
            prtc_scores += self.friends_mask[sharer.cpu()].to(device)
            prtc_scores = LogSoftmax(prtc_scores)
            score.append(prtc_scores)
        prtc_scores = torch.stack(score).mean(0)
        # query_mul = torch.concat(torch.split(query, self.n_heads, dim=2), dim =-1)
        # key_mul = torch.concat(torch.split(key, self.n_heads, dim=2),dim=-1)
        # prtc_scores = (query_mul * key_mul).sum(-1)
        # prtc_scores += self.friends_mask[sharer.cpu()].to(device)
        # prtc_scores = LogSoftmax(prtc_scores)

        res1 = self.friends[sharer.cpu()] == participant.cpu().unsqueeze(-1)

        row, col = res1.nonzero(as_tuple = True)
    
        prtc_loss = - prtc_scores[row, col].mean()


        return bprloss, prtc_loss,prtc_scores, consistent_loss



    def get_embedding(self):
        # h = self.node_features
        device = self.user_embedding.device

        # LightGCN on U_I graph
        self.sharer_view = self.sharer_view.to(device)
        self.participant_view = self.participant_view.to(device)
        user_embed = [self.user_embedding]
        item_embed = [self.item_embedding]
        h = {'sharer': self.user_embedding, 'item': self.item_embedding}

        for layer in self.layers:

            h_item = layer(self.sharer_view, h, ('sharer', 'rate', 'item'))
            h_user = layer(self.sharer_view, h, ('item', 'rated by', 'sharer'))

            user_embed.append(h_user)
            item_embed.append(h_item)
            h = {'sharer': h_user, 'item': h_item}

        user_embed_sharer = torch.mean(torch.stack(user_embed, dim = 0), dim = 0)
        item_embed_sharer = torch.mean(torch.stack(item_embed, dim = 0), dim = 0)

        h = {'participant': self.user_embedding, 'item': self.item_embedding}
        user_embed = [self.user_embedding]
        item_embed = [self.item_embedding]

        for layer in self.layers:

            h_item = layer(self.participant_view, h, ('participant', 'rate', 'item'))
            h_user = layer(self.participant_view, h, ('item', 'rated by', 'participant'))

            user_embed.append(h_user)
            item_embed.append(h_item)
            h = {'participant': h_user, 'item': h_item}

        user_embed_participant = torch.mean(torch.stack(user_embed, dim = 0), dim = 0)
        item_embed_participant = torch.mean(torch.stack(item_embed, dim = 0), dim = 0)

        # social
        self.graph_social = self.graph_social.to(device)
        h = {'user': self.user_embedding}
        user_social = [self.user_embedding]
        for layer in self.social_layer:
            h_user = layer(self.graph_social, h, ('user', 'friend', 'user'))
            user_social.append(h_user)
            h = {'user': h_user}
        user_embed_social = torch.mean(torch.stack(user_social, dim = 0), dim = 0)
        
        
        # final_user = torch.matmul(torch.concat([user_embed , user_embed_social], 1), self.concat_user)
        # final_item = item_embed   
        return user_embed_sharer, item_embed_sharer, user_embed_participant, item_embed_participant, user_embed_social


