
import torch
import math
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import GATConv, GCNConv,SAGEConv,GINConv,TransformerConv
from torch.nn import Parameter,BatchNorm1d,Dropout
from torch_scatter import scatter_add,scatter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops,add_self_loops
from torch.distributions import Normal
from torch import dropout, nn
from .layers import *
import torch.nn.functional as F
from utils import uniform
from collections import OrderedDict
from torch_geometric.nn import global_mean_pool
from models.autoencoder import negative_sampling
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score
from torch_geometric.data import Batch, Data
from typing import Union, Tuple, Optional
from torch_geometric.typing import PairTensor, Adj, OptTensor
import numpy as np
import torch_geometric



def softplus(x):
  return torch.nn.functional.softplus(x, beta=100)

def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)

def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)

class Encoder(torch.nn.Module):
    def __init__(self, args, in_channels, out_channels):
        super(Encoder, self).__init__()
        self.args = args
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=False)
        if args.model in ['GAE']:
            self.conv2 = GCNConv(2 * out_channels, out_channels, cached=False)
        elif args.model in ['VGAE']:
            self.conv_mu = GCNConv(2 * out_channels, out_channels, cached=False)
            self.conv_logvar = GCNConv(
                2 * out_channels, out_channels, cached=False)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        if self.args.model in ['GAE']:
            return self.conv2(x, edge_index)
        elif self.args.model in ['VGAE']:
            return self.conv_mu(x, edge_index), self.conv_logvar(x, edge_index)






class TransformerEncoder(torch.nn.Module):
    def __init__(self, args, in_channels, out_channels):
        super(TransformerEncoder, self).__init__()
        self.args = args
        self.conv1 = TransformerConv(in_channels, 2*out_channels)
        self.conv2 = TransformerConv(2*out_channels, out_channels)
        self.reset_parameters()


    def forward(self, x, edge_index, weights, inner_loop=True,edge_weight = None):

        if edge_weight is None:
            x = F.relu(self.conv1(x, edge_index))
            return self.conv2(x, edge_index)
        else:
            x = F.relu(self.conv1(x,edge_index = None,edge_weight= edge_weight))
            return self.conv2(x, edge_index = None,edge_weight=edge_weight)


    def reset_parameters(self):
        for tensor in self.parameters():
            if len(tensor.size())==1:
                bound = 1.0 / math.sqrt(tensor.size()[0])
                tensor.data.uniform_(-bound, bound)
            else:
                stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
                tensor.data.uniform_(-stdv, stdv)


class TransformerEncoder_original(torch.nn.Module):
    def __init__(self, args, in_channels, out_channels):
        super(TransformerEncoder_original, self).__init__()
        self.args = args
        self.conv1 = torch_geometric.nn.TransformerConv(in_channels, 2*out_channels)
        self.conv2 = torch_geometric.nn.TransformerConv(2*out_channels, out_channels)
        self.reset_parameters()


    def forward(self, x, edge_index, weights, inner_loop=True,edge_weight = None):

        if edge_weight is None:
            x = F.relu(self.conv1(x, edge_index))
            return self.conv2(x, edge_index)
        else:
            x = F.relu(self.conv1(x,edge_index = None,edge_weight= edge_weight))
            return self.conv2(x, edge_index = None,edge_weight=edge_weight)


    def reset_parameters(self):
        for tensor in self.parameters():
            if len(tensor.size())==1:
                bound = 1.0 / math.sqrt(tensor.size()[0])
                tensor.data.uniform_(-bound, bound)
            else:
                stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
                tensor.data.uniform_(-stdv, stdv)








class Prompt(nn.Module):
    def __init__(self, in_channels: int, p_num: int):
        super(Prompt, self).__init__()
        self.p_list = nn.Parameter(torch.Tensor(p_num, in_channels))
        self.a = nn.Linear(in_channels, p_num)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.p_list)
        self.a.reset_parameters()

    def add(self, x: Tensor):
        score = self.a(x)
        # weight = torch.exp(score) / torch.sum(torch.exp(score), dim=1).view(-1, 1)
        weight = F.softmax(score, dim=1)
        p = weight.mm(self.p_list)

        return x + p



class PromptMLP(nn.Module):
    def __init__(self, in_channels: int, p_num: int):
        super(PromptMLP, self).__init__()
        self.in_channels = in_channels
        self.p_list = nn.Parameter(torch.Tensor(in_channels,p_num))
        self.a = nn.Parameter(torch.Tensor(p_num, in_channels))
        self.Dropout = nn.Dropout(p=0.0)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.p_list)
        glorot(self.a)

    def forward(self, x: Tensor):

        if x.shape[1]>self.in_channels:
            score = torch.matmul(x[:,:self.in_channels],self.p_list)
        else: score = torch.matmul(x,self.p_list[:x.shape[1],:])

        weight = F.softmax(score, dim=1)
        p = weight.mm(self.Dropout(self.p_list.T))

        return  p


class Net(torch.nn.Module):
    def __init__(self,train_dataset):
        super(Net, self).__init__()
        self.conv1 = GATConv(train_dataset.num_features, 256, heads=4)
        self.lin1 = torch.nn.Linear(train_dataset.num_features, 4 * 256)
        self.conv2 = GATConv(4 * 256, 256, heads=4)
        self.lin2 = torch.nn.Linear(4 * 256, 4 * 256)
        self.conv3 = GATConv(
            4 * 256, train_dataset.num_classes, heads=6, concat=False)
        self.lin3 = torch.nn.Linear(4 * 256, train_dataset.num_classes)

    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index) + self.lin1(x))
        x = F.elu(self.conv2(x, edge_index) + self.lin2(x))
        x = self.conv3(x, edge_index) + self.lin3(x)
        return x




class WDiscriminator(torch.nn.Module):
    def __init__(self, hidden_size, hidden_size2=512):
        super(WDiscriminator, self).__init__()
        self.hidden = MyGCNConv(hidden_size, hidden_size2, cached=False)
        self.hidden2 = torch.nn.Linear(hidden_size2, hidden_size2)
        self.output = torch.nn.Linear(hidden_size2, 1)

    def forward(self, input_embd,edge_index):
        return self.output(F.leaky_relu(self.hidden2(F.leaky_relu(self.hidden(input_embd,edge_index), 0.2, inplace=True)), 0.2, inplace=True))

    def reset_parameters(self):
        #print('reset')
        glorot(self.hidden.weight)
        glorot(self.hidden2.weight)
        glorot(self.output.weight)



class empty_MLP(nn.Module):
    def __init__(self,device):
        super(empty_MLP, self).__init__()
        self.device = device
        self.Dropout = nn.Dropout(p=0.0)

    def forward(self, x):
        return torch.zeros([x.shape[0],0]).to(self.device)





class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(dim_in, dim_hidden)
        self.fc2 = nn.Linear(dim_hidden, dim_out)
        self.Dropout = nn.Dropout(p=0.0)
        glorot(self.fc1.weight)
        glorot(self.fc2.weight)


    def forward(self, x):

        return (self.fc2(F.relu(self.Dropout(self.fc1(x)))))

    def reset_parameters(self):
        print('reset')
        glorot(self.fc1.weight)
        glorot(self.fc2.weight)









class MLP2(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP2, self).__init__()
        self.fc1 = nn.Linear(dim_in, dim_hidden)
        self.fc2 = nn.Linear(dim_hidden, dim_hidden)
        self.fc3 = nn.Linear(dim_hidden, dim_out)
        glorot(self.fc1.weight)
        glorot(self.fc2.weight)
        glorot(self.fc2.weight)


    def forward(self, x):

        return self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))

    def reset_parameters(self):
        print('reset')
        glorot(self.fc1.weight)
        glorot(self.fc2.weight)
        glorot(self.fc3.weight)





class LightPrompt(torch.nn.Module):
    def __init__(self, token_dim, token_num_per_group, group_num=1, inner_prune=None):
        """
        :param token_dim:
        :param token_num_per_group:
        :param group_num:   the total token number = token_num_per_group*group_num, in most cases, we let group_num=1.
                            In prompt_w_o_h mode for classification, we can let each class correspond to one group.
                            You can also assign each group as a prompt batch in some cases.

        :param prune_thre: if inner_prune is None, then all inner and cross prune will adopt this prune_thre
        :param isolate_tokens: if Trure, then inner tokens have no connection.
        :param inner_prune: if inner_prune is not None, then cross prune adopt prune_thre whereas inner prune adopt inner_prune
        """
        super(LightPrompt, self).__init__()

        self.inner_prune = inner_prune

        self.token_list = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.empty(token_num_per_group, token_dim)) for i in range(group_num)])

        self.token_init(init_method="kaiming_uniform")

    def token_init(self, init_method="kaiming_uniform"):
        if init_method == "kaiming_uniform":
            for token in self.token_list:
                torch.nn.init.kaiming_uniform_(token, nonlinearity='leaky_relu', mode='fan_in', a=0.01)
        else:
            raise ValueError("only support kaiming_uniform init, more init methods will be included soon")

    def inner_structure_update(self):
        return self.token_view()

    def token_view(self, ):
        """
        each token group is viewed as a prompt sub-graph.
        turn the all groups of tokens as a batch of prompt graphs.
        :return:
        """
        pg_list = []
        for i, tokens in enumerate(self.token_list):
            # inner link: token-->token
            token_dot = torch.mm(tokens, torch.transpose(tokens, 0, 1))
            token_sim = torch.sigmoid(token_dot)  # 0-1

            inner_adj = torch.where(token_sim < self.inner_prune, 0, 1)
            edge_index = inner_adj.nonzero().t().contiguous()

            pg_list.append(Data(x=tokens, edge_index=edge_index, y=torch.tensor([i]).long()))

        pg_batch = Batch.from_data_list(pg_list)
        return pg_batch






class AE(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(AE, self).__init__()

        self.input = MLP(dim_in, dim_hidden,dim_out)
        self.output = MLP(dim_out,dim_hidden, dim_in)
        self.loss = torch.nn.MSELoss(size_average=True)



    def forward(self, x):

        return self.input(x)
    
    def con_loss(self,x,con_x):
        return self.loss(x,con_x) 

    def reset_parameters(self):
        print('reset')
        self.input.reset_parameters()
        self.output.reset_parameters()


        

class WDiscriminator_old(torch.nn.Module):
    def __init__(self, hidden_size, hidden_size2=512):
        super(WDiscriminator_old, self).__init__()
        self.hidden = torch.nn.Linear(hidden_size, hidden_size2)
        self.hidden2 = torch.nn.Linear(hidden_size2, hidden_size2)
        self.output = torch.nn.Linear(hidden_size2, 1)

    def forward(self, input_embd):
        return self.output(F.leaky_relu(self.hidden2(F.leaky_relu(self.hidden(input_embd), 0.2, inplace=True)), 0.2, inplace=True))

    def reset_parameters(self):
        print('reset')
        glorot(self.hidden.weight)
        glorot(self.hidden2.weight)
        glorot(self.output.weight)




class matchGAT3(torch.nn.Module):

    def __init__(self,in_channels,out_channels):
        super(matchGAT3, self).__init__()
        self.in_channels = in_channels
        self.out_channels= out_channels
        self.conv = TransEncoder(in_channels,out_channels)

        self.weight_l = nn.Sequential(nn.Linear(out_channels, out_channels, bias=True),nn.Linear(out_channels, 1, bias=True) )
        self.weight_r = nn.Sequential(nn.Linear(out_channels, out_channels, bias=True), nn.Linear(out_channels, 1, bias=True))

        self.reset_parameters()

    def reset_parameters(self):
        for tensor in self.parameters():
            if len(tensor.size())==1:
                bound = 1.0 / math.sqrt(tensor.size()[0])
                tensor.data.uniform_(-bound, bound)
            else:
                stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
                tensor.data.uniform_(-stdv, stdv)


    def forward(self,graphs):

        Xs = []
        for key in graphs.keys():
            graph = graphs[key]
            Xs.append(self.conv(graph.x.detach(), graph.edge_index))

        features = torch.cat(Xs,dim = 0)
        alpha_l = self.weight_l(features)
        alpha_r = self.weight_r(features)


        alpha = alpha_l+alpha_r.t()
        alpha = (alpha+alpha.t())/2
        adj = F.sigmoid(alpha)

        return adj
    
    def get_final_adj(self,adj,graphs,threshold):

        cum = 0
        for key in graphs.keys():
            graph = graphs[key]

            adj[cum:cum+graph.x.shape[0],cum:cum+graph.x.shape[0]] = 0.0
            adj[graph.edge_index[0]+cum,graph.edge_index[1]+cum] = 1.0
            cum+=graph.x.shape[0]

        adj = torch.where(adj>threshold,adj,torch.tensor(0, dtype=torch.float).to(adj))
        print('adj.sum()',adj.sum())
        return adj.nonzero().t().detach()

        

        



class matchGAT2(torch.nn.Module):

    def __init__(self,in_channels,out_channels):
        super(matchGAT2, self).__init__()
        self.in_channels = in_channels
        self.out_channels= out_channels
        self.conv = MetaEncoder(in_channels,out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        for tensor in self.parameters():
            if len(tensor.size())==1:
                bound = 1.0 / math.sqrt(tensor.size()[0])
                tensor.data.uniform_(-bound, bound)
            else:
                stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
                tensor.data.uniform_(-stdv, stdv)


    def forward(self,graphs):

        Xs = []
        for key in graphs.keys():
            graph = graphs[key]
            Xs.append(self.conv(graph.x.detach(), graph.edge_index))

        features = torch.cat(Xs,dim = 0)
        alpha =  features@features.t()  
        adj = F.sigmoid(alpha)
        print(adj)

        return adj
    
    def get_final_adj(self,adj,graphs,threshold):

        cum = 0
        for key in graphs.keys():
            graph = graphs[key]

            adj[cum:cum+graph.x.shape[0],cum:cum+graph.x.shape[0]] = 0.0
            adj[graph.edge_index[0]+cum,graph.edge_index[1]+cum] = 1.0
            cum+=graph.x.shape[0]

        adj = torch.where(adj>threshold,adj,torch.tensor(0, dtype=torch.float).to(adj))
        print('adj.sum()',adj.sum())
        return adj.nonzero().t().detach()

        










class ConnectMatch_mlp(torch.nn.Module):
    def __init__(self,args,node_dim,device,proto_num=256):
        super(ConnectMatch_mlp, self).__init__()
        self.node_dim =node_dim
        self.proto_num = proto_num
        self.device = device
        self.MLPs =torch.nn.ModuleDict()
        for key in args.data_dims.keys():
            self.MLPs[key] = MLP( args.data_dims[key], args.hidden_dims, self.node_dim).to(device)
        self.super_nodes = None
        self.reset_parameters()

    def reset_parameters(self):
        for tensor in self.parameters():
            if len(tensor.size())==1:
                bound = 1.0 / math.sqrt(tensor.size()[0])
                tensor.data.uniform_(-bound, bound)
            else:
                stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
                tensor.data.uniform_(-stdv, stdv)

    def set_super_nodes(self,graphs):
        num = self.proto_num//4
        self.super_nodes_dict = {}
        for key in self.MLPs.keys():
            x = graphs[key].x[np.random.choice(graphs[key].x.shape[0],num)]
            self.super_nodes_dict[key] = torch.from_numpy(x).to(torch.float).to(self.device)



    def forward(self,graphs):

        super_nodes = []
        for key in self.super_nodes_dict.keys():
            super_nodes.append(self.MLPs[key](self.super_nodes_dict[key]))
        super_nodes= torch.cat(super_nodes,0)

                


        cum = 0
        tmp = np.sum([graphs[key].x.shape[0] for key in graphs.keys()])
        adj = torch.tensor(np.zeros((tmp,tmp)))
        for key in graphs.keys():
            graph = graphs[key]
            adj[graph.edge_index[0]+cum,graph.edge_index[1]+cum] = 1.0
            cum+=graph.x.shape[0]

        features = torch.cat([graphs[name].x for name in graphs.keys()],dim = 0)
        adj = adj.to(features)
        down = F.sigmoid(super_nodes@features.t())
        features = torch.cat([features,super_nodes],dim=0)
        right = F.sigmoid(features@super_nodes.t())

        adj = torch.cat([adj,down],dim=0)
        adj = torch.cat([adj,right],dim=1)
        print(adj)

        self.super_nodes = super_nodes

        return adj
    

    def get_final_adj(self,adj,threshold):

        
        return torch.where(adj>threshold,1,0).nonzero().t().detach()

















class ConnectMatch(torch.nn.Module):
    def __init__(self,args,node_dim,proto_num=256):
        super(ConnectMatch, self).__init__()
        self.node_dim =node_dim
        self.proto_num = proto_num
        self.modal_encoder = nn.ModuleDict()
        for modal in args.modal_names:
            self.modal_encoder[modal] = MLP(node_dim, args.hidden_dims, args.share_dims)
        self.modal_encoder['virtual'] = MLP(node_dim, args.hidden_dims, args.share_dims)

        self.super_nodes = torch.nn.Parameter(torch.randn(self.proto_num,self.node_dim,dtype=torch.float))
        self.reset_parameters()

    def reset_parameters(self):
        for tensor in self.parameters():
            if len(tensor.size())==1:
                bound = 1.0 / math.sqrt(tensor.size()[0])
                tensor.data.uniform_(-bound, bound)
            else:
                stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
                tensor.data.uniform_(-stdv, stdv)


    def forward(self,graphs):


        cum = 0
        tmp = np.sum([graphs[key].x.shape[0] for key in graphs.keys()])
        adj = torch.tensor(np.zeros((tmp,tmp)))
        for key in graphs.keys():
            graph = graphs[key]
            adj[graph.edge_index[0]+cum,graph.edge_index[1]+cum] = 1.0
            cum+=graph.x.shape[0]

        features = torch.cat([self.modal_encoder[name](graphs[name].x.detach()) for name in graphs.keys()],dim = 0)
        super_nodes = self.modal_encoder['virtual'](self.super_nodes)
        adj = adj.to(features)
        print(super_nodes.dtype,features.dtype)
        down = F.sigmoid(super_nodes@features.t())
        features = torch.cat([features,super_nodes],dim=0)
        right = F.sigmoid(features@super_nodes.t())

        adj = torch.cat([adj,down],dim=0)
        adj = torch.cat([adj,right],dim=1)
        print(adj)

        return adj
    

    def get_final_adj(self,adj,threshold):



        adj = torch.where(adj>threshold,1,0)


        print('rrrrrr',adj[:4096,:4096].sum())
        print('rrrrrr',adj[-self.proto_num:,:4096].sum())
        print('rrrrrr',adj[:4096,-self.proto_num:].sum())
        print('rrrrrr',adj[-self.proto_num:,-self.proto_num:].sum())
        return adj.nonzero().t().detach()


    def get_test_relation(self,graph,xs,threshold,task):
        node_num = xs['text'].x.shape[0]

        remain_lists={'text':graph.text_list,'vision':graph.vision_list,'structure':graph.structure_list}

        cum = 0
        tmp = np.sum([xs[key].shape[0] for key in xs.keys()])+self.proto_num
        adj = torch.tensor(np.zeros((tmp,tmp)))
        for key in xs.keys():
            if task == 'link':
                adj[graph.train_edge_index[0]+cum,graph.train_edge_index[1]+cum] = 1.0
            else:adj[graph.edge_index[0]+cum,graph.edge_index[1]+cum] = 1.0
            cum+=xs[key].shape[0]

        features = torch.cat([self.modal_encoder[name]( xs[name][remain_lists[name]]) for name in xs.keys()],dim = 0)
        super_nodes = self.modal_encoder['virtual'](self.super_nodes)
        down_index = torch.where(F.sigmoid(super_nodes@features.t())>threshold,1,0).nonzero().t()

        all_list = graph.text_list + [i + node_num for i in graph.vision_list] + [i + node_num * 2 for i in graph.structure_list]

        adj[down_index[0]+cum,down_index[1][all_list]] =1
        adj[down_index[1][all_list],down_index[0]+cum] =1
        hub_index = torch.where(F.sigmoid(super_nodes@super_nodes.t())>threshold,1,0).nonzero().t()
        adj[hub_index[0]+cum,hub_index[1]+cum] =1

        return adj.nonzero().t().detach()

        



        












class AttMatch(torch.nn.Module):
    def __init__(self,in_channels,out_channels,heads = 1):
        super(AttMatch, self).__init__()
        self.in_channels = in_channels
        self.out_channels= out_channels
        self.heads = heads

        self.conv1 = SAGEConv(in_channels*2,out_channels)
        self.conv2 = SAGEConv(out_channels*2,out_channels)


        self.lin_key1 = torch.nn.Linear(in_channels, heads * out_channels)
        self.lin_query1 = torch.nn.Linear(in_channels, heads * out_channels)
        self.lin_value1 = torch.nn.Linear(in_channels, heads * in_channels)

        self.lin_key2 = torch.nn.Linear(out_channels, heads * out_channels)
        self.lin_query2 = torch.nn.Linear(out_channels, heads * out_channels)
        self.lin_value2 = torch.nn.Linear(out_channels, heads * out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        for tensor in self.parameters():
            if len(tensor.size())==1:
                bound = 1.0 / math.sqrt(tensor.size()[0])
                tensor.data.uniform_(-bound, bound)
            else:
                stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
                tensor.data.uniform_(-stdv, stdv)


    def get_attention(self,Xs,index,layer):
        source = Xs[index]
        target = torch.cat(Xs,dim = 0)

        if layer == 1:
            query = self.lin_query1(source).view(-1, self.heads * self.out_channels)
            key = self.lin_key1(target).view(-1, self.heads * self.out_channels)
            alpha = ( key@query.t() ) / math.sqrt(self.out_channels)
            alpha = F.softmax(alpha,dim=0)
            alpha = alpha/(alpha.sum(0)+1e-16)


        if layer == 2:
            query = self.lin_query2(source).view(-1, self.heads * self.out_channels)
            key = self.lin_key2(target).view(-1, self.heads * self.out_channels)
            alpha = ( key@query.t() ) / math.sqrt(self.out_channels)
            alpha = F.softmax(alpha,dim=0)
            alpha = alpha/(alpha.sum(0)+1e-16)
        

        return alpha



    def aggr_and_update(self,graphs,Xs,layer):
        Xs_new = []
        target = torch.cat(Xs,dim = 0)
        for i,key in enumerate(graphs.keys()):
            attention = self.get_attention(Xs,i,layer)

            if layer == 1:
            
                out = self.lin_value1(target).view(-1, self.heads * self.in_channels)
                out = attention.t() @ out

            if layer == 2:
            
                out = self.lin_value2(target).view(-1, self.heads * self.out_channels)
                out = attention.t() @ out







            ot_feature = Xs[i] - out
            features = torch.cat((Xs[i],ot_feature),dim = 1)
            if layer == 1:Xs_new.append(F.relu(self.conv1(x = features, edge_index = graphs[key].edge_index)))
            if layer == 2:Xs_new.append(self.conv2(x = features, edge_index = graphs[key].edge_index))

        return Xs_new
    


    def get_adj(self,Xs):

        features = torch.cat(Xs,dim = 0)
        
        alpha =  features@features.t()    #(self.lin(features).view(-1, self.head, self.out_channels)*self.att).sum(dim=-1).mean(1,keepdim = True)
        
        adj = F.sigmoid(alpha)
        print(adj)

        return adj


    def get_final_adj(self,adj,graphs,threshold):

        cum = 0
        for key in graphs.keys():
            graph = graphs[key]

            adj[cum:cum+graph.x.shape[0],cum:cum+graph.x.shape[0]] = 0.0
            adj[graph.edge_index[0]+cum,graph.edge_index[1]+cum] = 1.0
            cum+=graph.x.shape[0]

        adj = torch.where(adj>threshold,adj,torch.tensor(0, dtype=torch.float).to(adj))
        print('adj.sum()',adj.sum())
        return adj.nonzero().t().detach()


    def forward(self,graphs):
        Xs = [graphs[key].x for key in graphs.keys()]
        Xs1 = self.aggr_and_update( graphs,Xs,1)
        Xs2 = self.aggr_and_update(graphs,Xs1, 2)
        adj = self.get_adj(Xs2)

        
        return adj






class matchGAT(torch.nn.Module):

    def __init__(self,in_channels,out_channels,head = 1):
        super(matchGAT, self).__init__()
        self.in_channels = in_channels
        self.out_channels= out_channels
        self.head = head
        self.wight1 = nn.Linear(in_channels*2, out_channels, bias=True)
        self.wight2 = nn.Linear(out_channels*2, out_channels, bias=True)

        self.lin_l1 = nn.Linear(in_channels, head * out_channels, bias=True)
        self.lin_r1 = nn.Linear(in_channels, head * out_channels, bias=True)

        self.lin_l2 = nn.Linear(out_channels, head * out_channels, bias=True)
        self.lin_r2 = nn.Linear(out_channels, head * out_channels, bias=True)

        #self.lin_l = nn.Linear(out_channels, head * out_channels, bias=True)
        #self.lin_r = nn.Linear(out_channels, head * out_channels, bias=True)
        
        
        self.att_l1 = Parameter(torch.Tensor(1,head, out_channels))
        self.att_r1 = Parameter(torch.Tensor(1,head, out_channels))

        self.att_l2 = Parameter(torch.Tensor(1,head, out_channels))
        self.att_r2 = Parameter(torch.Tensor(1,head, out_channels))

        #self.att_l = Parameter(torch.Tensor(1,head, out_channels))
        #self.att_r = Parameter(torch.Tensor(1,head, out_channels))

        self.lin = nn.Linear(out_channels, head * out_channels, bias=True)
        self.att = Parameter(torch.Tensor(1,head, out_channels))

        self.reset_parameters()



    def reset_parameters(self):
        for tensor in self.parameters():
            if len(tensor.size())==1:
                bound = 1.0 / math.sqrt(tensor.size()[0])
                tensor.data.uniform_(-bound, bound)
            else:
                stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
                tensor.data.uniform_(-stdv, stdv)



    def get_attention(self,Xs,index,layer):
        source = Xs[index]
        target = torch.cat(Xs,dim = 0)


        if layer == 1:
            alpha_l = (self.lin_l1(source).view(-1, self.head, self.out_channels)*self.att_l1).sum(dim=-1).mean(1,keepdim = True)
            alpha_r = (self.lin_r1(target).view(-1, self.head, self.out_channels)*self.att_r1).sum(dim=-1).mean(1,keepdim = True)

        if layer == 2:
            alpha_l = (self.lin_l2(source).view(-1, self.head, self.out_channels)*self.att_l2).sum(dim=-1).mean(1,keepdim = True)
            alpha_r = (self.lin_r2(target).view(-1, self.head, self.out_channels)*self.att_r2).sum(dim=-1).mean(1,keepdim = True)

        
        
        alpha = alpha_l+alpha_r.t()

        

        alpha = F.leaky_relu(alpha,0.2)
        alpha = F.softmax(alpha,dim = 1)

        return alpha
    

    def aggr_and_update(self,graphs,Xs,layer):
        Xs_new = []
        target = torch.cat(Xs,dim = 0)
        for i,key in enumerate(graphs.keys()):
            attention = self.get_attention(Xs,i,layer)
            ot_feature = Xs[i] - torch.mm(attention, target)
            features = torch.cat((Xs[i],ot_feature),dim = 1)


            source_idx,target_idx = add_self_loops(graphs[key].edge_index)[0]
            messages = features.index_select(0, target_idx)
            aggregation = scatter(messages, source_idx, dim=0, dim_size=features.shape[0], reduce='mean')

            if layer == 1:aggregation = self.wight1(F.relu(aggregation))
            if layer == 2:aggregation = self.wight2(F.relu(aggregation))

            Xs_new.append(aggregation)

        return Xs_new
    

    def get_adj2(self,Xs):

        source = torch.cat(Xs,dim = 0)
        target = torch.cat(Xs,dim = 0)


        




        alpha_l = (self.lin_l(source).view(-1, self.head, self.out_channels)*self.att_l).sum(dim=-1).mean(1,keepdim = True)
        alpha_r = (self.lin_r(target).view(-1, self.head, self.out_channels)*self.att_r).sum(dim=-1).mean(1,keepdim = True)

        #alpha = (alpha_l+alpha_r)/2.0
        #alpha = alpha+alpha.t()

        alpha = alpha_l+alpha_r.t()
        alpha = F.sigmoid(alpha)

        adj = torch.where(alpha>0.6,alpha,torch.tensor(0, dtype=torch.float).to(alpha))
        return adj
    

    def get_adj(self,Xs):

        features = torch.cat(Xs,dim = 0)

        features = self.lin(features)
        
        alpha =  features@features.t()    #(self.lin(features).view(-1, self.head, self.out_channels)*self.att).sum(dim=-1).mean(1,keepdim = True)
        
        adj = F.sigmoid(alpha)
        print(adj)

        return adj
    
    def get_final_adj(self,adj,graphs,threshold):

        cum = 0
        for key in graphs.keys():
            graph = graphs[key]

            adj[cum:cum+graph.x.shape[0],cum:cum+graph.x.shape[0]] = 0.0
            adj[graph.edge_index[0]+cum,graph.edge_index[1]+cum] = 1.0
            cum+=graph.x.shape[0]

        adj = torch.where(adj>threshold,adj,torch.tensor(0, dtype=torch.float).to(adj))
        print('adj.sum()',adj.sum())
        return adj.nonzero().t().detach()


    def forward(self,graphs):
        Xs = [graphs[key].x for key in graphs.keys()]
        Xs1 = self.aggr_and_update( graphs,Xs,1)
        Xs2 = self.aggr_and_update(graphs,Xs1, 2)
        adj = self.get_adj(Xs2)

        
        return adj






class TransformerConv(MessagePassing):
    
    _alpha: OptTensor

    def __init__(self, in_channels: Union[int, Tuple[int,
                                                     int]], out_channels: int,
                 heads: int = 1, concat: bool = True, beta: bool = False,
                 dropout: float = 0., edge_dim: Optional[int] = None,
                 bias: bool = True, root_weight: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(TransformerConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.beta = beta and root_weight
        self.root_weight = root_weight
        self.concat = concat
        self.dropout = dropout
        self.edge_dim = edge_dim

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_key = torch.nn.Linear(in_channels[0], heads * out_channels)
        self.lin_query = torch.nn.Linear(in_channels[1], heads * out_channels)
        self.lin_value = torch.nn.Linear(in_channels[0], heads * out_channels)
        if edge_dim is not None:
            self.lin_edge = torch.nn.Linear(edge_dim, heads * out_channels, bias=False)
        else:
            self.lin_edge = self.register_parameter('lin_edge', None)

        if concat:
            self.lin_skip = torch.nn.Linear(in_channels[1], heads * out_channels,
                                   bias=bias)
            if self.beta:
                self.lin_beta = torch.nn.Linear(3 * heads * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)
        else:
            self.lin_skip = torch.nn.Linear(in_channels[1], out_channels, bias=bias)
            if self.beta:
                self.lin_beta = torch.nn.Linear(3 * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        if self.edge_dim:
            self.lin_edge.reset_parameters()
        self.lin_skip.reset_parameters()
        if self.beta:
            self.lin_beta.reset_parameters()


    def test(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, edge_weight = None):
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        print(self.propagate(edge_index, x=x,edge_attr=None))
        print(self.full_message( x=x[0], edge_weight=edge_weight))


    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, edge_weight = None):
        """"""

        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        # propagate_type: (x: PairTensor, edge_attr: OptTensor)
        if edge_weight is None:
            if edge_index.shape[1]>x[0].shape[0]: #x[0].shape[0]**2*0.01

                edge_weight = torch.zeros(x[0].shape[0],x[0].shape[0]).to(x[0].device)
                edge_weight[edge_index[0],edge_index[1]] = 1
                out= self.full_message( x=x[0], edge_weight=edge_weight)
            
            else: out = self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr, size=None)
        else: out = self.full_message( x=x[0], edge_weight=edge_weight)

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.root_weight:
            x_r = self.lin_skip(x[1])
            if self.lin_beta is not None:
                beta = self.lin_beta(torch.cat([out, x_r, out - x_r], dim=-1))
                beta = beta.sigmoid()
                out = beta * x_r + (1 - beta) * out
            else:
                out += x_r

        return out



    def full_message(self,x,edge_weight):
        query = self.lin_query(x).view(-1, self.heads * self.out_channels)
        key = self.lin_key(x).view(-1, self.heads * self.out_channels)

        alpha = ( key@query.t() ) / math.sqrt(self.out_channels) 
        #print('alpha',alpha)

        alpha = F.softmax(alpha,dim=0)
        
        alpha = alpha*edge_weight
        #alpha = alpha/(torch.clamp(alpha.sum(1),min = 1e-8))
        alpha = alpha/(alpha.sum(0)+1e-16)

        #print("full",alpha)
        
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = self.lin_value(x).view(-1, self.heads * self.out_channels)
        out = alpha.t() @ out

        return out.view(-1, self.heads, self.out_channels)



    def heresoftmax(self,src, index):
        src_max = scatter(src.detach(), index, dim = 0,reduce='max')
        out = src - src_max.index_select(0, index)
        out = out.exp()
        out_sum = scatter(out, index, dim = 0,reduce='sum') + 1e-16
        out_sum = out_sum.index_select(0, index)
        return out_sum





    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: OptTensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:



        query = self.lin_query(x_i).view(-1, self.heads, self.out_channels)
        key = self.lin_key(x_j).view(-1, self.heads, self.out_channels)

        if self.lin_edge is not None:
            assert edge_attr is not None
            edge_attr = self.lin_edge(edge_attr).view(-1, self.heads,
                                                      self.out_channels)
            key += edge_attr

        alpha = (query * key).sum(dim=-1) / math.sqrt(self.out_channels)
        
        #print(self.heresoftmax(alpha, index))


        alpha = softmax(alpha, index, ptr, size_i)

        #print('message',alpha)

        
        



        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = self.lin_value(x_j).view(-1, self.heads, self.out_channels)
        if edge_attr is not None:
            out += edge_attr

        out *= alpha.view(-1, self.heads, 1)
        return out

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)



    

    

            







class ALLEncoder(torch.nn.Module):
    def __init__(self, args, in_channels, out_channels,model):
        super(ALLEncoder, self).__init__()
        self.args = args
        if model == 'GAT':
            self.conv1 = GATConv(in_channels, 2*out_channels)
            self.conv2 = GATConv(2*out_channels, out_channels)
        elif model == 'GCN':
            self.conv1 = GCNConv(in_channels, 2*out_channels)
            self.conv2 = GCNConv(2*out_channels, out_channels)
        elif model == 'Transformer':
            self.conv1 = TransformerConv(in_channels, 2*out_channels)
            self.conv2 = TransformerConv(2*out_channels, out_channels)
        elif model == 'SAGE':
            self.conv1 = SAGEConv(in_channels, 2*out_channels)
            self.conv2 = SAGEConv(2*out_channels, out_channels)

            


        self.reset_parameters()


    def forward(self, x, edge_index, weights, inner_loop=True):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)

    def reset_parameters(self):
        for tensor in self.parameters():
            if len(tensor.size())==1:
                bound = 1.0 / math.sqrt(tensor.size()[0])
                tensor.data.uniform_(-bound, bound)
            else:
                stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
                tensor.data.uniform_(-stdv, stdv)



        
        



        




