
from copy import deepcopy
from sqlite3 import paramstyle
import ssl
import numpy as np
from sklearn.utils import shuffle
import torch
import itertools
seed = 0
torch.manual_seed(seed)
import torch.nn.functional as F
import argparse
from models.models import *
from data.load_data import *
from models.autoencoder import MyGAE,MyVGAE,MyTask,MLPDecoder
from train import *
from utils import get_positional_embedding, seed_everything,compute_acc_unsupervised
import scipy.sparse as sp

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='ele-fashion')
parser.add_argument('--modal_text', type=str, default='clip')
parser.add_argument('--modal_vision', type=str, default='clip')
parser.add_argument('--modal_structure', type=str, default='node2vec')
parser.add_argument('--modal_names', type=list, default=['text','vision','structure'])
parser.add_argument('--dataset_task', type = dict,default={'books-lp':'link','books-nc':'node','cloth-copurchase':'link','sports-copurchase':'link','ele-fashion':'node'})
parser.add_argument('--share_dims', type=int, default=128)
parser.add_argument('--hidden_dims', type=int, default=128)
parser.add_argument('--subgraph_scale', type=int, default=4096) #4096
parser.add_argument('--model_lr', type=float, default=0.005) #0.005
parser.add_argument('--feature_lr', type=float, default=0.005)
parser.add_argument('--beta', type=float, default=0.01)
parser.add_argument('--alpha', type=float, default=1)
parser.add_argument('--threshold', type=float, default=0.8)
parser.add_argument('--wd_lr', type=float, default=0.01)
parser.add_argument('--inner_lr', type=float, default=0.05)
parser.add_argument('--output_dims', default=128, type=int)
parser.add_argument('--info', type=str, default='')
parser.add_argument('--cuda', type = int,default=3)




        
def MGSHIP_pretrain(args, device,task, index): 

    
    file = open('./model/parameter/pretrain_process_for_{}_{}-{}-{}_{}_{}{}.csv'.format(args.dataset,args.modal_text,args.modal_vision,args.modal_structure,task, index,args.info), 'w')
    dataloader = dataGraph(args)
    args.modals= dataloader.modals
    

    MLPs = {}
    args.dims = [dataloader.original_feats[modal].shape[1] for modal in dataloader.modals]
    dim_sum=int(np.sum(args.dims))

    for modal in dataloader.modals:
        dim=dataloader.original_feats[modal].shape[1]
        MLPs[modal] = MLP(dim, args.hidden_dims, dim_sum-dim).to(device)

    wdiscriminator = WDiscriminator(dim_sum)
    optimizer_wd = torch.optim.Adam(wdiscriminator.parameters(), lr=args.wd_lr, weight_decay=5e-4)
    wdiscriminator.to(device)


    params = []
    for value in MLPs.values():
        params.extend(value.parameters())
    pretrain_model = MyGAE(TransformerEncoder(args, int(dim_sum), args.output_dims))
    pretrain_model.to(device)

    model_params = []
    model_params.extend(pretrain_model.parameters())

    if task == 'nmk':
        auxiliary = nn.Linear(args.output_dims, dim_sum, bias=True).to(device)
        model_params.extend(auxiliary.parameters())
    elif task == 'dgi':
        auxiliary = nn.Linear(args.output_dims, args.output_dims, bias=True).to(device)
        model_params.extend(auxiliary.parameters())
    elif task == 'sim':
        auxiliary = nn.Linear(args.output_dims, args.output_dims, bias=False).to(device)
        model_params.extend(auxiliary.parameters())
    else:
        auxiliary = None

    optimizer_all = torch.optim.Adam([{'params': params, 'lr': 0.0001},{'params': model_params}], lr=args.model_lr,
                                     weight_decay=5e-4) 

    epoch = 0
    loss_value = 1000000
    cnt = 0
    optimal_model = pretrain_model
    last_model = pretrain_model

    match_graph = ConnectMatch(args,dim_sum).to(device)

    while True:

        subgraphs = dataloader.fetch_subgraph(norm = True)
        dataloader.subgraph_to_tensor(subgraphs, device)

        pretrain_model, loss = MGSHIP_gradient(args, task,pretrain_model,auxiliary,subgraphs,optimizer_all, MLPs,wdiscriminator,optimizer_wd,match_graph, epoch, device, file)
        
        if epoch%3==0: 

            file.write(str(round(loss,4))+'\n')
            file.flush()

            print('loss', loss, cnt, epoch)
            if loss < loss_value  and epoch > 20:
                optimal_model = last_model
                loss_value = loss
                cnt = 0
                torch.save(optimal_model.encoder.state_dict(),
                        './model/parameter/pretrain_encoder_for_{}_{}-{}-{}_{}_{}{}.pth'.format(args.dataset,args.modal_text,args.modal_vision,args.modal_structure,task, index,args.info))
                
                torch.save(match_graph.state_dict(),
                        './model/parameter/pretrain_relation_anchor_for_{}_{}-{}-{}_{}_{}{}.pth'.format(args.dataset,args.modal_text,args.modal_vision,args.modal_structure,task, index,args.info))

                mlps_state = {}
                for data in args.modals:mlps_state[data] = MLPs[data].state_dict()
                torch.save(mlps_state,'./model/parameter/pretrain_feature_anchor_for_{}_{}-{}-{}_{}_{}{}.pth'.format(args.dataset,args.modal_text,args.modal_vision,args.modal_structure,task,index,args.info))

            else:
                cnt += 1

            if cnt == 100:  # 150
                

                file.close()
                break

            last_model = deepcopy(pretrain_model)

        epoch += 1















def MGSHIP_test1(args, device, index, pretrain_task,ratio, modal_ratio, num, tune = True):
    dataset = args.dataset
    task = args.dataset_task[dataset]
    modals = args.modal_names
    graph = load_test_graph(args.dataset, index ,args.modal_text,args.modal_vision,args.modal_structure)
    graph.to_tensor(device)
    graph.identify(task,ratio)
    graph.split_modal_nodes(modal_ratio)

    args.dims = [graph.text_feat.shape[1],graph.vision_feat.shape[1],graph.structure_feat.shape[1]]
    dim_sum = np.sum(args.dims)
    MLPs = {}
    MLPs_dict = torch.load('./model/pretrain_feature_anchor_for_{}_{}-{}-{}_{}_{}.pth'.format(args.dataset,args.modal_text,args.modal_vision,args.modal_structure, pretrain_task,num))
    graph.original_feats = {'text':graph.text_feat,'vision':graph.vision_feat, 'structure':graph.structure_feat}


    
    for valid in modals:
        dim = args.dims[modals.index(valid)]
        MLPs[valid] = MLP(dim, args.hidden_dims, dim_sum-dim).to(device)
       #MLPs[valid].load_state_dict(MLPs_dict[valid], strict=True)
        if valid == 'text':
            x = MLPs[valid](graph.text_feat)
            dim = int(np.sum(args.dims[:modals.index(valid)]))
            x = torch.cat([x[:,:dim],graph.text_feat,x[:,dim:]],dim=1)
            graph.text_feat = x
        if valid == 'vision':
            x = MLPs[valid](graph.vision_feat)
            dim = int(np.sum(args.dims[:modals.index(valid)]))
            x = torch.cat([x[:,:dim],graph.vision_feat,x[:,dim:]],dim=1)
            graph.vision_feat = x
        if valid == 'structure':
            x = MLPs[valid](graph.structure_feat)
            dim = int(np.sum(args.dims[:modals.index(valid)]))
            x = torch.cat([x[:,:dim],graph.structure_feat,x[:,dim:]],dim=1)
            graph.structure_feat = x

    graph = graph.multimodal_to_flat_graph(task)
    graph.modal_names = args.modal_names
    encoder = TransformerEncoder(args, int(dim_sum), args.output_dims)
    
    if task == 'node':
        for valid in modals:MLPs[valid].Dropout.p = 0.2
        num_classes = int(graph.labels.max().item() + 1)
        model = MyTask('node', encoder, args.output_dims, num_classes).to(device)
    else:
        for valid in modals:MLPs[valid].Dropout.p = 0.2
        model = MyTask('link', encoder, args.share_dims).to(device)
    model.encoder.load_state_dict(
        torch.load('./model/pretrain_encoder_for_{}_{}-{}-{}_{}_{}.pth'.format(args.dataset,args.modal_text,args.modal_vision,args.modal_structure, pretrain_task,num)), strict=True)
    mlp_params = []
    for value in MLPs.values():
        mlp_params.extend(value.parameters())
    if tune:
        optimizer = torch.optim.Adam(
            [{'params': mlp_params, 'lr': 0.001}, {'params': model.parameters(), 'lr': 0.001}],
                weight_decay=5e-4)

    else:
        for p in model.encoder.parameters(): p.requires_grad = False
        optimizer = torch.optim.Adam(
            [{'params': mlp_params, 'lr': 0.001}, {'params': model.decoder.parameters(), 'lr': 0.001}],
            weight_decay=5e-4)
    
    return train_graph(MLPs=MLPs, model=model, graph=graph, optimizer=optimizer, dims = args.dims,device=device)




def MGSHIP_test2(args, device, index, pretrain_task,ratio, modal_ratio, num, tune = False): 

    dataset = args.dataset
    task = args.dataset_task[dataset]
    modals = args.modal_names
    graph = load_test_graph(args.dataset, index ,args.modal_text,args.modal_vision,args.modal_structure,norm = flag)
    graph.to_tensor(device)
    graph.identify(task,ratio)
    graph.split_modal_nodes(modal_ratio)

    args.dims = [graph.text_feat.shape[1],graph.vision_feat.shape[1],graph.structure_feat.shape[1]]
    dim_sum = np.sum(args.dims)
    MLPs = {}
    MLPs_dict = torch.load('./model/pretrain_feature_anchor_for_{}_{}-{}-{}_{}_{}.pth'.format(args.dataset,args.modal_text,args.modal_vision,args.modal_structure, pretrain_task,num))
    graph.original_feats = {'text':graph.text_feat,'vision':graph.vision_feat, 'structure':graph.structure_feat}
    for valid in modals:
        dim = args.dims[modals.index(valid)]
        MLPs[valid] = MLP(dim, args.hidden_dims, dim_sum-dim).to(device)
        MLPs[valid].load_state_dict(MLPs_dict[valid], strict=True)
    
    graph.modal_names = args.modal_names
    encoder = TransformerEncoder_original(args, int(dim_sum), args.output_dims)

    if task == 'node':
        for valid in modals:MLPs[valid].Dropout.p = 0.2
        num_classes = int(graph.labels.max().item() + 1)
        model = MyTask('node', encoder, args.output_dims, num_classes).to(device)
    else:
        for valid in modals:MLPs[valid].Dropout.p = 0.2   #0.0
        model = MyTask('link', encoder, args.share_dims).to(device)
    model.encoder.load_state_dict(
        torch.load('./model/pretrain_encoder_for_{}_{}-{}-{}_{}_{}.pth'.format(args.dataset,args.modal_text,args.modal_vision,args.modal_structure, pretrain_task,num)), strict=True)
    mlp_params = []
    for value in MLPs.values():
        mlp_params.extend(value.parameters())
    if tune:
        optimizer = torch.optim.Adam(
            [{'params': mlp_params, 'lr': 0.001}, {'params': model.parameters(), 'lr': 0.001}],
                weight_decay=5e-4)

    else:
        for p in model.encoder.parameters(): p.requires_grad = False
        optimizer = torch.optim.Adam(
            [{'params': mlp_params, 'lr': 0.001}, {'params': model.decoder.parameters(), 'lr': 0.001}],
            weight_decay=5e-4)

    return train_graph_isolate(MLPs=MLPs, model=model, graph=graph, optimizer=optimizer, dims = args.dims,device=device) 





def MGSHIP_test3(args, device, index, pretrain_task,ratio, modal_ratio, num, tune = False): 


    dataset = args.dataset
    task = args.dataset_task[dataset]
    modals = args.modal_names
    graph = load_test_graph(args.dataset, index ,args.modal_text,args.modal_vision,args.modal_structure)
    graph.to_tensor(device)
    graph.identify(task,ratio)
    graph.split_modal_nodes(modal_ratio)


    

    args.dims = [graph.text_feat.shape[1],graph.vision_feat.shape[1],graph.structure_feat.shape[1]]
    dim_sum = np.sum(args.dims)
    MLPs = {}
    MLPs_dict = torch.load('./model/pretrain_feature_anchor_for_{}_{}-{}-{}_{}_{}.pth'.format(args.dataset,args.modal_text,args.modal_vision,args.modal_structure, pretrain_task,num))
    graph.original_feats = {'text':graph.text_feat,'vision':graph.vision_feat, 'structure':graph.structure_feat}
    for valid in modals:
        dim = args.dims[modals.index(valid)]
        MLPs[valid] = MLP(dim, args.hidden_dims, dim_sum-dim).to(device)
        MLPs[valid].load_state_dict(MLPs_dict[valid], strict=True)


    connectMatch = ConnectMatch(args,dim_sum)
    connectMatch.load_state_dict(torch.load('./model/pretrain_relation_anchor_for_{}_{}-{}-{}_{}_{}.pth'.format(args.dataset,args.modal_text,args.modal_vision,args.modal_structure, pretrain_task,num)))

    xs = {}
    for modal in graph.modal_names:
        modal_feat = MLPs[modal](graph.original_feats[modal])
        modal_dim = int(np.sum(args.dims[:graph.modal_names.index(modal)]))
        xs[modal] = torch.cat([modal_feat[:,:modal_dim],graph.original_feats[modal],modal_feat[:,modal_dim:]],dim=1).detach()

    
    graph.enrich_adj = connectMatch.get_test_relation(test_graph,xs,args.threshold,task)
    graph.anchor_nodes = torch.nn.Parameter(connectMatch.super_nodes.detach().clone()).to(device)



    
    graph.modal_names = args.modal_names
    encoder = TransformerEncoder_original(args, int(dim_sum), args.output_dims)

    if task == 'node':
        for valid in modals:MLPs[valid].Dropout.p = 0.2
        num_classes = int(graph.labels.max().item() + 1)
        model = MyTask('node', encoder, args.output_dims, num_classes).to(device)
    else:
        for valid in modals:MLPs[valid].Dropout.p = 0.2   #0.0
        model = MyTask('link', encoder, args.share_dims).to(device)
    model.encoder.load_state_dict(
        torch.load('./model/pretrain_encoder_for_{}_{}-{}-{}_{}_{}.pth'.format(args.dataset,args.modal_text,args.modal_vision,args.modal_structure, pretrain_task,num)), strict=True)
    mlp_params = []
    for value in MLPs.values():
        mlp_params.extend(value.parameters())
    if tune:
        optimizer = torch.optim.Adam(
            [{'params': mlp_params, 'lr': 0.001},{'params': graph.anchor_nodes, 'lr': 0.001}, {'params': model.parameters(), 'lr': 0.001}],
                weight_decay=5e-4)

    else:
        for p in model.encoder.parameters(): p.requires_grad = False
        optimizer = torch.optim.Adam(
            [{'params': mlp_params, 'lr': 0.001},{'params': graph.anchor_nodes, 'lr': 0.001},  {'params': model.decoder.parameters(), 'lr': 0.001}],
            weight_decay=5e-4)

    return train_graph_final(MLPs=MLPs, model=model, graph=graph, optimizer=optimizer, dims = args.dims,device=device) 








args = parser.parse_args()
if args.cuda != -1:
    device = torch.device("cuda:" + str(args.cuda))
else:
    device = torch.device("cpu")



MGSHIP_pretrain(args, device,'edge',0)
MGSHIP_test3(args, device, 0, 'dgi',0.02, 0.4, 0, tune = True)








