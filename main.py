
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
parser.add_argument('--fuse_scale', type=int, default=1024)
parser.add_argument('--model_lr', type=float, default=0.005) #0.005
parser.add_argument('--feature_lr', type=float, default=0.005)
parser.add_argument('--beta', type=float, default=0.01)
parser.add_argument('--wd_lr', type=float, default=0.01)
parser.add_argument('--output_dims', default=128, type=int)
parser.add_argument('--info', type=str, default='')
parser.add_argument('--cuda', type = int,default=3)











        
def PRIMG_pretrain(args, device,task, index): 

    
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
    loss_value = 0
    cnt = 0
    optimal_model = pretrain_model
    last_model = pretrain_model
    #match_graph = AttMatch(dim_sum,args.share_dims).to(device)

    match_graph = ConnectMatch(dim_sum).to(device)

    

    # match_graph = ConnectMatch_mlp(args,dim_sum,device,proto_num=256).to(device)
    # match_graph.set_super_nodes(dataloader.fetch_subgraph())



    

    while True:


        subgraphs = dataloader.fetch_subgraph(norm = True)
        fugraphs = dataloader.fetch_for_fused(norm = True)

        dataloader.subgraph_to_tensor(subgraphs, device)
        dataloader.subgraph_to_tensor(fugraphs, device)


        for idx,name in enumerate(dataloader.modals):
            fugraphs[name] = fugraphs[name].transformation(MLPs[name],int(np.sum(args.dims[:idx])))
            fugraphs[name].x = fugraphs[name].x.detach()


        


        pretrain_model, loss = PRIMG_gradient(args, task,pretrain_model,auxiliary,subgraphs,fugraphs,optimizer_all, MLPs,wdiscriminator,optimizer_wd,match_graph, epoch, device, file)
        
        if epoch%3==0:

            datas = args.modals
            valid_graph = load_test_graph(args.dataset, 7,args.modal_text,args.modal_vision,args.modal_structure,norm=True)
            valid_graph.to_tensor(device)
            valid_graph.identify(args.dataset_task[args.dataset],0.1)
            valid_graph.split_modal_nodes(0.4)



            '''
            valid_graph.original_feats = {'text':valid_graph.text_feat,'vision':valid_graph.vision_feat, 'structure':valid_graph.structure_feat}
            for valid in datas:
                if valid == 'text':
                    x = MLPs[valid](valid_graph.text_feat)
                    dim = int(np.sum(args.dims[:datas.index(valid)]))
                    x = torch.cat([x[:,:dim],valid_graph.text_feat,x[:,dim:]],dim=1)
                    valid_graph.text_feat = x
                if valid == 'vision':
                    x = MLPs[valid](valid_graph.vision_feat)
                    dim = int(np.sum(args.dims[:datas.index(valid)]))
                    x = torch.cat([x[:,:dim],valid_graph.vision_feat,x[:,dim:]],dim=1)
                    valid_graph.vision_feat = x
                if valid == 'structure':
                    x = MLPs[valid](valid_graph.structure_feat)
                    dim = int(np.sum(args.dims[:datas.index(valid)]))
                    x = torch.cat([x[:,:dim],valid_graph.structure_feat,x[:,dim:]],dim=1)
                    valid_graph.structure_feat = x
            
            valid_graph = valid_graph.multimodal_to_flat_graph(args.dataset_task[args.dataset])
            weights = OrderedDict(pretrain_model.named_parameters())
            if args.dataset_task[args.dataset] == 'node':emb = pretrain_model.encode(valid_graph.x, valid_graph.edge_index, weights).detach()
            else: emb = pretrain_model.encode(valid_graph.x,valid_graph.extra_info["train_edge_index"] , weights).detach()
            loss=compute_acc_unsupervised(emb, args.dataset_task[args.dataset],valid_graph.extra_info)
            '''


            for valid in datas:
                if valid == 'text':
                    x = MLPs[valid](valid_graph.text_feat)
                    dim = int(np.sum(args.dims[:datas.index(valid)]))
                    text_x = torch.cat([x[:,:dim],valid_graph.text_feat,x[:,dim:]],dim=1)
                if valid == 'vision':
                    x = MLPs[valid](valid_graph.vision_feat)
                    dim = int(np.sum(args.dims[:datas.index(valid)]))
                    vision_x = torch.cat([x[:,:dim],valid_graph.vision_feat,x[:,dim:]],dim=1)
                if valid == 'structure':
                    x = MLPs[valid](valid_graph.structure_feat)
                    dim = int(np.sum(args.dims[:datas.index(valid)]))
                    structure_x = torch.cat([x[:,:dim],valid_graph.structure_feat,x[:,dim:]],dim=1)

            feats = valid_graph.multimodal_to_isolate_graph(text_x,vision_x,structure_x)
            weights = OrderedDict(pretrain_model.named_parameters())
            if args.dataset_task[args.dataset] == 'node':
                emb_text = pretrain_model.encode(feats['text'], valid_graph.edge_index, weights).detach()
                emb_vision = pretrain_model.encode(feats['vision'], valid_graph.edge_index, weights).detach()
                emb_structure = pretrain_model.encode(feats['structure'], valid_graph.edge_index, weights).detach()
                loss=compute_acc_unsupervised_isolate(args.dataset_task[args.dataset],emb_text,emb_vision,emb_structure,valid_graph.train_labels,valid_graph.test_labels,valid_graph.labels)
            else:
                emb_text = pretrain_model.encode(feats['text'], valid_graph.train_edge_index, weights).detach()
                emb_vision = pretrain_model.encode(feats['vision'], valid_graph.train_edge_index, weights).detach()
                emb_structure = pretrain_model.encode(feats['structure'], valid_graph.train_edge_index, weights).detach()
                loss=compute_acc_unsupervised_isolate(args.dataset_task[args.dataset],emb_text,emb_vision,emb_structure,test_index_positive=valid_graph.test_edge_index,test_index_negative =valid_graph.test_edge_index_negative  )


            
            #(task, text_feats,vision_feats,structure_feats,train_labels = None,test_labels = None,labels = None, test_index_positive = None, test_index_negative = None):
             











            print(datas,loss)
            

            file.write(str(round(loss,4))+'\n')
            file.flush()
            






            print('loss', loss, cnt, epoch)
            if loss > loss_value  and epoch > 20:
                optimal_model = last_model
                loss_value = loss
                cnt = 0
                torch.save(optimal_model.encoder.state_dict(),
                        './model/parameter/pretrain_encoder_for_{}_{}-{}-{}_{}_{}{}.pth'.format(args.dataset,args.modal_text,args.modal_vision,args.modal_structure,task, index,args.info))
                
                torch.save(match_graph.super_nodes,
                        './model/parameter/pretrain_pivots_for_{}_{}-{}-{}_{}_{}{}.pth'.format(args.dataset,args.modal_text,args.modal_vision,args.modal_structure,task, index,args.info))

                mlps_state = {}
                for data in args.modals:mlps_state[data] = MLPs[data].state_dict()
                torch.save(mlps_state,'./model/parameter/pretrain_mlp_for_{}_{}-{}-{}_{}_{}{}.pth'.format(args.dataset,args.modal_text,args.modal_vision,args.modal_structure,task,index,args.info))

            else:
                cnt += 1

            if cnt == 100:  # 150
                

                file.close()
                break

            last_model = deepcopy(pretrain_model)

        epoch += 1















def PRIMG_test1(args, device, index, pretrain_task,ratio, modal_ratio, num, tune = True): 
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
    MLPs_dict = torch.load('./model/pretrain_mlp_for_{}_{}-{}-{}_{}_{}.pth'.format(args.dataset,args.modal_text,args.modal_vision,args.modal_structure, pretrain_task,num))
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
        for valid in modals:MLPs[valid].Dropout.p = 0.8
        num_classes = int(graph.labels.max().item() + 1)
        model = MyTask('node', encoder, args.output_dims, num_classes).to(device)
    else:
        for valid in modals:MLPs[valid].Dropout.p = 0.6
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




def PRIMG_test2(args, device, index, pretrain_task,ratio, modal_ratio, num, tune = False): 

    if num == 0: flag = True
    if num == 5: flag = False
    if args.dataset == 'ele-fashion': flag= True



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
    MLPs_dict = torch.load('./model/pretrain_mlp_for_{}_{}-{}-{}_{}_{}.pth'.format(args.dataset,args.modal_text,args.modal_vision,args.modal_structure, pretrain_task,num))
    graph.original_feats = {'text':graph.text_feat,'vision':graph.vision_feat, 'structure':graph.structure_feat}
    for valid in modals:
        dim = args.dims[modals.index(valid)]
        MLPs[valid] = MLP(dim, args.hidden_dims, dim_sum-dim).to(device)
        MLPs[valid].load_state_dict(MLPs_dict[valid], strict=True)
    
    graph.modal_names = args.modal_names
    encoder = TransformerEncoder_original(args, int(dim_sum), args.output_dims)

    if task == 'node':
        for valid in modals:MLPs[valid].Dropout.p = 0.95
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

    return train_graph_isolate(MLPs=MLPs, model=model, graph=graph, optimizer=optimizer, dims = args.dims,device=device) ###################










def parameter_beta_training(args,device):
    for beta in [0.001,0.005,0.015,0.02,0.025,0.1]:
        args.beta = beta
        args.info = '_norm=true_beta_'+str(beta)
        PRIMG_pretrain(args, device,'edge',0)
        PRIMG_pretrain(args, device,'nmk',0)





args = parser.parse_args()
if args.cuda != -1:
    device = torch.device("cuda:" + str(args.cuda))
else:
    device = torch.device("cpu")


PRIMG_test2(args, device, 10, 'dgi',0.02, 0.4, 5, tune = True)











