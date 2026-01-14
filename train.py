from turtle import pos
from copy import deepcopy
from unittest import result
import torch
import torch.nn.functional as F
import sklearn.neighbors
import copy
from collections import OrderedDict
import numpy as np
from utils import test,test2,seed_everything
from models.models import MLP,WDiscriminator,WDiscriminator_old,matchGAT
from models.autoencoder import negative_sampling
import time
from torch_geometric.nn import global_mean_pool
from sklearn.metrics import roc_auc_score, f1_score
from scipy.sparse import coo_array
from data.load_data import *
from utils import *



EPS = 1e-15




def train_wdiscriminator(graph_s, graph_t, wdiscriminator, optimizer_d, batch_d_per_iter=20):


    wdiscriminator.reset_parameters()

    for p in wdiscriminator.parameters(): p.requires_grad = True


    if not isinstance(graph_t,list):graph_t = [graph_t]


    for j in range(batch_d_per_iter):
        wdiscriminator.train()

        w1s = []
        for graph in graph_t:
            w1s.append(wdiscriminator(graph.x.detach(),graph.edge_index))


        w0 = wdiscriminator(graph_s.x.detach(),graph_s.edge_index)
        w1 = torch.vstack(w1s)

        loss = -torch.mean(w1) + torch.mean(w0)
        #if j% 40 ==0:print(loss.item())

        optimizer_d.zero_grad()
        loss.backward()
        optimizer_d.step()

        for p in wdiscriminator.parameters(): p.data.clamp_(-0.01, 0.01)

    return wdiscriminator














def train_graph_isolate(MLPs,model, graph,optimizer,dims,device,file=None):

    for modal in graph.modal_names: MLPs[modal].to(device)
        

    value = 0
    count = 0
    result = []

    for epoch in range(1,1500):


        model.train()
        for modal in graph.modal_names: MLPs[modal].train()
        
        weights = OrderedDict(model.named_parameters())

        xs = []
        for modal in graph.modal_names:
            modal_feat = MLPs[modal](graph.original_feats[modal])
            modal_dim = int(np.sum(dims[:graph.modal_names.index(modal)]))
            xs.append(torch.cat([modal_feat[:,:modal_dim],graph.original_feats[modal],modal_feat[:,modal_dim:]],dim=1))

        xs = graph.complete_modalities_graph(xs[0],xs[1],xs[2])
        if model.task == 'link':
            x_text = model.encode(xs['text'], graph.train_edge_index, weights, inner_loop=True)
            x_vision = model.encode(xs['vision'], graph.train_edge_index, weights, inner_loop=True)
            x_structure = model.encode(xs['structure'], graph.train_edge_index, weights, inner_loop=True)

            
        else:
            x_text = model.encode(xs['text'], graph.edge_index, weights, inner_loop=True)
            x_vision = model.encode(xs['vision'], graph.edge_index, weights, inner_loop=True)
            x_structure = model.encode(xs['structure'], graph.edge_index, weights, inner_loop=True)


        if model.task == 'link':
            loss = model.recon_loss_isolate(x_text,x_vision,x_structure,graph.train_edge_index)
        else:
            loss = model.class_loss_isolate(x_text[graph.train_labels],x_vision[graph.train_labels],x_structure[graph.train_labels], graph.labels[graph.train_labels])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for p in model.parameters():
            p.data.clamp_(-0.1, 0.1)

        model.eval()
        for modal in graph.modal_names: MLPs[modal].eval()




        with torch.no_grad():
            weights = OrderedDict(model.named_parameters())
            xs = []
            for modal in graph.modal_names:
                modal_feat = MLPs[modal](graph.original_feats[modal])
                modal_dim = int(np.sum(dims[:graph.modal_names.index(modal)]))
                xs.append(torch.cat([modal_feat[:,:modal_dim],graph.original_feats[modal],modal_feat[:,modal_dim:]],dim=1))

            xs = graph.complete_modalities_graph(xs[0],xs[1],xs[2])
            if model.task == 'link':
                x_text = model.encode(xs['text'], graph.train_edge_index, weights, inner_loop=True)
                x_vision = model.encode(xs['vision'], graph.train_edge_index, weights, inner_loop=True)
                x_structure = model.encode(xs['structure'], graph.train_edge_index, weights, inner_loop=True)
            else:
                x_text = model.encode(xs['text'], graph.edge_index, weights, inner_loop=True)
                x_vision = model.encode(xs['vision'], graph.edge_index, weights, inner_loop=True)
                x_structure = model.encode(xs['structure'], graph.edge_index, weights, inner_loop=True)


            if model.task == 'link':score = model.test_isolate(x_text,x_vision,x_structure,graph.test_edge_index,graph.test_edge_index_negative )[0]
            else: 

                score = model.class_test_isolate(x_text[graph.test_labels],x_vision[graph.test_labels],x_structure[graph.test_labels],graph.labels[graph.test_labels])



        if epoch in [0,2,4,6,8,10,20,50,100,150,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400]:print(epoch,value,loss.item())
        
        #if epoch<=500:print(epoch,score,loss.item())
        if file!=None:file.write('{},{},{},{}\n'.format(epoch,count,score))
        if value<score:
            value = score
            count = 0
        else:count+=1
        result.append(value)
    if file!=None:file.write('the best value:,{}\n'.format(value))
    print('the best value: ', value)
    return result




def train_graph_final(MLPs,model, graph,optimizer,dims,device,file=None):

    for modal in graph.modal_names: MLPs[modal].to(device)
        

    value = 0
    count = 0
    best = 0
 

    for epoch in range(1,1500):


        model.train()
        for modal in graph.modal_names: MLPs[modal].train()
        
        weights = OrderedDict(model.named_parameters())

        xs = []
        for modal in graph.modal_names:
            modal_feat = MLPs[modal](graph.original_feats[modal])
            modal_dim = int(np.sum(dims[:graph.modal_names.index(modal)]))
            xs.append(torch.cat([modal_feat[:,:modal_dim],graph.original_feats[modal],modal_feat[:,modal_dim:]],dim=1))

        xs = graph.complete_modalities_graph(xs[0],xs[1],xs[2])
        xs = torch.cat(xs['text'],xs['vision'],xs['structure'],graph.anchor_nodes,dim = 0)
        embedding = model.encode(xs, graph.crossmodel_adj, weights)
        num = graph.original_feats['text'].shape[0]
        x_text = embedding[:num,:]
        x_vision = embedding[num:num*2,:]
        x_structure = embedding[num*2:num*3,:]

        if model.task == 'link':
            loss = model.recon_loss_isolate(x_text,x_vision,x_structure,graph.train_edge_index)
        else:
            loss = model.class_loss_isolate(x_text[graph.train_labels],x_vision[graph.train_labels],x_structure[graph.train_labels], graph.labels[graph.train_labels])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for p in model.parameters():
            p.data.clamp_(-0.1, 0.1)

        model.eval()
        for modal in graph.modal_names: MLPs[modal].eval()




        with torch.no_grad():
            weights = OrderedDict(model.named_parameters())
            xs = []
            for modal in graph.modal_names:
                modal_feat = MLPs[modal](graph.original_feats[modal])
                modal_dim = int(np.sum(dims[:graph.modal_names.index(modal)]))
                xs.append(torch.cat([modal_feat[:,:modal_dim],graph.original_feats[modal],modal_feat[:,modal_dim:]],dim=1))

            xs = graph.complete_modalities_graph(xs[0],xs[1],xs[2])
            xs = torch.cat(xs['text'],xs['vision'],xs['structure'],graph.anchor_nodes,dim = 0)
            embedding = model.encode(xs, graph.crossmodel_adj, weights)
            num = graph.original_feats['text'].shape[0]
            x_text = embedding[:num,:]
            x_vision = embedding[num:num*2,:]
            x_structure = embedding[num*2:num*3,:]


            if model.task == 'link':score = model.test_isolate(x_text,x_vision,x_structure,graph.val_edge_index,graph.val_edge_index_negative )[0]
            else: 
                score = model.class_test_isolate(x_text[graph.val_labels],x_vision[graph.val_labels],x_structure[graph.val_labels],graph.labels[graph.val_labels])[0]

            if model.task == 'link':test_score = model.test_isolate(x_text,x_vision,x_structure,graph.test_edge_index,graph.test_edge_index_negative )[0]
            else: 
                test_score = model.class_test_isolate(x_text[graph.test_labels],x_vision[graph.test_labels],x_structure[graph.test_labels],graph.labels[graph.test_labels])[0]



        if epoch in [0,2,4,6,8,10,20,50,100,150,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400]:print(epoch,value,loss.item())
        
        #if epoch<=500:print(epoch,score,loss.item())
        if file!=None:file.write('{},{},{},{}\n'.format(epoch,count,score))
        if value<score:
            value = score
            best = test_score
            count = 0
        else:count+=1

    if file!=None:file.write('the best value:,{}\n'.format(value))
    print('the best value: ', value)
    return best






def train_graph(MLPs,model, graph,optimizer,dims,device,file=None):

    for modal in graph.modal_names: MLPs[modal].to(device)
        

    value = 0
    count = 0
    result = []

    for epoch in range(1,1000):

        model.train()
        for modal in graph.modal_names: MLPs[modal].train()
        
        weights = OrderedDict(model.named_parameters())

        xs = []

        for modal in graph.modal_names:
            modal_feat = MLPs[modal](graph.original_feats[modal])

            modal_dim = int(np.sum(dims[:graph.modal_names.index(modal)]))
            xs.append(torch.cat([modal_feat[:,:modal_dim],graph.original_feats[modal],modal_feat[:,modal_dim:]],dim=1))

        feats = []
        for i in range(xs[0].shape[0]):
            if i in graph.modal_lists['text']:feats.append(xs[0][i])
            if i in graph.modal_lists['vision']:feats.append(xs[1][i])
            if i in graph.modal_lists['structure']:feats.append(xs[2][i])


        x = torch.stack(feats, dim=0).to(device)




        
        if model.task == 'link':z = model.encode(x, graph.extra_info["train_edge_index"], weights, inner_loop=True)
        else:z = model.encode(x, graph.edge_index, weights, inner_loop=True)

        if model.task == 'link':loss = model.recon_loss(z,graph.extra_info["train_edge_index"])
        else:loss = model.class_loss(z[graph.extra_info['train_idx'][0]], graph.extra_info['train_idx'][1])
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for p in model.parameters():
            p.data.clamp_(-0.1, 0.1)

        model.eval()
        for modal in graph.modal_names: MLPs[modal].eval()
        



        with torch.no_grad():
            weights = OrderedDict(model.named_parameters())
            xs = []
            for modal in graph.modal_names:
                modal_feat = MLPs[modal](graph.original_feats[modal])
                modal_dim = int(np.sum(dims[:graph.modal_names.index(modal)]))
                xs.append(torch.cat([modal_feat[:,:modal_dim],graph.original_feats[modal],modal_feat[:,modal_dim:]],dim=1))

            feats = []
            for i in range(xs[0].shape[0]):
                if i in graph.modal_lists['text']:feats.append(xs[0][i])
                if i in graph.modal_lists['vision']:feats.append(xs[1][i])
                if i in graph.modal_lists['structure']:feats.append(xs[2][i])


            x = torch.stack(feats, dim=0).to(device)


            
            
            if model.task == 'link':z = model.encode(x, graph.extra_info["train_edge_index"], weights, inner_loop=True)
            else:z = model.encode(x, graph.edge_index, weights, inner_loop=True)


            if model.task == 'link':score = model.test_reduce(z,graph.extra_info['test_map'],graph.extra_info['test_neg_map'] )[0]
            else: 
                score = model.class_test_reduce(z[graph.extra_info['test_idx'][1]], graph.extra_info['test_idx'][2],graph.extra_info['test_idx'][0])



        if epoch in [0,2,4,6,8,10,20,50,100,150,200,300,400,500,600,700,800,900,1000]:print(epoch,value,loss.item())
        
        #if epoch<=500:print(epoch,score,loss.item())
        if file!=None:file.write('{},{},{},{}\n'.format(epoch,count,score))
        if value<score:
            value = score
            count = 0
        else:count+=1
        result.append(value)
    if file!=None:file.write('the best value:,{}\n'.format(value))
    print('the best value: ', value)
    return result





    




def crossmodel_graph_split(fused_graph,graphs,task):

    
    

    if task == 'edge':
        cum = 0
        test_index = []

        for key in graphs.keys():  
            graph = graphs[key]
            
            graph.link_split(0.2, 0.2)
            test_index.append(cum+graph.test_edge_index)

            fused_graph.adj[cum:cum+graph.x.shape[0],cum:cum+graph.x.shape[0]] = 0.0
            fused_graph.adj[graph.train_edge_index[0]+cum,graph.train_edge_index[1]+cum] = 1.0

            cum+=graph.x.shape[0]

        fused_graph.test_index = test_index
        fused_graph.adj_tensor = fused_graph.adj
    
    elif task == 'nmk':
        
        cum = 0
        test_index = []
        for key in graphs.keys():
            graph = graphs[key]
            graph.attr_split2(0.2)
            test_index.append(graph.mask+cum)
            fused_graph.adj[cum:cum+graph.x.shape[0],cum:cum+graph.x.shape[0]] = 0.0
            fused_graph.adj[graph.edge_index[0]+cum,graph.edge_index[1]+cum] = 1.0
            cum+=graph.x.shape[0]

        
        fused_graph.test_index = test_index
        fused_graph.adj_tensor = fused_graph.adj
        
        fused_graph.xx = torch.cat([graphs[name].xx for name in graphs.keys()]+[fused_graph.super_nodes],dim =0)
        
    
    elif task == 'sim':


        cum = 0
        for key in graphs.keys():
            graph = graphs[key]
            fused_graph.adj[cum:cum+graph.x.shape[0],cum:cum+graph.x.shape[0]] = 0.0
            fused_graph.adj[graph.edge_index[0]+cum,graph.edge_index[1]+cum] = 1.0
            cum+=graph.x.shape[0]

        fused_graph.adj_tensor = fused_graph.adj


    else:
        
        all_edge_index = []
        
        cum = 0

        for key in graphs.keys():
            graph = graphs[key]

            all_edge_index.append(graph.edge_index+cum)

            fused_graph.adj[cum:cum+graph.x.shape[0],cum:cum+graph.x.shape[0]] = 0.0
            fused_graph.adj[graph.edge_index[0]+cum,graph.edge_index[1]+cum] = 1.0
            cum+=graph.x.shape[0]


        fused_graph.adj_tensor = fused_graph.adj
        fused_graph.original_adj = torch.cat(all_edge_index,dim=1)


        tmp = []
        node_nums = np.cumsum([0]+[graphs[name].x.shape[0] for name in graphs.keys()])
        for i in range(len(node_nums)-1):
            tmp.append(np.arange(node_nums[i],node_nums[i+1])+1)
            tmp[-1][-1] = node_nums[i]
        tmp.append(np.arange(node_nums[-1],fused_graph.x.shape[0]))
        arr = np.concatenate(tmp)
        
        fused_graph.arr = arr
    return fused_graph,graphs




def get_reco_loss(args, task, base_model, base_auxiliary,graphs,match_graph,device):

   
        

    adj = match_graph(graphs)

    fused_graph = Graph(None,None)
    fused_graph.adj = adj
    fused_graph.super_nodes = match_graph.super_nodes
    fused_graph,graphs = fused_split_new(fused_graph,graphs,task)



    models = reco_training_inner(args, task, base_model, base_auxiliary, fused_graph,graphs,device)
    fused_graph,graphs = fused_split_new(fused_graph,graphs,task)

  
    cum = 0
    loss = 0
    for index,key in enumerate(graphs.keys()):
        model=models[index]
        graph = graphs[key]
        weights = OrderedDict(model[0].named_parameters())
        if task == 'edge':
            z_val1 = model[0].encode(x = fused_graph.x, edge_index = None,
                                           weights = weights, edge_weight = fused_graph.adj_tensor)

            loss1 = model[0].recon_loss(z_val1, fused_graph.test_index[index])

            z_val2 = model[0].encode(graph.x.detach(), graph.train_edge_index, weights)
            loss2 = model[0].recon_loss(z_val2, graph.test_edge_index)


        elif task == 'sim':
            vice_model = gen_ran_output(model[0])

            embedding = fused_graph.x
            output_positive = model[0].encode(x = embedding, edge_index = None,weights = weights, edge_weight = fused_graph.adj_tensor)


            output_negative = vice_model.encode(x = embedding, edge_index = None,weights = weights, edge_weight = fused_graph.adj_tensor)
            output_negative = output_negative[cum:cum+graph.x.shape[0],:]

            loss1 = loss_cal2(output_positive,output_negative,model[1])


            output_positive = model[0].encode(graph.x, graph.edge_index, weights)
            output_negative = vice_model.encode(graph.x, graph.edge_index, weights)

            loss2 = loss_cal2(output_positive,output_negative,model[1])




        elif task == 'nmk':


            z_val = model[0].encode(x = fused_graph.xx, edge_index = None,
                                            weights = weights, edge_weight = fused_graph.adj_tensor)
            z_val1 = model[1](z_val)
            loss1 = torch.nn.MSELoss()(z_val1[fused_graph.test_index[index],int(np.sum(args.dims[:index])):int(np.sum(args.dims[:index]))+args.dims[index]], 
                                            graph.x[graph.mask,int(np.sum(args.dims[:index])):int(np.sum(args.dims[:index]))+args.dims[index]])
                    
            z_val = model[0].encode(graph.xx, graph.edge_index, weights)
            z_val2 = model[1](z_val)
            loss2 = torch.nn.MSELoss()(z_val2[graph.mask,int(np.sum(args.dims[:index])):int(np.sum(args.dims[:index]))+args.dims[index]], 
                                            graph.x[graph.mask,int(np.sum(args.dims[:index])):int(np.sum(args.dims[:index]))+args.dims[index]])

        else:
            embedding = fused_graph.x
            output_positive = model[0].encode(x = embedding, edge_index = None,weights = weights, edge_weight = fused_graph.adj_tensor)
            output_positive = output_positive[cum:cum+graph.x.shape[0],:]
                
            arr = np.arange(embedding.shape[0])
            arr[cum+graph.x.shape[0]-1]= arr[cum]-1
            arr[cum:cum+graph.x.shape[0]] = arr[cum:cum+graph.x.shape[0]] +1



            embedding = embedding[arr]

            output_negative = model[0].encode(x = embedding, edge_index = None,weights = weights, edge_weight = fused_graph.adj_tensor)
            output_negative = output_negative[cum:cum+graph.x.shape[0],:]
            summary_emb = torch.sigmoid(torch.mean(output_positive, dim=0, keepdim=True))
            discriminator_summary = model[1](summary_emb).T
            positive_score = output_positive @ discriminator_summary
            negative_score = output_negative @ discriminator_summary
            loss1 = torch.nn.BCEWithLogitsLoss()(positive_score,
                                                torch.ones_like(positive_score)) + torch.nn.BCEWithLogitsLoss()(
            negative_score, torch.zeros_like(negative_score))


            output_positive = model[0].encode(graph.x, graph.edge_index, weights)
            arr = torch.arange(graph.x.shape[0]) + 1
            arr[-1:] = torch.arange(1)
            output_negative = model[0].encode(graph.x[arr], graph.edge_index, weights)
            summary_emb = torch.sigmoid(torch.mean(output_positive, dim=0, keepdim=True))
            discriminator_summary = model[1](summary_emb).T
            positive_score = output_positive @ discriminator_summary
            negative_score = output_negative @ discriminator_summary
            loss2 = torch.nn.BCEWithLogitsLoss()(positive_score,
                                                torch.ones_like(positive_score)) + torch.nn.BCEWithLogitsLoss()(
            negative_score, torch.zeros_like(negative_score))

        cum+=graph.x.shape[0]
        loss+=(loss1+loss2)


    return loss





def reco_training_inner(args, task, base_model, base_auxiliary, fused_graph,graphs,device):
    
    models = []
    cum = 0
    for index,key in enumerate(graphs.keys()):
        graph = graphs[key]
        model = deepcopy(base_model),deepcopy(base_auxiliary)
        model_params = []
        model_params.extend(model[0].parameters())
        if model[1] is not None: model_params.extend(model[1].parameters())


        optimizer= torch.optim.Adam([{'params': model_params}], lr=0.001,weight_decay=5e-4)   ########args.model_lr  dgi:0.001
        
        for epoch in range(10): #20
            model[0].train()
            if model[1] is not None:model[1].train()

            weights = OrderedDict(model[0].named_parameters())

            if task == 'edge':
                z_val1 = model[0].encode(x = fused_graph.x, edge_index = None,
                                           weights = weights, edge_weight = fused_graph.adj_tensor.detach())

                loss1 = model[0].recon_loss(z_val1, fused_graph.test_index[index])

                z_val2 = model[0].encode(graph.x.detach(), graph.train_edge_index, weights)
                loss2 = model[0].recon_loss(z_val2, graph.test_edge_index)

            elif task == 'sim':


                
                vice_model = gen_ran_output(model[0])

                embedding = fused_graph.x
                output_positive = model[0].encode(x = embedding, edge_index = None,weights = weights, edge_weight = fused_graph.adj_tensor.detach())
                #output_positive = output_positive[cum:cum+graph.x.shape[0],:]

                output_negative = vice_model.encode(x = embedding, edge_index = None,weights = weights, edge_weight = fused_graph.adj_tensor.detach())
                #output_negative = output_negative[cum:cum+graph.x.shape[0],:]

                loss1 = loss_cal2(output_positive,output_negative,model[1])


                output_positive = model[0].encode(graph.x, graph.edge_index, weights)
                output_negative = vice_model.encode(graph.x, graph.edge_index, weights)

                loss2 = loss_cal2(output_positive,output_negative,model[1])
                

            elif task == 'nmk':




                z_val = model[0].encode(x = fused_graph.xx, edge_index = None,
                                            weights = weights, edge_weight = fused_graph.adj_tensor.detach())
                z_val1 = model[1](z_val)
                loss1 = torch.nn.MSELoss()(z_val1[fused_graph.test_index[index],int(np.sum(args.dims[:index])):int(np.sum(args.dims[:index]))+args.dims[index]], 
                                            graph.x[graph.mask,int(np.sum(args.dims[:index])):int(np.sum(args.dims[:index]))+args.dims[index]])
                    
                z_val = model[0].encode(graph.xx, graph.edge_index, weights)
                z_val2 = model[1](z_val)
                loss2 = torch.nn.MSELoss()(z_val2[graph.mask,int(np.sum(args.dims[:index])):int(np.sum(args.dims[:index]))+args.dims[index]], 
                                            graph.x[graph.mask,int(np.sum(args.dims[:index])):int(np.sum(args.dims[:index]))+args.dims[index]])

            else:
                embedding = fused_graph.x
                output_positive = model[0].encode(x = embedding, edge_index = None,weights = weights, edge_weight = fused_graph.adj_tensor.detach())
                output_positive = output_positive[cum:cum+graph.x.shape[0],:]

                arr = np.arange(embedding.shape[0])
                arr[cum+graph.x.shape[0]-1]= arr[cum]-1
                arr[cum:cum+graph.x.shape[0]] = arr[cum:cum+graph.x.shape[0]] +1
                

                #arr = fused_graph.arr

                embedding = embedding[arr]

                output_negative = model[0].encode(x = embedding, edge_index = None,weights = weights, edge_weight = fused_graph.adj_tensor.detach())
                output_negative = output_negative[cum:cum+graph.x.shape[0],:]
                summary_emb = torch.sigmoid(torch.mean(output_positive, dim=0, keepdim=True))
                discriminator_summary = model[1](summary_emb).T
                positive_score = output_positive @ discriminator_summary
                negative_score = output_negative @ discriminator_summary
                loss1 = torch.nn.BCEWithLogitsLoss()(positive_score,
                                                torch.ones_like(positive_score)) + torch.nn.BCEWithLogitsLoss()(
                negative_score, torch.zeros_like(negative_score))


                output_positive = model[0].encode(graph.x, graph.edge_index, weights)
                arr = torch.arange(graph.x.shape[0]) + 1
                arr[-1:] = torch.arange(1)
                output_negative = model[0].encode(graph.x[arr], graph.edge_index, weights)
                summary_emb = torch.sigmoid(torch.mean(output_positive, dim=0, keepdim=True))
                discriminator_summary = model[1](summary_emb).T
                positive_score = output_positive @ discriminator_summary
                negative_score = output_negative @ discriminator_summary
                loss2 = torch.nn.BCEWithLogitsLoss()(positive_score,
                                                torch.ones_like(positive_score)) + torch.nn.BCEWithLogitsLoss()(
                negative_score, torch.zeros_like(negative_score))

            
            loss = loss1+loss2
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        print(loss1,loss2)
        cum+=graph.x.shape[0]


        for p in model[0].parameters(): p.requires_grad = False
        if model[1] is not None: 
            for p in model[1].parameters(): p.requires_grad = False
        models.append(model)


        

    return models





def MGSHIP_gradient(args, task, pretrain_model, auxiliary, graphs, optimizer,MLPs,wdiscriminator,optimizer_wd,match_graph, epoch, device, file):

    torch.autograd.set_detect_anomaly(True)
    

    adj = match_graph(graphs)

    crossmodel_graph = Graph(None,None)
    crossmodel_graph.adj = adj
    crossmodel_graph.super_nodes = match_graph.super_nodes
    crossmodel_graph.x = torch.cat([graphs[name].x.detach() for name in args.modal_names]+[match_graph.super_nodes],dim =0) 

    crossmodel_graph,graphs = crossmodel_graph_split(crossmodel_graph,graphs,task)

    reco_losses = get_reco_loss(args, task, pretrain_model, auxiliary,graphs,match_graph,device)

    task_losses = []

    

    pretrain_model.train()
    if auxiliary: auxiliary.train()


    for index,name in enumerate(args.modals):
        graphs[name] = graphs[name].transformation(MLPs[name],int(np.sum(args.dims[:index])))
        

    graphs['crossmodel'] = crossmodel_graph
    graphs['crossmodel'].adj = match_graph.get_final_adj(graphs['crossmodel'].adj,args.threshold)
    if task == 'nmk':graphs['crossmodel'].original_x = graphs['crossmodel'].x.clone().detach()


    print(graphs['crossmodel'].x.shape)
    print(graphs['crossmodel'].edge_index.shape)


    weights = OrderedDict(pretrain_model.named_parameters())


    

    task_losses = []
    dis_losses = []

    for index,name in enumerate( graphs.keys()):

        Dis_loss = torch.tensor(0.0).to(device)
        graph= graphs[name]
        if name!='crossmodel':
            

            wdiscriminator_copy = copy.deepcopy(
                train_wdiscriminator(graph, graphs['crossmodel'], wdiscriminator,
                                     optimizer_wd, batch_d_per_iter=80))

            for p in wdiscriminator_copy.parameters(): p.requires_grad = False
            wdiscriminator_copy.to(device)


            w1 = wdiscriminator_copy( graphs['crossmodel'].x, graphs['crossmodel'].edge_index)
            w0 = wdiscriminator_copy(graph.x, graph.edge_index)

            Dis_loss = (torch.mean(w1) - torch.mean(w0))

        



        if task == 'edge':

            z_val = pretrain_model.encode(graph.x, graph.train_edge_index, weights)
            loss = pretrain_model.recon_loss(z_val, graph.test_edge_index)
            

        elif task == 'nmk':

            z_val = pretrain_model.encode(graph.x, graph.edge_index, weights)
            z_val = auxiliary(z_val)
            if name == 'crossmodel':
                
                for ind,k in enumerate(args.modals):
                    z_val[ind*args.fuse_scale:(1+ind)*args.fuse_scale,0:int(np.sum(args.dims[:ind]))]=0
                    z_val[ind*args.fuse_scale:(1+ind)*args.fuse_scale,args.dims[ind]+int(np.sum(args.dims[:ind])):]=0
                    graph.original_x[ind*args.fuse_scale:(1+ind)*args.fuse_scale,0:int(np.sum(args.dims[:ind]))]=0
                    graph.original_x[ind*args.fuse_scale:(1+ind)*args.fuse_scale,args.dims[ind]+int(np.sum(args.dims[:ind])):]=0

                    
                loss = torch.nn.MSELoss()(z_val[graph.mask], graph.original_x[graph.mask])
            else:loss = torch.nn.MSELoss()(z_val[graph.mask,int(np.sum(args.dims[:index])):int(np.sum(args.dims[:index]))+args.dims[index]], graph.original_x[graph.mask])

            
        elif task == 'sim':
            if name == 'crossmodel': 
                vice_model = gen_ran_output(pretrain_model)
                output_positive = pretrain_model.encode(graph.x, graph.edge_index, weights)[:graph.num]
                output_negative = vice_model.encode(graph.x, graph.edge_index, weights)[:graph.num]
                loss = loss_cal2(output_positive,output_negative,auxiliary)
            else:
                vice_model = gen_ran_output(pretrain_model)
                output_positive = pretrain_model.encode(graph.x, graph.edge_index, weights)
                output_negative = vice_model.encode(graph.x, graph.edge_index, weights)
                loss = loss_cal2(output_positive,output_negative,auxiliary)
    
        else:
            


            if name == 'crossmodel':
                arr = torch.arange(graph.x.shape[0])#graph.arr
                output_positive = pretrain_model.encode(graph.x, graph.original_adj, weights)[:graph.num]
                output_negative = pretrain_model.encode(graph.x[arr], graph.edge_index, weights)[:graph.num]
            else:
                output_positive = pretrain_model.encode(graph.x, graph.edge_index, weights)
            

                arr = torch.arange(graph.x.shape[0]) + 1
                arr[-1:] = torch.arange(1)
                output_negative = pretrain_model.encode(graph.x[arr], graph.edge_index, weights)


            
            


            summary_emb = torch.sigmoid(torch.mean(output_positive, dim=0, keepdim=True))
            discriminator_summary = auxiliary(summary_emb).T
            positive_score = output_positive @ discriminator_summary
            negative_score = output_negative @ discriminator_summary


            if name == 'crossmodel': loss = (torch.nn.BCEWithLogitsLoss()(positive_score, #0.1
                                                torch.ones_like(positive_score)) + torch.nn.BCEWithLogitsLoss()(
            negative_score, torch.zeros_like(negative_score)))
            else:loss = torch.nn.BCEWithLogitsLoss()(positive_score,
                                                torch.ones_like(positive_score)) + torch.nn.BCEWithLogitsLoss()(
            negative_score, torch.zeros_like(negative_score))


            
        dis_losses.append(Dis_loss)
        task_losses.append(loss)
        print(Dis_loss.item(),loss.item())

    file.write('{},{},{},{},{}\n'.format(epoch, '-'.join(args.modals),','.join([str(round(lo.item(),4)) for lo in reco_losses]), ','.join([str(round(lo.item(),4)) for lo in task_losses]),','.join([str(round(lo.item(),4)) for lo in dis_losses[:4]])))
    file.flush()


    if len(task_losses) != 0:
        
        optimizer.zero_grad()
        pretrain_batch_loss = torch.stack(task_losses).mean() + args.alphs* torch.stack(reco_losses).mean() * args.beta+torch.stack(dis_losses).mean() * args.beta
        pretrain_batch_loss.backward()

        optimizer.step()

    for p in pretrain_model.parameters():
        p.data.clamp_(-0.1, 0.1)

    return pretrain_model, pretrain_batch_loss.item()
