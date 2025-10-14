
from collections import defaultdict
import torch
from torch_geometric.data import Data, DataLoader
import numpy as np
import torch_geometric.transforms as T
from sklearn.preprocessing import StandardScaler
import scipy.sparse as sp
import os.path
from pathlib import Path
from random import randint
import random
import math
from torch_geometric.utils import to_undirected,subgraph
from models.models import MLP




def negative_sampling_identify(pos_edge_index, num_nodes):

    def fetch(arr,num,r = 0):
        d = int((len(arr)+0.0)/num)
        result = list(range(r,len(arr),d))
        if len(result)>num:result = result[:num]
        return result


    idx = (pos_edge_index[0] * num_nodes + pos_edge_index[1]) 
    idx = idx.to(torch.device('cpu'))

    rng =  [i * num_nodes+j for i in range(num_nodes) for j in range(num_nodes) if i<j ]  #range(num_nodes**2)
    perm = torch.tensor(fetch(rng, idx.size(0)))
    mask = torch.from_numpy(np.isin(perm, idx).astype(np.uint8))
    rest = mask.nonzero(as_tuple = False).view(-1)

    index = 0
    while rest.numel() > 0:  # pragma: no cover
        tmp = torch.tensor(fetch(rng, rest.size(0),index))
        perm[rest] = tmp
        mask = torch.from_numpy(np.isin(perm, idx).astype(np.uint8))
        rest = mask.nonzero(as_tuple = False).view(-1)
        index+=1

    row, col = torch.div(perm, num_nodes,rounding_mode='floor'), perm % num_nodes   




    return torch.stack([row, col], dim=0).to(pos_edge_index.device)





def load_test_graph(dataset,index,text,vision,structure,norm = True):
    dataset_str = '/data/lmk/mm-graph/' + dataset + '/test_graphs/'
    file = dataset_str+'test_graph{}.npz'.format(index)
    file = np.load(file,allow_pickle=True)
    text_feat = file['feats_text'].item()[text] 
    vision_feat = file['feats_vision'].item()[vision] 
    structure_feat = file['feats_structure'].item()[structure] 

    if norm:
        scaler = StandardScaler()
        scaler.fit(text_feat)
        text_feat = scaler.transform(text_feat)

        scaler = StandardScaler()
        scaler.fit(vision_feat)
        vision_feat = scaler.transform(vision_feat)

        scaler = StandardScaler()
        scaler.fit(structure_feat)
        structure_feat = scaler.transform(structure_feat)



    edge_index = file['edge_index']
    labels = file['labels']
    graph = test_graph(text_feat = text_feat,vision_feat=vision_feat,structure_feat=structure_feat,edge_index=edge_index,labels=labels)
    return graph



def load_test_graph_share(dataset,index,num):
    dataset_str = '/home/lmk/LAMP-GNN/lamp_data/test/' + dataset + '/'
    file = dataset_str+'test_graph{}.npz'.format(index)
    file = np.load(file)
    feats = file['feats']

    mlp = MLP(feats.shape[1], 128, 128)
    mlp.load_state_dict(torch.load('./result/autoencoder_for_{}_{}.pth'.format(dataset,num)), strict=True)


    feats = mlp(torch.from_numpy(feats).to(torch.float)).cpu().detach().numpy()

    scaler = StandardScaler()
    scaler.fit(feats)
    feats = scaler.transform(feats)

    edge_index = file['edge_index']
    labels = file['labels']
    graph = Graph(x = feats,edge_index=edge_index,is_tensor=False,labels=labels)
    return graph








class test_graph:
    def __init__(self,text_feat,vision_feat,structure_feat,edge_index,labels = None,is_tensor = False):
        super(test_graph, self).__init__()
        self.text_feat = text_feat
        self.vision_feat = vision_feat
        self.structure_feat = structure_feat
        self.edge_index = edge_index
        self.labels = labels
        self.is_tensor = is_tensor



    def get_sub_edge_index(self,edge_index, list_sample, relabel_nodes=True):
        
        if not torch.is_tensor(list_sample):
            list_sample = torch.tensor(list_sample, dtype=torch.long)

        sub_edge_index, _ = subgraph(
            subset=list_sample,
            edge_index=edge_index,
            relabel_nodes=relabel_nodes
        )

        node_mapping = None
        if relabel_nodes:
            node_mapping = list_sample   #{old.item(): new for new, old in enumerate(list_sample)}

        return sub_edge_index, node_mapping

    def get_concat_graph(self,device):
        text_feat = np.ones_like(self.text_feat) #np.random.normal(loc=0, scale=1, size=self.text_feat.shape) #  
        vision_feat = np.ones_like(self.vision_feat) #np.random.normal(loc=0, scale=1, size=self.vision_feat.shape) # 
        structure_feat = np.ones_like(self.structure_feat) #np.random.normal(loc=0, scale=1, size=self.structure_feat.shape) # 

        text_feat[self.text_list] = self.text_feat[self.text_list]
        vision_feat[self.vision_list] = self.vision_feat[self.vision_list]
        structure_feat[self.structure_list] = self.structure_feat[self.structure_list]

        feats = np.concatenate([text_feat,vision_feat,structure_feat],axis = 1)
        return Graph(feats,self.edge_index,False,self.labels)
    

    def get_incomplete_features(self,):
        text_feat = np.zeros_like(self.text_feat)
        vision_feat = np.zeros_like(self.vision_feat)
        structure_feat = np.zeros_like(self.structure_feat)

        text_feat[self.text_list] = self.text_feat[self.text_list]
        vision_feat[self.vision_list] = self.vision_feat[self.vision_list]
        structure_feat[self.structure_list] = self.structure_feat[self.structure_list]

        self.text_feat = text_feat
        self.vision_feat = vision_feat
        self.structure_feat = structure_feat

 






    def to_tensor(self,device):
        self.device = device

        if self.is_tensor:return


        edge_index = torch.from_numpy(self.edge_index).to(device)

        self.text_feat = torch.from_numpy(self.text_feat).to(torch.float).to(device)
        self.vision_feat = torch.from_numpy(self.vision_feat).to(torch.float).to(device)
        self.structure_feat = torch.from_numpy(self.structure_feat).to(torch.float).to(device)


        self.edge_index = edge_index
        self.is_tensor = True


        if not self.labels is None:self.labels = torch.LongTensor(self.labels).to(device)
    
    
    





    def divide_graph(self,device):

        edge_index_text, self.node_mapping_text = self.get_sub_edge_index(self.edge_index,self.text_list,True)
        self.graph_text = Graph(self.text_feat[self.text_list],edge_index_text).to(device)

        edge_index_vision, self.node_mapping_vision = self.get_sub_edge_index(self.edge_index,self.vision_list,True)
        self.graph_vision = Graph(self.vision_feat[self.vision_list],edge_index_vision).to(device)


        edge_index_structure, self.node_mapping_structure = self.get_sub_edge_index(self.edge_index,self.structure_list,True)
        self.graph_structure = Graph(self.structure_feat[self.structure_list],edge_index_structure).to(device)


    def complete_modalities_graph(self,text_feats, vision_feats, structure_feats):
        
        N, d = text_feats.shape

        text_list, vision_list, structure_list = self.text_list,self.vision_list,self.structure_list


        mask = torch.zeros((N, 3), dtype=torch.float32, device=text_feats.device)
        mask[text_list, 0] = 1
        mask[vision_list, 1] = 1
        mask[structure_list, 2] = 1


        feats = torch.stack([text_feats, vision_feats, structure_feats], dim=1)

        sum_feats = feats * mask.unsqueeze(-1)          
        sum_feats = sum_feats.sum(dim=1, keepdim=True)  


        count = mask.sum(dim=1, keepdim=True).clamp(min=1).unsqueeze(-1)


        mean_feats = sum_feats / count


        mask_expand = mask.unsqueeze(-1)  # (N,3,1)
        feats_completed = feats * mask_expand + mean_feats * (1 - mask_expand)


        text_feats_new, vision_feats_new, structure_feats_new = feats_completed.unbind(dim=1)

        return {'text':text_feats_new,'vision':vision_feats_new,'structure':structure_feats_new} 







    def multimodal_to_isolate_graph(self,convert_text,convert_vision,convert_structure ):
        
        text_set = set(self.text_list)
        vision_set = set(self.vision_list)
        structure_set = set(self.structure_list)
        N, d = convert_text.shape


        texts,visions,structures = [],[],[]
        for i in range(N):
            has_text = i in text_set
            has_vision = i in vision_set
            has_structure = i in structure_set

            if has_text + has_vision + has_structure == 1:
                
                if has_text:
                    visions.append(convert_text[i])
                    structures.append(convert_text[i])
                    texts.append(convert_text[i])
                elif has_vision:
                    texts.append(convert_vision[i])
                    structures.append(convert_vision[i])
                    visions.append(convert_vision[i])
                elif has_structure:
                    texts.append(convert_structure[i])
                    visions.append(convert_structure[i])
                    structures.append(convert_structure[i])
            
            elif has_text + has_vision + has_structure == 2:
                
                if not has_text:
                    texts.append((convert_vision[i] + convert_structure[i]) / 2)
                    visions.append(convert_vision[i])
                    structures.append(convert_structure[i])
                    
                if not has_vision:
                    visions.append((convert_text[i] + convert_structure[i]) / 2)
                    structures.append(convert_structure[i])
                    texts.append(convert_text[i])
                if not has_structure:
                    structures.append((convert_text[i] + convert_vision[i]) / 2)
                    texts.append(convert_text[i])
                    visions.append(convert_vision[i])

        text_feats = torch.stack(texts,dim=0)
        vision_feats = torch.stack(visions,dim=0)
        structure_feats = torch.stack(structures,dim=0)


        return {'text':text_feats,'vision':vision_feats,'structure':structure_feats}

                


    def multimodal_to_flat_graph(self,task ):
        
        
        device = self.edge_index.device
        edge_index = self.edge_index
        text_feats = self.text_feat
        vision_feats = self.vision_feat
        structure_feats = self.structure_feat
        text_list = self.text_list
        vision_list = self.vision_list
        structure_list = self.structure_list
        labels = self.labels
        if task == 'node':
            train_idx = self.train_labels
            val_idx = self.val_labels
            test_idx = self.test_labels
        else:
            train_edge_index = self.train_edge_index
            val_edge_index = self.val_edge_index
            test_edge_index = self.test_edge_index
            test_edge_index_negative = self.test_edge_index_negative
            

        N, d = text_feats.shape
        feats = []
        modality_map = {}  # (node, modality) -> new_node_id
        node2modalities = {}  # node_id -> [new_node_ids]
        cur_id = 0


        for i in range(N):
            node2modalities[i] = []
            if i in text_list:
                feats.append(text_feats[i])
                modality_map[(i, "text")] = cur_id
                node2modalities[i].append(cur_id)
                cur_id += 1
            if i in vision_list:
                feats.append(vision_feats[i])
                modality_map[(i, "vision")] = cur_id
                node2modalities[i].append(cur_id)
                cur_id += 1
            if i in structure_list:
                feats.append(structure_feats[i])
                modality_map[(i, "structure")] = cur_id
                node2modalities[i].append(cur_id)
                cur_id += 1

        feats = torch.stack(feats, dim=0).to(device)

        edge_set = set()

        for i in range(N):
            mods = node2modalities[i]
            for j in range(len(mods)):
                for k in range(j+1, len(mods)):
                    u, v = mods[j], mods[k]
                    edge_set.add((u, v))
                    edge_set.add((v, u))


        E = edge_index.shape[1]
        for e in range(E):
            u, v = edge_index[0, e].item(), edge_index[1, e].item()
            for mu in node2modalities[u]:
                for mv in node2modalities[v]:
                    edge_set.add((mu, mv))
                    edge_set.add((mv, mu))

        edge_index_new = torch.LongTensor(list(edge_set)).t().contiguous().to(device)

        new_data = Graph(x=feats, edge_index=edge_index_new,is_tensor=True,labels=labels)

        new_data.original_feats =self.original_feats
        new_data.modal_lists = {'text':text_list,'vision':vision_list,'structure':structure_list}



        extra_info = {}

        if task == "node":

            train_nodes = [[],[]]
            
            for i in train_idx:
                train_nodes[0].extend(node2modalities[i])
                train_nodes[1].extend([labels[i]]*len(node2modalities[i]))
            extra_info["train_idx"] =[np.array(train_nodes[0]),torch.LongTensor(train_nodes[1]).to(device)]


            val_nodes = [[],[],[]]
            for index,i in enumerate(val_idx):
                val_nodes[0].extend([index]*len(node2modalities[i]))        	
                val_nodes[1].extend(node2modalities[i])
                val_nodes[2].append(labels[i])
            extra_info["val_idx"] =[torch.LongTensor(val_nodes[0]).to(device),np.array(val_nodes[1]),torch.LongTensor(val_nodes[2]).to(device)]
            
            test_nodes = [[],[],[]]
            for index,i in enumerate(test_idx):
                test_nodes[0].extend([index]*len(node2modalities[i]))        	
                test_nodes[1].extend(node2modalities[i])
                test_nodes[2].append(labels[i])

            
            extra_info["test_idx"] =[torch.LongTensor(test_nodes[0]).to(device),np.array(test_nodes[1]),torch.LongTensor(test_nodes[2]).to(device)]
            #print(extra_info["test_idx"][0].shape,extra_info["test_idx"][1].shape,extra_info["test_idx"][2].shape)
            

        elif task == "link":

            train_edges = []
            for e in range(train_edge_index.shape[1]):
                u, v = train_edge_index[0, e].item(), train_edge_index[1, e].item()
                train_edges = [(mu, mv) for mu in node2modalities[u] for mv in node2modalities[v]]
            for i in range(N):
                mods = node2modalities[i]
                for j in range(len(mods)):
                    for k in range(j+1, len(mods)):
                        u, v = mods[j], mods[k]
                        train_edges.append((u, v))
                        train_edges.append((v, u))
            extra_info["train_edge_index"] = torch.LongTensor(train_edges).t().contiguous().to(device)


            def edge_map(edge_index,device):
                res = [[],[]]
                for e in range(edge_index.shape[1]):
                    u, v = edge_index[0, e].item(), edge_index[1, e].item()
                    edge_list = [(mu, mv) for mu in node2modalities[u] for mv in node2modalities[v]]
                    res[0].extend([e]*len(edge_list))
                    res[1].extend(edge_list)
                return [torch.LongTensor(res[0]).to(device),torch.LongTensor(res[1]).t().contiguous().to(device)]

            extra_info["val_map"] = edge_map(val_edge_index,device)
            extra_info["test_map"] = edge_map(test_edge_index,device)
            extra_info["test_neg_map"] = edge_map(test_edge_index_negative,device)
        new_data.extra_info = extra_info

        return new_data



















    def aggregate_values(self,embeddings_list):
        emb_dim = embeddings_list.size(1)
        sum_emb = torch.zeros((self.text_feat.shape[0] , emb_dim), dtype=torch.float)
        count_emb = torch.zeros((self.text_feat.shape[0],), dtype=torch.float)
        for value,index in zip(embeddings_list[:len(self.node_mapping_text)],self.node_mapping_text):
            sum_emb[index]+=value
            count_emb[index] += 1
        for value,index in zip(embeddings_list[len(self.node_mapping_text):len(self.node_mapping_text)+len(self.node_mapping_vision)],self.node_mapping_vision):
            sum_emb[index]+=value
            count_emb[index] += 1
        for value,index in zip(embeddings_list[len(self.node_mapping_text)+len(self.node_mapping_vision):len(self.node_mapping_text)+len(self.node_mapping_vision)+len(self.node_mapping_structure)],self.node_mapping_structure):
            sum_emb[index]+=value
            count_emb[index] += 1
        final_emb = sum_emb / count_emb.unsqueeze(1)

        return final_emb






    
    

    def split_modal_nodes(self, modal_ratio):

        if modal_ratio == 1:
            N = self.text_feat.shape[0]
            self.text_list, self.vision_list, self.structure_list =  list(range(N)), list(range(N)), list(range(N))
            return 



        import math

        N = self.text_feat.shape[0]
        r = modal_ratio
        M = math.ceil(r * N)  

        modal1 = set()
        modal2 = set()
        modal3 = set()

        node_modal_map = {i: set() for i in range(N)}

        for i in range(N):
            m = i % 3
            if m == 0:
                modal1.add(i)
                node_modal_map[i].add(1)
            elif m == 1:
                modal2.add(i)
                node_modal_map[i].add(2)
            else:
                modal3.add(i)
                node_modal_map[i].add(3)

        for i in range(N):
            current_modals = node_modal_map[i]
            if len(current_modals) >= 2:
                continue
            current_modal = list(current_modals)[0]
            candidates = {1, 2, 3} - current_modals
            candidate_list = sorted(list(candidates), key=lambda x: [len(modal1), len(modal2), len(modal3)][x-1])
            for c in candidate_list:
                modal_set = modal1 if c == 1 else modal2 if c == 2 else modal3
                if len(modal_set) < M:
                    modal_set.add(i)
                    node_modal_map[i].add(c)
                    break

        def fill_modal(modal_set, modal_idx):
            while len(modal_set) < M:
                for i in range(N):
                    if modal_idx in node_modal_map[i]:
                        continue
                    if len(node_modal_map[i]) < 2:
                        modal_set.add(i)
                        node_modal_map[i].add(modal_idx)
                        break
                else:
                    for i in range(N):
                        if modal_idx not in node_modal_map[i]:
                            modal_set.add(i)
                            node_modal_map[i].add(modal_idx)
                            break

        fill_modal(modal1, 1)
        fill_modal(modal2, 2)
        fill_modal(modal3, 3)

        modal1_list = sorted(list(modal1))
        modal2_list = sorted(list(modal2))
        modal3_list = sorted(list(modal3))

        self.text_list, self.vision_list, self.structure_list =  modal1_list, modal2_list, modal3_list
    





    def identify(self,task,train_ratio):




        if task == 'link':
            if self.is_tensor:
                edge_index = self.edge_index.detach()

            row = edge_index[0]
            col = edge_index[1]

            mask = row < col
            row, col = row[mask], col[mask]

            tmp = int(1/train_ratio)
            train_set = []
            tmp_set = []
            val_set = []
            test_set = []

            for i in range(row.size(0)):
                if i%tmp == 0:train_set.append(i)
                else:tmp_set.append(i)

            tmp = int((len(tmp_set)+0.0)/int(0.1*row.size(0)))
            for i,ele in enumerate(tmp_set):
                if i%tmp == 0:val_set.append(ele)
                else:test_set.append(ele)
            

            r, c = row[train_set], col[train_set]
            train_pos_edge_index = to_undirected(torch.stack([r, c], dim=0))
            r, c = row[val_set], col[val_set]
            val_pos_edge_index = to_undirected(torch.stack([r, c], dim=0))
            r, c = row[test_set], col[test_set]
            test_pos_edge_index = torch.stack([r, c], dim=0) #to_undirected(torch.stack([r, c], dim=0))

            train_pos_edge_index = train_pos_edge_index.detach()
            test_pos_edge_index = test_pos_edge_index.detach()
            val_pos_edge_index = val_pos_edge_index.detach()

            self.train_edge_index = train_pos_edge_index
            self.test_edge_index = test_pos_edge_index
            self.val_edge_index = val_pos_edge_index
            self.test_edge_index_negative = negative_sampling_identify(self.test_edge_index,self.text_feat.size(0))

        else:
            
            train_num = int(self.labels.size(0)*train_ratio)
            val_num = int(self.labels.size(0)*0.1)

            
            num_classes = int(self.labels.max().item()+1)

            label_arr = defaultdict(list)
            for index,label in enumerate(self.labels):
                label_arr[label.item()].append(index)

            #for i in range(num_classes):print(len(label_arr[i]))
            labels = []
            while len(label_arr)!=0:
                for key in sorted(label_arr.keys()):
                    if len(label_arr[key]) == 0:
                        label_arr.pop(key)
                        continue
                    labels.append(label_arr[key].pop())
                

            train_set = labels[:train_num]
            val_set = labels[train_num:train_num+val_num]
            test_set = labels[train_num+val_num:]


            self.train_labels=train_set
            self.test_labels = test_set
            self.val_labels = val_set




            print('train_labels:',len(self.train_labels),'  val_labels:',len(self.val_labels),' test_labels:',len(self.test_labels))
















class Graph:
    def __init__(self,x,edge_index,is_tensor = False,labels = None):
        super(Graph, self).__init__()
        self.x = x
        self.edge_index = edge_index
        self.is_tensor = is_tensor
        self.labels = labels


    def to(self,device):
        self.x = self.x.to(device)
        self.edge_index = self.edge_index.to(device)
        self.is_tensor = True

        
    def to_tensor(self,device):

        if self.is_tensor:return


        edge_index = torch.from_numpy(self.edge_index).to(device)

        x = torch.from_numpy(self.x).to(torch.float).to(device)

        self.x = x
        self.edge_index = edge_index
        self.is_tensor = True


        if not self.labels is None:
            if len(self.labels.shape) == 2:self.labels = torch.FloatTensor(self.labels).to(device)
            else: self.labels = torch.LongTensor(self.labels).to(device)



    def detach(self):
        return Graph(self.x.detach(),self.edge_index,True)


    def transformation(self,MLP,num):

        x = MLP(self.x)
        #print(x[:,:num].shape,self.x.shape,x[:,num:].shape)
        x = torch.cat([x[:,:num],self.x,x[:,num:]],dim=1)
        graph = Graph(x,self.edge_index,True)
        graph.original_x = self.x
        graph.adj = self.adj
        return graph
    

    def transformation_mlp(self,MLP):

        x = MLP(self.x)
        graph = Graph(x,self.edge_index,True)
        graph.original_x = self.x
        graph.adj = self.adj
        return graph


    def promptamation(self,MLPs,train_dataset,name):

        xs = []
        for data in train_dataset:
            if data == name:xs.append(self.x)
            else: xs.append(MLPs[data](self.x))
        x = torch.cat(xs,dim = 1)

        return Graph(x,self.edge_index,True)



        


    def node_split(self,label_num,instance_num):
        '''
        permutation = np.random.permutation(self.labels.shape[1])
        self.indices = np.random.choice(self.labels.shape[0],label_num,replace = False)
        self.indices.sort()
        labels = self.labels[:,permutation]
        self.train_nodes = np.unique(labels[self.indices,:instance_num].flatten())
        self.test_nodes = np.setdiff1d(np.unique(labels[self.indices].flatten()),self.train_nodes)
        '''
        self.indices = np.random.choice(list(self.labels.keys()),label_num,replace = False)
        self.indices.sort()
        self.train_nodes =[]
        for i in self.indices:self.train_nodes.extend(np.random.choice(np.setdiff1d(self.labels[i],self.indices),instance_num,replace=False))
        self.train_nodes = np.unique(self.train_nodes)
        self.test_nodes = []
        for i in self.indices:self.test_nodes.extend(self.labels[i])


        self.test_nodes = np.setdiff1d(np.unique(self.test_nodes),self.train_nodes)
        self.test_nodes = np.setdiff1d(self.test_nodes,self.indices)
        #if len(self.test_nodes)>5*len(self.train_nodes):self.test_nodes = np.random.choice(self.test_nodes,3*len(self.train_nodes),replace=False)

        #print(len(self.train_nodes),len(self.test_nodes))


        train_pos_link = []
        train_neg_link = []
        test_pos_link = []
        test_neg_link = []
        train_nodes = self.train_nodes
        test_nodes = self.test_nodes
        for i in train_nodes:
            for j in self.indices:
                if i in self.labels[j]:train_pos_link.append((i,j))
                else:train_neg_link.append((i,j))


        ground_truth = []
        for i in test_nodes:
            ground_truth.append([])
            for j in self.indices:
                if i in self.labels[j]:
                    test_pos_link.append((i,j))
                    ground_truth[-1].append(1)
                else:
                    test_neg_link.append((i,j))
                    ground_truth[-1].append(0)

        self.ground_truth = np.array(ground_truth)


        self.train_pos_edge_index = torch.tensor(np.array(train_pos_link).T).to(self.edge_index)
        self.train_neg_edge_index = torch.tensor(np.array(train_neg_link).T).to(self.edge_index)
        self.test_pos_edge_index = torch.tensor(np.array(test_pos_link).T).to(self.edge_index)
        self.test_neg_edge_index = torch.tensor(np.array(test_neg_link).T).to(self.edge_index)

        #print(self.train_pos_edge_index.size(),self.train_neg_edge_index.size(),self.test_pos_edge_index.size(),self.test_neg_edge_index.size())
        #print(len(self.train_nodes),len(self.test_nodes),self.ground_truth.sum(1))


        '''
        result = []
        indices = set(self.indices)
        for i in range(self.edge_index.size(1)):
            x,y = int(self.edge_index[0,i]),int(self.edge_index[1,i])
            if (x in indices and y not in indices) or (x not in indices and y in indices):result.append(0)
            else:result.append(1)



        self.train_edge_index = self.edge_index[:,np.where(result)[0]]
        additional_edge_index = torch.tensor(np.array(train_pos_link+[(j,i) for i,j in train_pos_link]).T).to(self.edge_index)
        self.train_edge_index = torch.hstack([self.train_edge_index,additional_edge_index])


        print(self.train_edge_index.size(),self.train_pos_edge_index.size(),self.test_pos_edge_index.size())
        '''





    def attr_split(self,ratio):
        mask_node = np.random.choice(self.x.shape[0],int(self.x.shape[0]*ratio),replace=False)
        self.attr = self.x[mask_node,:].detach()
        self.x[mask_node] = 0
        self.mask = mask_node

        #print(self.original_x[mask_node] )

    def attr_split2(self,ratio):
        mask_node = np.random.choice(self.x.shape[0],int(self.x.shape[0]*ratio),replace=False)
        #self.attr = self.x[mask_node,:].detach()
        self.xx = self.x.clone().detach()
        self.xx[mask_node] = 0
        self.mask = mask_node






    def link_split(self,train_ratio,test_ratio):

        row, col = self.edge_index


    # Return upper triangular portion.
        mask = row < col
        row, col = row[mask], col[mask]

        n_v = int(math.floor(train_ratio * row.size(0)))
        n_t = int(math.floor(test_ratio * row.size(0)))

    # Positive edges.
        perm = torch.randperm(row.size(0))
        row, col = row[perm], col[perm]

        r, c = row[:n_v], col[:n_v]
        train_pos_edge_index = torch.stack([r, c], dim=0)
        train_pos_edge_index = to_undirected(train_pos_edge_index)
        r, c = row[n_v:n_v + n_t], col[n_v:n_v + n_t]
        test_pos_edge_index = torch.stack([r, c], dim=0)

        r, c = row[n_v + n_t:], col[n_v + n_t:]
        val_pos_edge_index = torch.stack([r, c], dim=0)


        train_pos_edge_index = train_pos_edge_index.detach()
        val_pos_edge_index = val_pos_edge_index.detach()
        test_pos_edge_index = test_pos_edge_index.detach()

        num_nodes = self.x.shape[0]
        neg_adj_mask = torch.ones(num_nodes, num_nodes, dtype=torch.uint8)
        neg_adj_mask = neg_adj_mask.triu(diagonal=1)
        neg_adj_mask[row, col] = 0

        neg_row, neg_col = neg_adj_mask.nonzero(as_tuple=False).t()
        perm = torch.tensor(random.sample(range(neg_row.size(0)), n_t))
        perm = perm.to(torch.long)
        neg_row, neg_col = neg_row[perm], neg_col[perm]
        test_neg_edge_index = torch.stack([neg_row, neg_col], dim=0)

        self.train_edge_index = train_pos_edge_index
        self.test_edge_index = test_pos_edge_index






        '''

        if self.is_tensor:
            edge_index = self.edge_index.detach()

        row = edge_index[0]
        col = edge_index[1]

        # Return upper triangular portion.
        mask = row < col
        row, col = row[mask], col[mask]

        n_v = int(math.floor(train_ratio * row.size(0)))
        n_t = int(math.floor(test_ratio * row.size(0)))

        # Positive edges.
        perm = torch.randperm(row.size(0))
        row, col = row[perm], col[perm]

        r, c = row[:n_v], col[:n_v]
        train_pos_edge_index = to_undirected(torch.stack([r, c], dim=0))
        r, c = row[n_v:n_v + n_t], col[n_v:n_v + n_t]
        #test_pos_edge_index = to_undirected(torch.stack([r, c], dim=0))
        test_pos_edge_index = torch.stack([r, c], dim=0)

        train_pos_edge_index = train_pos_edge_index.detach()
        test_pos_edge_index = test_pos_edge_index.detach()

        self.train_edge_index = train_pos_edge_index
        self.test_edge_index = test_pos_edge_index
        '''


    def split_nodes(self,train_ratio,test_ratio):

        #### node split
        n_v = int(math.floor(train_ratio * self.x.size(0)))
        n_t = int(math.floor(test_ratio * self.x.size(0)))
        node_perm = torch.randperm(self.x.size(0))
        self.train_nodes = node_perm[:n_v]
        self.test_nodes = node_perm[n_v:n_v + n_t]

















    
    def identify(self,task,train_ratio):




        if task == 'link':
            if self.is_tensor:
                edge_index = self.edge_index.detach()

            row = edge_index[0]
            col = edge_index[1]

            mask = row < col
            row, col = row[mask], col[mask]

            tmp = int(1/train_ratio)
            train_set = []
            tmp_set = []
            val_set = []
            test_set = []

            for i in range(row.size(0)):
                if i%tmp == 0:train_set.append(i)
                else:tmp_set.append(i)

            tmp = int((len(tmp_set)+0.0)/int(0.1*row.size(0)))
            for i,ele in enumerate(tmp_set):
                if i%tmp == 0:val_set.append(ele)
                else:test_set.append(ele)
            

            r, c = row[train_set], col[train_set]
            train_pos_edge_index = to_undirected(torch.stack([r, c], dim=0))
            r, c = row[val_set], col[val_set]
            val_pos_edge_index = to_undirected(torch.stack([r, c], dim=0))
            r, c = row[test_set], col[test_set]
            test_pos_edge_index = torch.stack([r, c], dim=0) #to_undirected(torch.stack([r, c], dim=0))

            train_pos_edge_index = train_pos_edge_index.detach()
            test_pos_edge_index = test_pos_edge_index.detach()
            val_pos_edge_index = val_pos_edge_index.detach()

            self.train_edge_index = train_pos_edge_index
            self.test_edge_index = test_pos_edge_index
            self.val_edge_index = val_pos_edge_index
            self.test_edge_index_negative = negative_sampling_identify(self.test_edge_index,self.x.size(0))

        else:
            #label_num = int(self.labels.size(0)*train_ratio)
            #self.train_labels=list(range(label_num)) 
            #self.test_labels = list(range(label_num,self.labels.size(0)))
            '''
            tmp = int(1/train_ratio)+1
            train_set = []
            tmp_set = []
            val_set = []
            test_set = []



            for i in range(self.labels.size(0)):
                if i%tmp == 0:train_set.append(i)
                else:tmp_set.append(i)

            tmp = int((len(tmp_set)+0.0)/int(0.1*self.labels.size(0)))
            for i,ele in enumerate(tmp_set):
                if i%tmp == 0:val_set.append(ele)
                else:test_set.append(ele)

            self.train_labels=train_set
            self.test_labels = test_set
            self.val_labels = val_set
            '''
            train_num = int(self.labels.size(0)*train_ratio)
            val_num = int(self.labels.size(0)*0.1)

            if len(self.labels.shape) == 2:
                num_classes = int(self.labels.shape[1])

                label_arr = defaultdict(list)
                for index in range(num_classes):
                    label_arr[index] = np.where(self.labels[:,index].detach().cpu().numpy())[0].tolist()
                labels = []
                while len(label_arr)!=0:
                    for key in sorted(label_arr.keys()):
                        if len(label_arr[key]) == 0:
                            label_arr.pop(key)
                            continue
                        labels.append(label_arr[key].pop())
                    

                train_set = set()
                while len(train_set)!=train_num:train_set.add(labels.pop(0))
                val_set = set()
                while len(val_set)!=val_num:
                    value = labels.pop(0)
                    if value not in train_set:val_set.add(value)
                test_set = []
                for i in range(self.labels.size(0)):
                    if i not in train_set and i not in val_set:test_set.append(i)

                train_set = list(train_set)
                val_set = list(val_set)



            else:
                num_classes = int(self.labels.max().item()+1)

                label_arr = defaultdict(list)
                for index,label in enumerate(self.labels):
                    label_arr[label.item()].append(index)

                #for i in range(num_classes):print(len(label_arr[i]))
                labels = []
                while len(label_arr)!=0:
                    for key in sorted(label_arr.keys()):
                        if len(label_arr[key]) == 0:
                            label_arr.pop(key)
                            continue
                        labels.append(label_arr[key].pop())
                    

                train_set = labels[:train_num]
                val_set = labels[train_num:train_num+val_num]
                test_set = labels[train_num+val_num:]


            self.train_labels=train_set
            self.test_labels = test_set
            self.val_labels = val_set




            print('train_labels:',len(self.train_labels),'  val_labels:',len(self.val_labels),' test_labels:',len(self.test_labels))

            

            

            

        




















class dataGraph:

    def __init__(self,args):
        super(dataGraph, self).__init__()
        self.args = args
        self.dataset = args.dataset
        self.text = args.modal_text
        self.vision = args.modal_vision
        self.structure = args.modal_structure

        self.load_datasets()


    





    def load_datasets(self):

        dataset_str ='/data/lmk/mm-graph/{}/'.format(self.dataset)
        text_feats = np.load(dataset_str + '{}_text_feature_raw.npy'.format(self.text), mmap_mode='r')
        vision_feats = np.load(dataset_str + '{}_vision_feature_raw.npy'.format(self.vision), mmap_mode='r')
        structure_feats = np.load(dataset_str + '{}_structure_feature_raw.npy'.format(self.structure), mmap_mode='r')
        self.adj_full = sp.load_npz(dataset_str + 'adj_full.npz')
        self.original_feats = {'text':text_feats,'vision':vision_feats,'structure':structure_feats}
        self.modals = ('text','vision','structure')
        







    def fetch_for_fused(self,norm = True):
        
        subgraphs = {}
        #print(self.train_dataset)
        for modal in self.modals:
            sampling_graph,_ = self.sampling(modal,self.args.fuse_scale,norm)
            sampling_graph.adj = _
            subgraphs[modal] = sampling_graph
        return subgraphs






    def fetch_subgraph(self,norm = True):
        subgraphs = {}
        #print(self.train_dataset)
        for modal in self.modals:
            sampling_graph,_ = self.sampling(modal,self.args.subgraph_scale,norm)
            sampling_graph.adj = _
            subgraphs[modal] = sampling_graph
        return subgraphs
        

    def subgraph_to_tensor(self, subgraphs, device):
        for key in subgraphs.keys():
            subgraphs[key].to_tensor(device)


    def sampling(self,modal, n_samples=2000,norm =  True):
        mat = self.adj_full
        g_vertices = list(range(mat.shape[0]))

        sample = set()
        n_iter = 100 * n_samples

        num_vertices = len(g_vertices)

        current = g_vertices[randint(0, num_vertices-1)]
        sample.add(current)
        count = 0

        while len(sample) < n_samples:
            count += 1
            if count > n_iter: return 0
            if random.random() < 0.0002:
                current = g_vertices[randint(0, num_vertices-1)]
                sample.add(current)
                continue
            else:neighbors = mat[current, :].nonzero()[1]
            if len(neighbors) == 0:
                continue
            current = random.choice(neighbors)
            sample.add(current)

        sample = sorted(sample)
        adj = mat[sample, :][:, sample]
        adj = adj.tolil()
        for i in range(len(sample)):
            adj[i, i] = 0

        adj = adj.tocoo()
        adj_ = np.vstack((adj.row, adj.col)).astype(np.long)


        feats = self.original_feats[modal][sample]
        if norm:
            scaler = StandardScaler()
            scaler.fit(feats)
            feats = scaler.transform(feats)

        return Graph(feats, adj_),adj
    

    


