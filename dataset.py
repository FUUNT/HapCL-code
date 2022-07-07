import torch
import numpy as np
from torch.utils.data import Dataset
from scipy.sparse import csr_matrix
import scipy.sparse as sp

device='cuda:0'


class HapDataset(Dataset):

    def __init__(self, train_path, val_path, test_path):
        self.trainItem, self.trainBasket, self.train_b2i_weight= np.array([],dtype=np.int32), np.array([],dtype=np.int32), np.array([],dtype=np.int32)
        self.valItem, self.valBasket, self.val_b2i_weight = np.array([],dtype=np.int32), np.array([],dtype=np.int32), np.array([],dtype=np.int32)
        self.testItem, self.testBasket, self.test_b2i_weight = np.array([],dtype=np.int32), np.array([],dtype=np.int32), np.array([],dtype=np.int32)
        self.basket2id_dict = {}
        self.item2id_dict = {}
        self.id2basket_dict = {}
        self.num_baskets = 1
        self.num_items = 1
        
        self.aug_trainBasket = None
        self.aug_train_b2i_weight = None
        self.pos_trainItem = None
        self.neg_trainItem = None

        self.posBasketItemNet = None
        self.negBasketItemNet = None
        
        with open(train_path,'r') as f:
            for l in f.readlines():
                baskets=l.split('|')[:-1]

                btar=baskets[-1].strip('\n').split(' ')
                for i in range(len(btar)):
                    if btar[i] not in self.item2id_dict:
                        self.item2id_dict[btar[i]]=self.num_items
                        self.num_items=self.num_items+1
                
                bseq=baskets[:-1]
                for b in bseq:
                    b=b.strip('\n').split(' ')
                    for i in range(len(b)):
                        if b[i] not in self.item2id_dict:
                            self.item2id_dict[b[i]]=self.num_items
                            self.num_items=self.num_items+1
                            b[i]=self.item2id_dict[b[i]]
                        else:
                            b[i]=self.item2id_dict[b[i]]
                    b.sort()
                    if tuple(b) not in self.basket2id_dict:
                        self.basket2id_dict[tuple(b)]=self.num_baskets
                        self.id2basket_dict[self.num_baskets]=b
                        self.num_baskets=self.num_baskets+1
                        self.trainBasket=np.append(self.trainBasket,[self.basket2id_dict[tuple(b)]]*len(b))
                        self.trainItem=np.append(self.trainItem,b)
                        self.train_b2i_weight=np.append(self.train_b2i_weight,[1]*len(b))
                    else:
                        if self.basket2id_dict[tuple(b)] in self.trainBasket:
                            self.train_b2i_weight[self.trainBasket==self.basket2id_dict[tuple(b)]]+=1
                        else:
                            self.trainBasket=np.append(self.trainBasket,[self.basket2id_dict[tuple(b)]]*len(b))
                            self.trainItem=np.append(self.trainItem,b)
                            self.train_b2i_weight=np.append(self.train_b2i_weight,[1]*len(b))



        with open(val_path,'r') as f:
            for l in f.readlines():
                baskets=l.split('|')[:-1]

                btar=baskets[-1].strip('\n').split(' ')
                for i in range(len(btar)):
                    if btar[i] not in self.item2id_dict:
                        self.item2id_dict[btar[i]]=self.num_items
                        self.num_items=self.num_items+1

                bseq=baskets[:-1]
                for b in bseq:
                    b=b.strip('\n').split(' ')
                    for i in range(len(b)):
                        if b[i] not in self.item2id_dict:
                            self.item2id_dict[b[i]]=self.num_items
                            self.num_items=self.num_items+1
                            b[i]=self.item2id_dict[b[i]]
                        else:
                            b[i]=self.item2id_dict[b[i]]
                    b.sort()
                    if tuple(b) not in self.basket2id_dict:
                        self.basket2id_dict[tuple(b)]=self.num_baskets
                        self.id2basket_dict[self.num_baskets]=b
                        self.num_baskets=self.num_baskets+1
                        self.valBasket=np.append(self.valBasket,[self.basket2id_dict[tuple(b)]]*len(b))
                        self.valItem=np.append(self.valItem,b)
                        self.val_b2i_weight=np.append(self.val_b2i_weight,[1]*len(b))
                    else:
                        if self.basket2id_dict[tuple(b)] in self.valBasket:
                            self.val_b2i_weight[self.valBasket==self.basket2id_dict[tuple(b)]]+=1
                        else:
                            self.valBasket=np.append(self.valBasket,[self.basket2id_dict[tuple(b)]]*len(b))
                            self.valItem=np.append(self.valItem,b)
                            self.val_b2i_weight=np.append(self.val_b2i_weight,[1]*len(b))



        with open(test_path,'r') as f:
            for l in f.readlines():
                baskets=l.split('|')[:-1]
                
                btar=baskets[-1].strip('\n').split(' ')
                for i in range(len(btar)):
                    if btar[i] not in self.item2id_dict:
                        self.item2id_dict[btar[i]]=self.num_items
                        self.num_items=self.num_items+1
                        
                bseq=baskets[:-1]
                for b in bseq:
                    b=b.strip('\n').split(' ')
                    for i in range(len(b)):
                        if b[i] not in self.item2id_dict:
                            self.item2id_dict[b[i]]=self.num_items
                            self.num_items=self.num_items+1
                            b[i]=self.item2id_dict[b[i]]
                        else:
                            b[i]=self.item2id_dict[b[i]]
                    b.sort()
                    if tuple(b) not in self.basket2id_dict:
                        self.basket2id_dict[tuple(b)]=self.num_baskets
                        self.id2basket_dict[self.num_baskets]=b
                        self.num_baskets=self.num_baskets+1
                        self.testBasket=np.append(self.testBasket,[self.basket2id_dict[tuple(b)]]*len(b))
                        self.testItem=np.append(self.testItem,b)
                        self.test_b2i_weight=np.append(self.test_b2i_weight,[1]*len(b))
                    else:
                        if self.basket2id_dict[tuple(b)] in self.testBasket:
                            self.test_b2i_weight[self.testBasket==self.basket2id_dict[tuple(b)]]+=1
                        else:
                            self.testBasket=np.append(self.testBasket,[self.basket2id_dict[tuple(b)]]*len(b))
                            self.testItem=np.append(self.testItem,b)
                            self.test_b2i_weight=np.append(self.test_b2i_weight,[1]*len(b))
        
        print("HapDataset is ready to go")
    
    
    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))
        
    def getSparseGraph(self, graph_type='original',run_type='train'):
        adj_mat = sp.dok_matrix((self.num_baskets + self.num_items, self.num_baskets + self.num_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()

        if graph_type=='original':
            if run_type=='train':
                R = csr_matrix((self.train_b2i_weight, (self.trainBasket, self.trainItem)),
                                      shape=(self.num_baskets, self.num_items)).tolil()
            elif run_type=='val':
                R = csr_matrix((self.val_b2i_weight, (self.valBasket, self.valItem)),
                                      shape=(self.num_baskets, self.num_items)).tolil()
            elif run_type=='test':
                R = csr_matrix((self.test_b2i_weight, (self.testBasket, self.testItem)),
                                      shape=(self.num_baskets, self.num_items)).tolil()
        elif graph_type=='pos':
            R = csr_matrix((self.aug_train_b2i_weight, (self.aug_trainBasket, self.pos_trainItem)),
                                      shape=(self.num_baskets, self.num_items)).tolil()   
        elif graph_type=='neg':
            R = csr_matrix((self.aug_train_b2i_weight, (self.aug_trainBasket, self.neg_trainItem)),
                                      shape=(self.num_baskets, self.num_items)).tolil()
                                      

        adj_mat[:self.num_baskets, self.num_baskets:] = R
        adj_mat[self.num_baskets:, :self.num_baskets] = R.T
        adj_mat = adj_mat.todok()

        
        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        
        norm_adj = d_mat.dot(adj_mat)
        norm_adj = norm_adj.dot(d_mat)
        norm_adj = norm_adj.tocsr()

        Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
        Graph = Graph.coalesce().to(device)
        
        return Graph

        



class RecDataset(Dataset):
    def __init__(self, path, basket2id_dict, item2id_dict):
        self.Basketseqs=[]
        self.Baskettars=[]
        self.Basketseq_length=[]
        self.basket2id_dict=basket2id_dict
        self.item2id_dict=item2id_dict
        with open(path,'r') as f:
            for l in f.readlines():
                bseq_id=[]
                bseq=l.split('|')[:-1]
                for b in bseq[:-1]:
                    b=b.strip('\n').split(' ')
                    for i in range(len(b)):
                        b[i]=self.item2id_dict[b[i]]
                    b.sort()
                    bseq_id.append(self.basket2id_dict[tuple(b)])
                self.Basketseqs.append(bseq_id)#basket_seq_len*basket_len
                self.Basketseq_length.append(len(bseq_id))#basket_seq_len
                btar=bseq[-1].strip('\n').split(' ')
                vector_btar=[0]*(len(self.item2id_dict)+1)
                for i in range(len(btar)):
                    vector_btar[self.item2id_dict[btar[i]]]=1
                self.Baskettars.append(vector_btar)
        print("RecDataset is ready to go")
        
    def __getitem__(self, index):
        return torch.tensor(self.Basketseqs[index]),torch.tensor(self.Basketseq_length[index]),torch.tensor(self.Baskettars[index])
    
    def __len__(self):
        return len(self.Basketseqs)
        
