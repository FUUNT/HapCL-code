import torch
from dataset import HapDataset
from torch import nn
from torch.nn.utils.rnn import pad_sequence
import numpy as np


class HapCL(nn.Module):
    def __init__(self, 
                 config:dict, 
                 dataset:HapDataset):
        super(HapCL, self).__init__()
        self.config = config
        self.dataset = dataset
        self.num_baskets  = self.dataset.num_baskets
        self.num_items  = self.dataset.num_items
        self.id2basket_dict=self.dataset.id2basket_dict
        self.n_layers = self.config['GCN_n_layers']
        self.n_interest=self.config['n_interest']

        self.gru=nn.GRU(input_size=config['gcn_latent_dim'], 
                        hidden_size=config['gru_latent_dim'], 
                        num_layers=config['gru_num_layers'],
                        dropout=config['dropout'],
                        batch_first=True)
        self.sigmoid=nn.Sigmoid()
        self.softmax=nn.Softmax(dim=2)
        self.relu=nn.ReLU()
        self.item_gru = nn.GRU(
            input_size=self.config['gcn_latent_dim'],
            hidden_size=config['gru_latent_dim'],
            num_layers=config['gru_num_layers'],
            dropout=config['dropout'],
            batch_first=True,
        )
        self.linear_bseq_interest=nn.Linear(config['gcn_latent_dim'],self.n_interest*config['gcn_latent_dim'],bias=False)
        self.linear_iseq_interest=nn.Linear(config['gcn_latent_dim'],self.n_interest*config['gcn_latent_dim'],bias=False)
        self.merge_bakset_interest=nn.Linear(self.n_interest,1,bias=False)
        self.merge_item_interest=nn.Linear(self.n_interest,1,bias=False)

        self.embedding_basket = torch.nn.Embedding(num_embeddings=self.num_baskets, embedding_dim=self.config['gcn_latent_dim'], padding_idx=0)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.config['gcn_latent_dim'], padding_idx=0)
        nn.init.normal_(self.embedding_basket.weight, mean=0, std=0.1)
        nn.init.normal_(self.embedding_item.weight, mean=0, std=0.1)


    def get_graph(self, run_type='train'):
        if run_type=='train':
            self.g_droped = self.dataset.getSparseGraph(graph_type='original',run_type='train') 
            self.pos_g_droped = self.dataset.getSparseGraph(graph_type='pos',run_type='train') 
            self.neg_g_droped = self.dataset.getSparseGraph(graph_type='neg',run_type='train') 
        elif run_type=='val':
            self.g_droped = self.dataset.getSparseGraph(graph_type='original',run_type='val') 
        elif run_type=='test':
            self.g_droped = self.dataset.getSparseGraph(graph_type='original',run_type='test') 
     

    def computer(self, graph_type='original'):
        baskets_emb = self.embedding_basket.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([baskets_emb, items_emb])
        embs = [all_emb]
        
        if graph_type=='original':
            for layer in range(self.n_layers):
                all_emb = torch.sparse.mm(self.g_droped, all_emb)
                embs.append(all_emb)
        elif graph_type=='pos':
            for layer in range(self.n_layers):
                all_emb = torch.sparse.mm(self.pos_g_droped, all_emb)
                embs.append(all_emb)
        elif graph_type=='neg':
            for layer in range(self.n_layers):
                all_emb = torch.sparse.mm(self.neg_g_droped, all_emb)
                embs.append(all_emb)
            
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        baskets, items = torch.split(light_out, [self.num_baskets, self.num_items])
        if graph_type=='original':
            return baskets.to(self.config['device']), items.to(self.config['device'])
        else:
            return baskets
    
    def augment(self):
        baskets, items=self.computer(graph_type='original')   
        baskets=baskets.detach().cpu()
        items=items.detach().cpu()
        basket_item_intensity=torch.mm(baskets,items.t()) 
        item_filter=torch.argsort(basket_item_intensity,dim=1,descending=True)
        del basket_item_intensity
        aug_trainBasket=np.array([])
        aug_train_b2i_weight=np.array([])
        pos_trainItem=np.array([])
        neg_trainItem=np.array([])
        for i in range(1,len(item_filter)+1):
            if i in self.dataset.trainBasket:
                aug_trainBasket=np.append(aug_trainBasket, [i]*self.config['num_aug'])
                aug_train_b2i_weight=np.append(aug_train_b2i_weight,[1]*self.config['num_aug'])
                pos_trainItem=np.append(pos_trainItem,item_filter[i][:self.config['num_aug']])
                neg_trainItem=np.append(neg_trainItem,item_filter[i][-self.config['num_aug']:])
        self.dataset.aug_trainBasket=np.append(self.dataset.trainBasket, aug_trainBasket)
        self.dataset.aug_train_b2i_weight=np.append(self.dataset.train_b2i_weight,aug_train_b2i_weight)
        self.dataset.pos_trainItem=np.append(self.dataset.trainItem, pos_trainItem)
        self.dataset.neg_trainItem=np.append(self.dataset.trainItem, neg_trainItem)
        del aug_trainBasket,aug_train_b2i_weight,pos_trainItem,neg_trainItem,item_filter
        print("Data augmentation Finish...")


    def gather_embs(self, embs, gather_index):
        embs = embs.reshape(1,embs.shape[0],embs.shape[1]).expand(gather_index.shape[0],-1,-1)
        gather_index = gather_index.reshape(gather_index.shape[0], gather_index.shape[1], 1).expand(-1, -1, embs.shape[-1])
        embs_tensor = embs.gather(dim=1, index=gather_index)
        return embs_tensor
    
    def gather_indexes(self, output, gather_index):
        gather_index = gather_index.reshape(1,-1, 1, 1).expand(output.shape[0],-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=2, index=gather_index)
        return output_tensor.squeeze(2)
    

    def forward(self, bseq, bseq_len, run_type='train'):
        if run_type=='train':
            baskets, items=self.computer(graph_type='original')  
            pos_baskets=self.computer(graph_type='pos')   
            neg_baskets=self.computer(graph_type='neg')  

        else:
            baskets, items=self.computer(graph_type='original')   
        
        item_seq=[]
        item_seq_len=[]
        for b in bseq:
            item=[]
            for bid in b:
                if int(bid)!=0:
                    item=item+self.id2basket_dict[int(bid)]
            item_seq.append(torch.as_tensor(item))
            item_seq_len.append(len(item))
        item_seq=pad_sequence(item_seq,batch_first=True).to(self.config['device'])
        item_seq_len=torch.as_tensor(item_seq_len).to(self.config['device'])
        
        basket_bseq_emb=self.gather_embs(baskets,bseq)
        basket_bseq_emb=self.linear_bseq_interest(basket_bseq_emb)
        basket_bseq_emb=basket_bseq_emb.reshape(basket_bseq_emb.shape[0],basket_bseq_emb.shape[1],-1,self.n_interest)
        basket_bseq_emb=basket_bseq_emb.permute(3,0,1,2)
        basket_bseq_emb=basket_bseq_emb.reshape(-1,basket_bseq_emb.shape[2],basket_bseq_emb.shape[3])
        basket_bseq_repre,_=self.gru(basket_bseq_emb) 
        basket_bseq_repre=basket_bseq_repre.reshape(self.n_interest,-1,basket_bseq_repre.shape[1],basket_bseq_repre.shape[2])
        batch_bseq_emb=self.gather_indexes(basket_bseq_repre,bseq_len-1)
        batch_bseq_emb=batch_bseq_emb.permute(1,0,2)
        interest_atten=torch.matmul(batch_bseq_emb,batch_bseq_emb.permute(0,2,1))
        interest_atten=self.softmax(interest_atten)
        basket_logits=torch.matmul(batch_bseq_emb,items.T)
        basket_logits=basket_logits.permute(0,2,1)
        basket_logits=torch.matmul(basket_logits,interest_atten)
        basket_logits=self.merge_bakset_interest(basket_logits)
        basket_logits=basket_logits.squeeze(2)

        
        item_seq_emb=self.gather_embs(items,item_seq)
        item_seq_emb=self.linear_iseq_interest(item_seq_emb)
        item_seq_emb=item_seq_emb.reshape(item_seq_emb.shape[0],item_seq_emb.shape[1],-1,self.n_interest)
        item_seq_emb=item_seq_emb.permute(3,0,1,2)
        item_seq_emb=item_seq_emb.reshape(-1,item_seq_emb.shape[2],item_seq_emb.shape[3])
        item_iseq_repre,_=self.item_gru(item_seq_emb) 
        item_iseq_repre=item_iseq_repre.reshape(self.n_interest,-1,item_iseq_repre.shape[1],item_iseq_repre.shape[2])
        batch_iseq_emb=self.gather_indexes(item_iseq_repre,item_seq_len-1)
        batch_iseq_emb=batch_iseq_emb.permute(1,0,2)
        interest_atten=torch.matmul(batch_iseq_emb,batch_iseq_emb.permute(0,2,1))
        interest_atten=self.softmax(interest_atten)
        item_logits=torch.matmul(batch_iseq_emb,items.T)
        item_logits=item_logits.permute(0,2,1)
        item_logits=torch.matmul(item_logits,interest_atten)
        item_logits=self.merge_item_interest(item_logits)
        item_logits=item_logits.squeeze(2)
        

        logits=basket_logits+item_logits
        logits=self.sigmoid(logits)

        
        if run_type=='train':
            basket_bseq_pos_emb=self.gather_embs(pos_baskets,bseq)
            basket_bseq_pos_emb=self.linear_bseq_interest(basket_bseq_pos_emb)
            basket_bseq_pos_emb=basket_bseq_pos_emb.reshape(basket_bseq_pos_emb.shape[0],basket_bseq_pos_emb.shape[1],-1,self.n_interest)
            basket_bseq_pos_emb=basket_bseq_pos_emb.permute(3,0,1,2)
            basket_bseq_pos_emb=basket_bseq_pos_emb.reshape(-1,basket_bseq_pos_emb.shape[2],basket_bseq_pos_emb.shape[3])
            basket_bseq_pos_repre,_=self.gru(basket_bseq_pos_emb) 
            basket_bseq_pos_repre=basket_bseq_pos_repre.reshape(self.n_interest,-1,basket_bseq_pos_repre.shape[1],basket_bseq_pos_repre.shape[2])
            batch_pos_bseq_emb=self.gather_indexes(basket_bseq_pos_repre,bseq_len-1)
            batch_pos_bseq_emb=batch_pos_bseq_emb.permute(1,2,0)
            batch_pos_bseq_emb=batch_pos_bseq_emb.reshape(batch_pos_bseq_emb.shape[0],-1)

            basket_bseq_neg_emb=self.gather_embs(neg_baskets,bseq)
            basket_bseq_neg_emb=self.linear_bseq_interest(basket_bseq_neg_emb)
            basket_bseq_neg_emb=basket_bseq_neg_emb.reshape(basket_bseq_neg_emb.shape[0],basket_bseq_neg_emb.shape[1],-1,self.n_interest)
            basket_bseq_neg_emb=basket_bseq_neg_emb.permute(3,0,1,2)
            basket_bseq_neg_emb=basket_bseq_neg_emb.reshape(-1,basket_bseq_neg_emb.shape[2],basket_bseq_neg_emb.shape[3])
            basket_bseq_neg_repre,_=self.gru(basket_bseq_neg_emb) 
            basket_bseq_neg_repre=basket_bseq_neg_repre.reshape(self.n_interest,-1,basket_bseq_neg_repre.shape[1],basket_bseq_neg_repre.shape[2])
            batch_neg_bseq_emb=self.gather_indexes(basket_bseq_neg_repre,bseq_len-1)
            batch_neg_bseq_emb=batch_neg_bseq_emb.permute(1,2,0)
            batch_neg_bseq_emb=batch_neg_bseq_emb.reshape(batch_neg_bseq_emb.shape[0],-1)

            batch_bseq_emb=batch_bseq_emb.permute(0,2,1)
            batch_bseq_emb=batch_bseq_emb.reshape(batch_bseq_emb.shape[0],-1)
            return logits,batch_pos_bseq_emb,batch_neg_bseq_emb,batch_bseq_emb,item_logits,basket_logits
        
        else:
            return logits