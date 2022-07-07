import torch
import torch.nn as nn
from dataset import HapDataset,RecDataset
from model import HapCL
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import *
import utils
import random
import numpy as np

random.seed(2022)
np.random.seed(2022)
torch.manual_seed(2022)          
torch.cuda.manual_seed_all(2022)     

config = {}
config['batch_size'] = 32
config['gcn_latent_dim'] = 32
config['gru_latent_dim'] = 32
config['GCN_n_layers'] = 2
config['gru_num_layers'] = 2
config['layer_norm_eps']=1e-5
config['dropout'] = 0.3
config['device'] = 'cuda:0'
config['train_path'] ="./data/beauty/train.txt"
config['val_path'] ="./data/beauty/val.txt"
config['test_path'] ="./data/beauty/test.txt"
config['run_type']='train'
config['num_aug']=2
config['n_interest']=4
config['epoch']=60
config['lr']=0.01
config['topk']=[5,30]
config['val_k']=5
config['val_step']=1
config['patience']=10
config['weight_decay']=10e-5
config['lambda1']=1
config['lambda2']=1
config['lambda3']=1



def collate_fn(data):
    bseq=[]
    bseq_len=[]
    btar=[]
    for i in data:
        bseq.append(i[0])
        bseq_len.append(i[1])
        btar.append(i[2])

    bseq=pad_sequence(bseq,batch_first=True)
    bseq_len=torch.tensor(bseq_len)
    btar=torch.stack(btar,dim=0)
    return bseq,bseq_len,btar

def recloss(logits,targets):
    rec_loss=-torch.sum(targets*torch.log(logits+10e-24))/torch.sum(targets)-torch.sum((1-targets)*torch.log(1-logits+10e-24))/torch.sum(1-targets)
    return rec_loss


def cllossnp(batch_pos_bseq_emb,batch_neg_bseq_emb):
    ce=nn.CrossEntropyLoss()
    sim_pp = torch.matmul(batch_pos_bseq_emb, batch_pos_bseq_emb.T) 
    sim_nn = torch.matmul(batch_neg_bseq_emb, batch_neg_bseq_emb.T)
    sim_pn = torch.matmul(batch_pos_bseq_emb, batch_neg_bseq_emb.T)
    d = sim_pn.shape[-1]
    sim_pp[..., range(d), range(d)] = 0.0
    sim_nn[..., range(d), range(d)] = 0.0
    raw_scores1 = torch.cat([sim_pn, sim_pp], dim=-1)
    raw_scores2 = torch.cat([sim_nn, sim_pn.transpose(-1, -2)], dim=-1)
    all_scores = torch.cat([raw_scores1, raw_scores2], dim=-2)
    labels = torch.arange(2 * d, dtype=torch.long, device=all_scores.device)
    cl_loss = ce(all_scores, labels)
    return cl_loss

def disloss(batch_bseq_emb,batch_iseq_emb):
    dloss=torch.mean(torch.pow(batch_bseq_emb-batch_iseq_emb,2))
    return dloss



if __name__=="__main__":

    lgcn_dataset = HapDataset(config['train_path'], config['val_path'], config['test_path'])
    HapCL=HapCL(config,lgcn_dataset).to(config['device'])
    if config['run_type']=='train':
        best_val=0
        count=0
        optimizer=torch.optim.Adam(HapCL.parameters(),lr=config['lr'],weight_decay=config['weight_decay'])
        train_dataset=RecDataset(config['train_path'], lgcn_dataset.basket2id_dict, lgcn_dataset.item2id_dict)
        train_recloader=DataLoader(train_dataset,batch_size=config['batch_size'],collate_fn=collate_fn,drop_last=True)
        print("Begin to train......")
        for e in range(config['epoch']):
            HapCL=HapCL.train()
            HapCL.g_droped = None
            HapCL.pos_g_droped = None
            HapCL.neg_g_droped = None
            HapCL.get_graph(run_type='aug')
            HapCL.augment()
            HapCL.get_graph(run_type='train')
            print("Graphs construct Finish...")
            for batch_id,(bseq,bseq_len,btar) in tqdm(enumerate(train_recloader)):
                bseq=bseq.to(config['device'])
                bseq_len=bseq_len.to(config['device'])
                btar=btar.to(config['device'])
                logits, batch_pos_bseq_emb, batch_neg_bseq_emb, batch_bseq_emb,item_logits,basket_logits=HapCL(bseq, bseq_len,'train')
                rec_loss=recloss(logits,btar)
                cl_loss=cllossnp(batch_pos_bseq_emb, batch_neg_bseq_emb)
                logits_loss=disloss(item_logits,basket_logits)
                loss=config['lambda1']*rec_loss+config['lambda2']*cl_loss+config['lambda3']*logits_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            if e%config['val_step']==0:
                HapCL.eval()
                HapCL.get_graph(run_type='val')
                val_dataset=RecDataset(config['val_path'], lgcn_dataset.basket2id_dict, lgcn_dataset.item2id_dict)
                val_recloader=DataLoader(val_dataset,batch_size=config['batch_size'],collate_fn=collate_fn,drop_last=True)
                f1s={}
                hits={}
                gts=0
                sample_num=0
                for k in config['topk']:
                    f1s[k]=0
                    hits[k]=0
                for batch_id,(bseq,bseq_len,btar) in tqdm(enumerate(val_recloader)):
                    sample_num=sample_num+len(bseq_len)
                    gts=gts+len(torch.nonzero(btar))
                    bseq=bseq.to(config['device'])
                    bseq_len=bseq_len.to(config['device'])
                    btar=btar.to(config['device'])
                    logits=HapCL(bseq, bseq_len,'val')  #[batch_size,num_items]
                    for k in config['topk']:
                        preds=torch.topk(logits,dim=1,k=k).indices   #[batch_size,k]
                        preds=torch.zeros_like(logits).scatter_(1,preds,1)
                        f1s[k]=f1s[k]+sum(utils.f1(preds,btar))
                        hits[k]=hits[k]+utils.hit(preds,btar)
                
                if f1s[config['val_k']]>best_val:
                    best_val=f1s[config['val_k']]
                    count=0
                    torch.save(HapCL.state_dict(), 'beauty.pth')
                    print("Model of epoch {} is saved.".format(e))
                else:
                    count=count+1
                    print("Counter {} of {}".format(count,config['patience']))
                    if count>=config['patience']:
                        break
            

    elif config['run_type']=='test':
        HapCL.eval()
        HapCL.get_graph(run_type='test')
        test_dataset=RecDataset(config['test_path'], lgcn_dataset.basket2id_dict, lgcn_dataset.item2id_dict)
        test_recloader=DataLoader(test_dataset,batch_size=config['batch_size'],collate_fn=collate_fn,drop_last=True)
        HapCL.load_state_dict(torch.load('beauty.pth'))
        f1s={}
        hits={}
        gts=0
        sample_num=0
        for k in config['topk']:
            f1s[k]=0
            hits[k]=0
        for batch_id,(bseq,bseq_len,btar) in tqdm(enumerate(test_recloader)):
            sample_num=sample_num+len(bseq_len)
            gts=gts+len(torch.nonzero(btar))
            bseq=bseq.to(config['device'])
            bseq_len=bseq_len.to(config['device'])
            btar=btar.to(config['device'])
            logits=HapCL(bseq, bseq_len,'test')  #[batch_size,num_items]
            for k in config['topk']:
                preds=torch.topk(logits,dim=1,k=k).indices   #[batch_size,k]
                preds=torch.zeros_like(logits).scatter_(1,preds,1)
                f1s[k]=f1s[k]+sum(utils.f1(preds,btar))
                hits[k]=hits[k]+utils.hit(preds,btar)