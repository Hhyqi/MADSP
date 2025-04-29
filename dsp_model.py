import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from lifelines.utils import concordance_index
from scipy.stats import pearsonr,spearmanr
import copy
import torch
import pickle
import time
import math
from torch.utils import data
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import SequentialSampler
from torch_geometric.nn import global_max_pool as gmp, global_mean_pool
from prettytable import PrettyTable
from subword_nmt.apply_bpe import BPE
from get_dataset import GetData
from torch_geometric.data import InMemoryDataset, Batch
from torch_geometric import data as DATA
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, cohen_kappa_score,auc
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split
import sys
device = torch.device('cuda:2')

con_loss_T = 0.05                   
class data_process_loader(data.Dataset):            
    def __init__(self, list_IDs, labels, drug_df,rna_df,types):
        self.labels = labels
        self.list_IDs = list_IDs
        self.drug_df1 = torch.Tensor(np.array([data[0][0:] for data in drug_df['drug_encoding']])).float().to(device)
        self.drug_df2 = torch.Tensor(np.array([data[1][0:] for data in drug_df['drug_encoding']])).float().to(device)
        self.cell_df = torch.Tensor(np.array([data for data in rna_df['cell_encoding']])).float().to(device)
        self.types = types
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        #index = self.list_IDs[index]
        d_f1 = self.drug_df1[index]
        d_f2 = self.drug_df2[index]
        cl_f = self.cell_df[index]
        y = self.labels[index]
        t = self.types[index]
        return d_f1,d_f2,cl_f,y,t
    
def collate(data_list):
    data_list = data_list
    DAF1 = [data[0] for data in data_list]
    DBF1 = [data[1] for data in data_list]
    CF1 = [data[2] for data in data_list]
    label = [data[3] for data in data_list]
    targets = [data[4] for data in data_list]
    return DAF1, DBF1,CF1,label,targets
    
class Predictor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super(Predictor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size // 2),
            nn.Dropout(0.5),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self,feat):
        out = self.network(feat)
        return out
    
class MultiHeadAttention(torch.nn.Module):
    def __init__(self,input_dim,n_heads,ouput_dim=None):
        
        super(MultiHeadAttention, self).__init__()
        self.d_k=self.d_v=input_dim//n_heads
        self.n_heads = n_heads
        if ouput_dim==None:
            self.ouput_dim=input_dim
        else:
            self.ouput_dim=ouput_dim
        self.W_Q = torch.nn.Linear(input_dim, self.d_k * self.n_heads, bias=False)
        self.W_K = torch.nn.Linear(input_dim, self.d_k * self.n_heads, bias=False)
        self.W_V = torch.nn.Linear(input_dim, self.d_v * self.n_heads, bias=False)
        self.fc = torch.nn.Linear(self.n_heads * self.d_v, self.ouput_dim, bias=False)
    def forward(self,X):
        ## (S, D) -proj-> (S, D_new) -split-> (S, H, W) -trans-> (H, S, W)
        batch_size, seq_len, _ = X.shape
        Q=self.W_Q(X).view( -1, self.n_heads, self.d_k).transpose(0,1)
        K=self.W_K(X).view( -1, self.n_heads, self.d_k).transpose(0,1)
        V=self.W_V(X).view( -1, self.n_heads, self.d_v).transpose(0,1)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        # context: [n_heads, len_q, d_v], attn: [n_heads, len_q, len_k]
        attn = torch.nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)  
        # context: [len_q, n_heads * d_v]
        context = context.transpose(1, 2).reshape(-1, self.n_heads * self.d_v)
        output = self.fc(context)
        output = output.reshape((X.shape[0], X.shape[1],X.shape[2]))
        return output

    
class EncoderLayer(torch.nn.Module):
    def __init__(self,input_dim,n_heads):
        super(EncoderLayer, self).__init__()
        self.attn = MultiHeadAttention(input_dim,n_heads)
        self.AN1=torch.nn.LayerNorm(input_dim)
        self.l1=torch.nn.Linear(input_dim, input_dim)
        self.AN2=torch.nn.LayerNorm(input_dim)
    def forward (self,X):
        output=self.attn(X)
        
        X=self.AN1(output+X)
        output=self.l1(X)
        X=self.AN2(output+X)
        return X

def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class feature_encoder(torch.nn.Module):  # twin network
    def __init__(self, vector_size,n_heads,n_layers):
        super(feature_encoder, self).__init__()

        self.layers = torch.nn.ModuleList([EncoderLayer(vector_size, n_heads) for _ in range(n_layers)])
        self.AN = torch.nn.LayerNorm(vector_size)

        self.l1 = torch.nn.Linear(vector_size, 128)
        self.dr = torch.nn.Dropout(0.5)
        

    def forward(self, X):
        for layer in self.layers:
            X = layer(X)
        X1=self.AN(X)
        X2 = self.l1(X1)
        return X2 
class Model(torch.nn.Module):
    def __init__(self,hiddim):
        super(Model, self).__init__()
        self.predictor = Predictor(1792,hiddim)
        self.AN = feature_encoder(1024,8,2) #8
    def forward(self, VF):
        
        DAF1 = torch.stack(VF[0])
        DBF1 = torch.stack(VF[1])
        CF1 = torch.stack(VF[2])
        DFC1 = torch.cat((DAF1,DBF1), dim=1) 
        
        AC = self.AN(DFC1)
        AC = AC.reshape(AC.shape[0], AC.shape[1]*AC.shape[2])
        FC = torch.cat((AC,CF1), 1)
        X = self.predictor(FC)
        return X
    
class myloss(nn.Module):
    def __init__(self):
        super(myloss,self).__init__()
        self.loss1 = nn.MSELoss(reduction='sum')
    def forward(self, X, inputs):
        loss1 = self.loss1(inputs, X)
        return loss1
    

class DSPSCL:
    def __init__(self,modeldir,foldnum,hiddim,mmse):
        self.model = Model(hiddim)
        self.modeldir = modeldir
        self.record_file = os.path.join(self.modeldir, "valid_fold"+str(foldnum)+".txt")
        self.pkl_file = os.path.join(self.modeldir, "loss_curve_iter.pkl")
        self.msemax = mmse
        self.fold = foldnum
    def test(self,datagenerator,model):
        y_label = []
        y_pred = []
        model.eval()
        testloss = nn.MSELoss(reduction='sum')
        tloss = 0
        for i, v_f in enumerate(datagenerator):
            
            label = v_f[3]
            label = torch.Tensor(list(label)).to(device)
            score = self.model(v_f)
            score = torch.squeeze(score, 1).float().to(device)
            loss = testloss(label,score)
            logits = torch.squeeze(score).detach().cpu().numpy()
            label_ids = label.to('cpu').numpy()
            y_label = y_label + label_ids.flatten().tolist()
            y_pred = y_pred + logits.flatten().tolist() 
            tloss  +=  loss.item()
            t = time.time()

        model.train()
        mset =  mean_squared_error(y_label, y_pred)
        
        return np.sqrt(mean_squared_error(y_label, y_pred)), \
               pearsonr(y_label, y_pred)[0], \
               spearmanr(y_label, y_pred)[0], \
               concordance_index(y_label, y_pred),y_label,y_pred
    
    def adjust_learning_rate(self,optimizer, lr):
        for param_group in optimizer.param_groups:

            param_group['lr'] = lr

    def train(self, tr_drug, tr_cl,testdata):
        lr = 1e-4
        BATCH_SIZE = 512
        train_epoch = 200
        self.model = self.model.to(device)
        opt = torch.optim.Adam(self.model.parameters(), lr = lr)
        loss_history = []
        params = {'batch_size': BATCH_SIZE,'shuffle': True,'num_workers': 0,'drop_last': False}
        tr_generator = data.DataLoader(data_process_loader(
            tr_drug.index.values, tr_drug.label.values, tr_drug, tr_cl,tr_drug.type.values),collate_fn=collate, **params)
        print('--- Go for Training ---')
        my_loss=myloss()
        for epo in range(train_epoch):
            running_loss = 0.0
            for i, v_f in enumerate(tr_generator):
                label = v_f[3]
                opt.zero_grad()
                score = self.model(v_f)
                score = torch.squeeze(score, 1).float().to(device)
                label = torch.Tensor(list(label)).to(device)
                loss = my_loss(score, label)
                loss.backward()
                opt.step()            
                running_loss += loss.item()
            #print(str(epo)+"  ", end="")
            
            
            self.predict(testdata)
            print("\r", end="")
            print("{}%: ".format(int((epo)*(100/train_epoch))+1), "▓" * int((epo // 2)*(100/train_epoch)), end="")
            sys.stdout.flush()
        with open(self.pkl_file, 'wb') as pck:
            pickle.dump(loss_history, pck)
        print('\n--- Training Finished ---')
        
    def predict(self,test_generator):
        with torch.set_grad_enabled(False):
            rmse,person, spearman, CI,y_true,y_pred   = self.test(test_generator, self.model)
            # print(
            # 'RMSE: ' + str(rmse)[:7] +
            # ' , Pearson Correlation: ' + str(person)[:7] +
            # ' Spearman Correlation: ' + str(spearman)[:7] +
            # ' , Concordance Index: ' + str(CI)[:7])
        return rmse,person,spearman,CI
    def predict_auc(self,test_generator):
        with torch.set_grad_enabled(False):
            rmse,person, spearman, CI,y_true,y_pred  = self.test(test_generator, self.model)
        y_pred_binary = [ 1 if x >= 30 else 0 for x in y_pred ]
        y_true_binary = [ 1 if x >= 30 else 0 for x in y_true ]
        roc_score = roc_auc_score(y_true_binary, y_pred)
        precision, recall, _ = precision_recall_curve(y_true_binary, y_pred_binary)
        auprc_score = auc(recall, precision)
        accuracy = accuracy_score( y_true_binary, y_pred_binary)
        precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
        kappa = cohen_kappa_score(y_true_binary, y_pred_binary)
        print(
            'auc: ' + str(roc_score)[:7] +
            ' , aupr: ' + str(auprc_score)[:7] +
            ' acc: ' + str(accuracy)[:7] +
            ' , pcc: ' + str(precision)[:7])
        return roc_score, auprc_score, accuracy, precision, kappa
    def save_model(self):
        torch.save(self.model.state_dict(), self.modeldir + '/model.pt')
