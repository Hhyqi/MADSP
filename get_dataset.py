import numpy as np
import pandas as pd
import os
import torch.nn as nn
from datetime import datetime
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, auc, f1_score, accuracy_score, precision_score, recall_score, cohen_kappa_score
from itertools import islice 
import os
from collections import OrderedDict
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import Chem,DataStructs
import seaborn as sns
import hickle as hkl
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from rdkit.Chem.Draw import IPythonConsole
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric import data as DATA
import pickle
import torch
from transformers import BertTokenizer, AutoTokenizer,BertModel,RobertaTokenizer,RobertaModel
time_str = str(datetime.now().strftime('%y%m%d%H%M'))

#files
OUTPUT_DIR = 'results/results_loewe/'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def CalculateGraphFeat(feat_mat,adj_list):
    assert feat_mat.shape[0] == len(adj_list)
    adj_mat = np.zeros((len(adj_list), len(adj_list)), dtype='float32')
    for i in range(len(adj_list)):
        nodes = adj_list[i]
        for each in nodes:
            adj_mat[i,int(each)] = 1
    assert np.allclose(adj_mat,adj_mat.T)
    x, y = np.where(adj_mat == 1)
    adj_index = np.array(np.vstack((x, y)))
    return [feat_mat,adj_index]

def FeatureExtract(drug_feature):
    drug_data = [[] for item in range(len(drug_feature))]
    for i in range(len(drug_feature)):
        feat_mat,adj_list,_ = drug_feature.iloc[i]
        drug_data[i] = CalculateGraphFeat(feat_mat,adj_list)
    return drug_data

class GetData():
    def __init__(self):
        PATH = '/home/work/paper/baseline/MADSP-main'
        self.synergyfile    =   PATH + '/data/synergyloewe.txt'
        self.drugsmilefile  =   PATH + '/data/smiles.csv'
        self.targetfile     =   PATH + "/data/drug_protein_feature.pkl"
        self.pathwayfile    =   PATH + "/data//drug_pathway_feature.pkl"
        self.cinfofile      =   PATH + '/data/cell2id.tsv'
        self.cfeaturefile   =   PATH + '/data/oneil_cellline_feat.npy'
    # 生成摩根指纹函数
    def product_fps(self,data):
        """传入smiles编码文件列表"""
        data = [x for x in data if x is not None]
        data_mols = [Chem.MolFromSmiles(s) for s in data]
        data_mols = [x for x in data_mols if x is not None]
        data_fps = [AllChem.GetMorganFingerprintAsBitVect(x,3,1024) for x in data_mols]
        return data_fps
    
    def feature_vector(self,feature_name, df, vector_size): 
        # Jaccard Similarity
        def Jaccard(matrix):
            matrix = np.mat(matrix)
            numerator = matrix * matrix.T
            denominator = np.ones(np.shape(matrix)) * matrix.T + matrix * np.ones(np.shape(matrix.T)) - matrix * matrix.T
            return numerator / denominator

        all_feature = []
        for name in feature_name:
            all_feature.append(df[name])  # obtain all the features
        sim_matrix = Jaccard(np.array(all_feature))
        pca = PCA(n_components=vector_size)  # PCA dimension
        pca.fit(sim_matrix)
        sim_matrix = pca.transform(sim_matrix)
        return sim_matrix
    
    def create_data(self,datatype):
        if datatype == 'morgan':
            compound_iso_smiles = []
            df = pd.read_csv(self.drugsmilefile) 
            compound_iso_smiles += list(df['smile'])
            compound_iso_smiles = compound_iso_smiles
            morgan_fp = {}
            for index,smile in enumerate(compound_iso_smiles):
                morgan_fp[df['name'][index]] = list(map(int,self.product_fps([smile])[0].ToBitString())) 
            return morgan_fp
    # 归一化
    def data_format(self, data):
        data = (data - data.min()) / (data.max() - data.min())  ### minmax_normalized
        data = list(data)
        return data
    def get_typelabel(self,score):
        max = 30 
        min = 0
        if score >= 30:
            return 1
        if score < 30:
            return 0

    def get_feature(self,drug_feature,cell_feature):
        durg_dataset = {'drug_encoding':[],'label':[],'fold':[],'type':[]}
        cell_dataset = {'cell_encoding':[],'label':[],'fold':[]}
        with open(self.synergyfile, 'r') as f:
            f.readline()
            for line in f:
                drug1, drug2, cellname, score, fold = line.rstrip().split('\t')
                drug1_feature = drug_feature[drug1]
                drug2_feature = drug_feature[drug2]
                c_feature = cell_feature[cellname]
                all_feature = [drug1_feature, drug2_feature]
                durg_dataset['drug_encoding'].append(all_feature)
                durg_dataset['label'].append(float(score))
                durg_dataset['fold'].append(int(fold))
                durg_dataset['type'].append(self.get_typelabel(float(score)))
                cell_dataset['cell_encoding'].append(c_feature)
                cell_dataset['label'].append(float(score))
                cell_dataset['fold'].append(int(fold))
                all_feature2 = [drug2_feature, drug1_feature]
                durg_dataset['drug_encoding'].append(all_feature2)
                durg_dataset['label'].append(float(score))
                durg_dataset['fold'].append(int(fold))
                durg_dataset['type'].append(self.get_typelabel(float(score)))
                cell_dataset['cell_encoding'].append(c_feature)
                cell_dataset['label'].append(float(score))
                cell_dataset['fold'].append(int(fold))

    
        durg_dataset['drug_encoding'] = np.array(durg_dataset['drug_encoding'], dtype=np.float64)
        
        durg_dataset['label'] = np.array(durg_dataset['label'])
        durg_dataset['fold'] = np.array(durg_dataset['fold'])
        durg_dataset['type'] = np.array(durg_dataset['type'])
        cell_dataset['cell_encoding'] = np.array(cell_dataset['cell_encoding'])
        cell_dataset['label'] = np.array(cell_dataset['label'])
        cell_dataset['fold'] = np.array(cell_dataset['fold'])
        return durg_dataset,cell_dataset

    def slipt(self,drugdata,celldata,foldnum):
        test_fold = foldnum
        val_fold = 4    
        idx_tr = np.where(drugdata['fold'] != test_fold)
        idx_test = np.where(drugdata['fold']==test_fold)
        idx_val = np.where(drugdata['fold']!=val_fold)

    
        idx = idx_test[0]
        traindata = pd.DataFrame()
        testdata = pd.DataFrame()
        train_cldata = pd.DataFrame()
        test_cldata = pd.DataFrame()
        traindata['drug_encoding'] =[drugdata['drug_encoding'][i] for i in idx_tr[0]]
        traindata['label'] =[drugdata['label'][i] for i in idx_tr[0]]
        traindata['type'] =[drugdata['type'][i] for i in idx_tr[0]]
        traindata['index'] =range(traindata['label'].shape[0])
        train_cldata['cell_encoding'] = [celldata['cell_encoding'][i] for i in idx_tr[0]]
        train_cldata['label'] = [celldata['label'][i] for i in idx_tr[0]]
        train_cldata['index'] = range(train_cldata['label'].shape[0])
        testdata['drug_encoding'] =[drugdata['drug_encoding'][i] for i in idx]
        testdata['label'] =[drugdata['label'][i] for i in idx]
        testdata['type'] =[drugdata['type'][i] for i in idx]
        testdata['index'] =range(testdata['label'].shape[0])
        test_cldata['cell_encoding'] = [celldata['cell_encoding'][i] for i in idx]
        test_cldata['label'] = [celldata['label'][i] for i in idx]
        test_cldata['index'] = range(test_cldata['label'].shape[0])
        return traindata, train_cldata, testdata, test_cldata

    def prepare(self):
        # 得到药物特征
        durg_morgan = self.create_data('morgan')
        # drug_target = np.load(self.targetfile)
        # drug_pathway = np.load(self.pathwayfile)
        with open(self.targetfile, 'rb') as f:
            drug_target = pickle.load(f)
        with open(self.pathwayfile, 'rb') as f:
            drug_pathway = pickle.load(f)
        df = pd.read_csv(self.drugsmilefile) 
        feature_name = df["name"]
        vector_size = len(feature_name)
        drug_feature = {}
        morgan_vector = np.zeros((len(np.array(feature_name).tolist()), 0), dtype=float)
        morgan_vector = np.hstack((morgan_vector, self.feature_vector(feature_name, durg_morgan, vector_size)))
        morgan_vector = np.pad(morgan_vector, ((0, 0), (0, 1024-morgan_vector.shape[1])), mode='constant')
        # drug_target = np.pad(drug_target, ((0, 0), (0, 88-drug_target.shape[1])), mode='constant')
        # drug_pathway = np.pad(drug_pathway, ((0, 0), (0, 88-drug_pathway.shape[1])), mode='constant')
        for i in range(len(feature_name)):
            #durg_morgan[feature_name[i]] = np.pad(durg_morgan[feature_name[i]], (0, 1024-durg_morgan[feature_name[i]].shape[0]), mode='constant')
            drug_feature[feature_name[i]] =  [list(durg_morgan[feature_name[i]]),list(morgan_vector[i]),list(drug_target[feature_name[i]]),list(drug_pathway[feature_name[i]])]# 
        cell_info = pd.read_csv(self.cinfofile, sep='\t', header=0)
        c_feature = np.load(self.cfeaturefile)
        cell_feature = {}
        for i in range(len(cell_info['cell'])):
            cell_feature[cell_info['cell'][i]] = np.array(list(c_feature[i])) #

        return drug_feature,cell_feature
