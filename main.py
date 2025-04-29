# python3
# -*- coding:utf-8 -*-
import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from lifelines.utils import concordance_index
from scipy.stats import pearsonr,spearmanr
import copy
import time
from torch.utils import data
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import SequentialSampler

from prettytable import PrettyTable
from subword_nmt.apply_bpe import BPE
from get_dataset import GetData
from dsp_model import DSPSCL,data_process_loader,collate
import tensorflow as tf
import torch 
if __name__ == '__main__':
    df = GetData()
    d_feature , c_feature = df.prepare()
    durg_dataset,cell_dataset = df.get_feature(d_feature,c_feature)
    rmse_all = 0
    person_all = 0
    spearman_all = 0
    CI_all = 0
    hiddim = [8192]
    for fn in range(0,5):
        fold = fn
        traindrdata, traincldata, testdata, testcldata = df.slipt(durg_dataset,cell_dataset,fold)
        test_generator = data.DataLoader(data_process_loader(
                    testdata.index.values, testdata.label.values, testdata, testcldata,testdata.type.values),collate_fn=collate,batch_size=512)
        modeldir = 'Modelscl'
        modelfile = modeldir + '/model.pt'
        if not os.path.exists(modeldir):
            os.mkdir(modeldir)

        for hid in hiddim:
            model = DSPSCL(modeldir=modeldir,foldnum = fold,hiddim=hid,mmse = 1000)
            model.train(tr_drug=traindrdata, tr_cl=traincldata,testdata = test_generator)

        rmse,person,spearman,ci = model.predict(test_generator)
        rmse_all = rmse_all + rmse
        person_all = person_all + person
        spearman_all = spearman_all + spearman
        CI_all = CI_all + ci
        model.save_model()
        print("Model Saveed :{}".format(modelfile))
        print("Rmse:"+str(rmse)+" Person:"+str(person)+" Spearman:"+str(spearman)+" CI:"+str(ci))
    print("Rmse_AVG:"+str(rmse_all/5)+" Person_AVG:"+str(person_all/5)+" Spearman_AVG:"+str(spearman_all/5)+" CI_AVG:"+str(CI_all/5))
